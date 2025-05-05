import os
import re
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shapely.geometry as sg
import numpy as np
import pandas as pd
import openslide
from openslide import open_slide
from shapely.validation import make_valid
import traceback
import logging
import h5py


def create_polygons(df):
    polygons = []
    for _, row in df.iterrows():
        x, y = row['coord_x'], row['coord_y']
        polygon = Polygon([(x, y), (x + 512, y), (x + 512, y + 512), (x, y + 512)])
        polygons.append(polygon)
    return polygons
    

def create_dataframe(arrays):
    augmented, coords, features, zoom = arrays
    df_coords = pd.DataFrame(coords, columns=['coord_x', 'coord_y'])
    df_features = pd.DataFrame(features, columns=[f'feature_{i+1}' for i in range(features.shape[1])])

    # If 'augmented' data exists, create a DataFrame for it and concatenate it with the others
    if augmented is not None:
        df_augmented = pd.DataFrame(augmented, columns=['augmented'])
        df = pd.concat([df_coords, df_augmented, df_features], axis=1)
    else:
        df = pd.concat([df_coords, df_features], axis=1)

    if zoom is not None:
        df_zoom = pd.DataFrame(zoom, columns=['zoom'])
        df = pd.concat([df_coords, df_zoom, df_features], axis=1)
    else:
        df = pd.concat([df_coords, df_features], axis=1)

    return df


def read_h5_file(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        # print(f.keys())  # print all keys

        coords = f['coords'][:]
        feats = f['feats'][:]

        # check if "augmented" data exists, and if so, read it
        if 'augmented' in f.keys():
            augmented = f['augmented'][:]
        else:
            augmented = None

        # check if "zoom" data exists, and if so, read it
        if 'zoom' in f.keys():
            zoom = f['zoom'][:]
        else:
            zoom = None

        return augmented, coords, feats, zoom


def write_to_h5(df, h5_file_path):
    with h5py.File(h5_file_path, 'w') as f:
        # Create datasets for 'coords', 'feats' and possibly 'augmented'
        coords = f.create_dataset('coords', data=df[['coord_x', 'coord_y']].values)
        feats = f.create_dataset('feats', data=df[[col for col in df.columns if 'feature_' in col]].values)

        # If 'augmented' column exists in the DataFrame, create a dataset for it
        if 'augmented' in df.columns:
            augmented = f.create_dataset('augmented', data=df['augmented'].values)
        

def read_annotations(annon_path):
    polygons = []
    rectcoords_list = []

    with open(annon_path, 'r') as f:
        lines = f.readlines()

    headers = [h.strip() for h in lines[0].split(',')]  # Assuming CSV is comma separated
    if 'X_base' not in headers or 'Y_base' not in headers:
        raise IndexError('Unable to find "X_base" and "Y_base" columns in CSV file.')

    index_x = headers.index('X_base')
    index_y = headers.index('Y_base')

    roi_coords = []
    for line in lines[1:]:  # Skip the header
        elements = line.split(',')
        if elements[index_x] == 'X_base' or elements[index_y] == 'Y_base':
            # If we encounter a new 'X_base' or 'Y_base', save the previous polygon (if exists)
            if roi_coords and len(set(roi_coords)) >= 3:  # Ensure we have at least 3 unique points
                polygons.append(sg.Polygon(roi_coords))
                rectcoords_list.append([
                    [max(coord[0] for coord in roi_coords), min(coord[0] for coord in roi_coords)],
                    [min(coord[1] for coord in roi_coords), max(coord[1] for coord in roi_coords)]
                ])
            # Start a new polygon
            roi_coords = []
            continue
        else:
            roi_coords.append((float(elements[index_x]), float(elements[index_y])))  # Convert coordinates to numeric

    # Save the last polygon
    if roi_coords and len(set(roi_coords)) >= 3:
        polygons.append(sg.Polygon(roi_coords))
        rectcoords_list.append([
            [max(coord[0] for coord in roi_coords), min(coord[0] for coord in roi_coords)],
            [min(coord[1] for coord in roi_coords), max(coord[1] for coord in roi_coords)]
        ])

    for polygon in polygons:
        # remove invalid polygons and apply buffer
        if isinstance(polygon, sg.Polygon):
            if contains_nan_or_inf(polygon):
                polygon = fix_invalid_polygon(polygon)
                print("Invalid polygon was fixed.")
            # try to catch possible topology exceptions, e.g. due to polygon intersecting with itself
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
                print("Invalid polygon was fixed.")
            if not polygon.is_valid:
                polygon = make_valid(polygon)
                print("Invalid polygon was fixed using make_valid.")

    # Combine the individual polygons into a single MultiPolygon object
    annPolys = sg.MultiPolygon(polygons)

    return annPolys, np.int32(rectcoords_list)

def find_substring_in_list(strings, substring):
    return [s for s in strings if substring in s]


def extract_coordinates(filename):
    """Extract coordinates from filename. If coordinates were written as X,Y """
    match = re.search(r'\((\d+),(\d+)\)', filename)    # adjust here to the correct pattern!
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None

def get_slide_mpp(slide: openslide.OpenSlide) -> float:
    try:
        slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        print(f"Slide MPP successfully retrieved from metadata: {slide_mpp}")
    except KeyError:
        # Try to extract from comments
        try:
            slide_mpp = extract_mpp_from_comments(slide)
            if slide_mpp:
                print(f"MPP retrieved from comments after initial failure: {slide_mpp}")
            else:
                print(f"MPP is missing in the comments, attempting to extract from metadata...")
                slide_mpp = extract_mpp_from_metadata(slide)
                print(f"MPP re-matched from metadata after initial failure: {slide_mpp}")
        except Exception as e:
            print(f"MPP could not be loaded from the slide!") #Error: {e}")
            traceback.print_exc()
            slide_mpp = None
    return slide_mpp


def extract_mpp_from_metadata(slide: openslide.OpenSlide) -> float:
    from xml.dom.minidom import parseString
    from xml.parsers.expat import ExpatError
    try:
        # Retrieve the ImageDescription property
        xml_path = slide.properties.get('tiff.ImageDescription', None)
        if not xml_path:
            raise ValueError("No ImageDescription found in slide properties.")

        # Check and print the first 100 characters to inspect content type
        print(f"Content length: {len(xml_path)}")
        print(f"First 100 characters: {xml_path[:100]}")

        # Determine if content is XML by checking if it starts with "<"
        if xml_path.strip().startswith('<'):
            # Attempt to parse as XML
            try:
                doc = parseString(xml_path)
                collection = doc.documentElement
                images = collection.getElementsByTagName("Image")
                if not images:
                    raise ValueError("No Image tag found in XML.")

                pixels = images[0].getElementsByTagName("Pixels")
                if not pixels:
                    raise ValueError("No Pixels tag found in XML.")

                mpp = float(pixels[0].getAttribute("PhysicalSizeX"))
                print(f"MPP extracted from XML metadata: {mpp}")
                return mpp
            except ExpatError as xml_error:
                print(f"XML parsing error: {xml_error}")
                # If parsing fails, continue to try text extraction
                print("Failed to parse as XML, attempting plain text extraction.")

        mpp_pattern = r'(\d+\.\d+|\d+) ?(microns|µm|um|mpp)'
        match = re.search(mpp_pattern, xml_path, re.IGNORECASE)
        if match:
            mpp_value = match.group(1)
            mpp = float(mpp_value)
            print(f"MPP extracted from plain text metadata: {mpp}")
            return mpp
        else:
            raise ValueError("MPP value not found in the metadata text.")
    
    except Exception as e:
        print(f"Failed to extract MPP from metadata: {e}")
        traceback.print_exc()
        return None


def create_polygons_from_filenames(filenames):
    """Extract coordinates from filename. Assumes that coordinates were written as X,Y """
    polygons = []
    for filename in filenames:
        coords = extract_coordinates(filename)
        if coords:
            x, y = coords  # adjust here depending how tile fnames were written!
            polygon = Polygon([(x, y), (x + 512, y), (x + 512, y + 512), (x, y + 512)])  # (x, y) top left
            polygons.append(polygon)
    return polygons

def extract_mpp_from_comments(slide: openslide.OpenSlide) -> float:
    slide_properties = slide.properties.get('openslide.comment')
    pattern = r'<PixelSizeMicrons>(.*?)</PixelSizeMicrons>'
    match = re.search(pattern, slide_properties)
    if match:
        return match.group(1)
    else:
        return None

def contains_nan_or_inf(polygon):
    """Add checks for NaN/Inf values in polygon"""
    for x, y in polygon.exterior.coords:
        if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
            return True
    return False


def fix_invalid_polygon(polygon):
    new_coords = [(x, y) for x, y in polygon.exterior.coords if
                  not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y))]
    return sg.Polygon(new_coords)
