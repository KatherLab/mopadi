import os
import re
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shapely.geometry as sg
import numpy as np
import openslide
from openslide import open_slide
from shapely.affinity import scale
import logging
import shutil
from tqdm import tqdm
from utils import *
from shapely import speedups
import rtree


def is_within_bounds(tile, region):
    """
    Check if the bounding box of the tile overlaps with the bounding box of the region.
    """
    # tile and region are both bounding boxes in the format ((x1, y1), (x2, y2))
    tile_top_left, tile_bottom_right = tile
    region_top_left, region_bottom_right = region
    return not (tile_top_left[0] > region_bottom_right[0] or tile_bottom_right[0] < region_top_left[0] or tile_top_left[1] > region_bottom_right[1] or tile_bottom_right[1] < region_top_left[1])


def main():
    correctly_extracted_slide_nr = 0
    skipped_slides = 0
    roi_dir = '/home/laura/data/TCGA-CRC/CRC_csv_annotations'
    roi_fnames = os.listdir(roi_dir)

    slide_dir = '/home/laura/data/TCGA-CRC/TCGA-CRC-DX-IMGS'
    slides_fnames = os.listdir(slide_dir)

    # these are the folders containing tiles from each slide
    img_dir = '/home/laura/data/TCGA-CRC/TCGA-CRC-tiles_512x512'
    img_fnames = os.listdir(img_dir)

    save_dir = '/home/laura/data/TCGA-CRC/TCGA-CRC-tiles_512x512-only-tumor-tiles'
    os.makedirs(save_dir, exist_ok=True)

    for fname in tqdm(slides_fnames, desc="Iterating through slides..."):

        tiles_folder = fname.split('.svs')[0]
        print(f'Tiles folder: {tiles_folder}')
        new_dir = os.path.join(save_dir, tiles_folder)
        if os.path.exists(new_dir):
            print('Tiles have been already processed for this slide, skipping...')
            continue


        slide = open_slide(os.path.join(slide_dir, fname))

        try:
            slide_mpp = get_slide_mpp(slide)
        except Exception as err:
            print('Could not find MPP, the slide will be skipped')
            skipped_slides = skipped_slides + 1
            continue
            
        if slide_mpp is None:
            print('Could not find MPP, the slide will be skipped')
            skipped_slides = skipped_slides + 1
            continue
        tile_fnames = os.listdir(os.path.join(img_dir, tiles_folder))
        polygons = create_polygons_from_filenames(os.listdir(os.path.join(img_dir, tiles_folder)))
        tiles_with_polygons = list(zip(tile_fnames, polygons))
        
        try:
            roi_fname = find_substring_in_list(roi_fnames, fname.split('.svs')[0])
            if len(roi_fname)>1:
                print('Found multiple csv files. Taking just the 1st one.')
            elif len(roi_fname)==0:
                print('Could not find corresponding CSV file; the slide will be skipped...')
                skipped_slides = skipped_slides + 1
                continue
            elif len(roi_fname)==1:
                print(f'Corresponding annotations file was found: {roi_fname[0]}')
        except Exception as err:
            print(f"Exception during CSV file reading: {err}; the slide will be skipped")
            skipped_slides = skipped_slides + 1
            continue

        try:
            ann, _ = read_annotations(os.path.join(roi_dir, roi_fname[0]))
        except Exception as err:
            print(f"Exception during reading of polygons occured: {err}. Most likely the CSV file is empty. This slide will be skipped...")
            skipped_slides = skipped_slides + 1
            continue

        target_mpp = 256/224
        scale_factor = slide_mpp / target_mpp
        scaled_annPolys = scale(ann, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
        
        fig, ax = plt.subplots()
        if isinstance(scaled_annPolys, sg.MultiPolygon):
            for i, polygon in enumerate(scaled_annPolys.geoms):
                x, y = polygon.exterior.xy
                ax.fill(x, y, alpha=0.5, fc='r', label=f'Annotated Tissue')
        else:
            x, y = scaled_annPolys.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='r', ec='Annotated Tissue')

        for tile in polygons:
            x, y = tile.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='b', ec='none', label='Extracted Tiles')

        legend_elements = [Patch(facecolor='red', edgecolor='r', alpha=0.5, label='Annotated Tissue'),
                           Patch(facecolor='blue', edgecolor='b', alpha=0.5, label='Extracted Tiles')]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.invert_yaxis()
        plt.savefig(f'plots-tcga-crc-512x512/{fname.split(".svs")[0]}.png')
        plt.close()
        
        # Insert polygons into R-tree spatial index
        index = rtree.index.Index()
        for idx, polygon in enumerate(polygons):
            index.insert(idx, polygon.bounds)        
        
        os.makedirs(new_dir, exist_ok=True) 
        
        
        for tile, tile_polygon in tqdm(tiles_with_polygons, total=len(tiles_with_polygons), desc="Checking tiles..."):
            try:
                if index.count(tile_polygon.bounds) > 0:
                    if isinstance(scaled_annPolys, Polygon) and tile_polygon.intersects(scaled_annPolys):
                        intersection = scaled_annPolys.intersection(tile_polygon)
                        if (intersection.area / tile_polygon.area) >= 0.6 and not os.path.exists(os.path.join(new_dir, tile)):
                            shutil.copy(os.path.join(img_dir, tiles_folder, tile), new_dir)
                    elif isinstance(scaled_annPolys, MultiPolygon):
                        for polygon in scaled_annPolys.geoms:
                            if tile_polygon.intersects(polygon):
                                intersection = polygon.intersection(tile_polygon)
                                if (intersection.area / tile_polygon.area) >= 0.6 and not os.path.exists(os.path.join(new_dir, tile)):
                                    shutil.copy(os.path.join(img_dir, tiles_folder, tile), new_dir)
            except Exception as err:
                print(f"There was an error with a polygon, which was not caught by existing checks, skipping this tile. Error: {err}")
                continue
        
                
        correctly_extracted_slide_nr = correctly_extracted_slide_nr + 1
        
    print(f'Could correctly process {correctly_extracted_slide_nr} slides; skipped slides: {skipped_slides}')


if __name__ == '__main__':
    main()
    