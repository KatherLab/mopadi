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
import concurrent.futures
import shutil
from tqdm import tqdm
from data_prep.utils import *
from shapely import speedups
from matplotlib.patches import Polygon as MatplotlibPolygon
import rtree
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


def process_slide(fname, slide_dir, img_dir, save_dir, roi_dir, target_mpp, generate_plots):

    tiles_folder = fname.split('.')[0]
    #print(f'Tiles folder: {tiles_folder}')
    filtered_tiles_dir = os.path.join(save_dir, tiles_folder)

    if os.path.exists(filtered_tiles_dir) and len(os.listdir(filtered_tiles_dir)) > 0:
        #print(f'Tiles have been already processed for {tiles_folder} slide, skipping...')
        return

    tiles_dir = os.path.join(img_dir, tiles_folder)
    if not os.path.exists(os.path.join(img_dir, tiles_folder)):
        #print("Tiles folder could not be found, likely due to missing MPP, skipping...")
        return

    tile_fnames = os.listdir(tiles_dir)

    os.makedirs(filtered_tiles_dir, exist_ok=True)
    slide = open_slide(os.path.join(slide_dir, fname))

    try:
        slide_mpp = get_slide_mpp(slide)
    except Exception as err:
        print(f'Could not find MPP, the slide {tiles_folder} will be skipped: {err}')
        return False

    if slide_mpp is None:
        print(f'Could not find MPP, the slide {tiles_folder} will be skipped')
        return False

    polygons = create_polygons_from_filenames(tile_fnames)
    tiles_with_polygons = list(zip(tile_fnames, polygons))

    # find corresponding annotations file
    try:
        roi_fname = find_substring_in_list(os.listdir(roi_dir), fname.split('.svs')[0])
        if len(roi_fname) > 1:
            print('Found multiple csv files. Taking just the 1st one.')
        elif len(roi_fname) == 0:
            print('Could not find corresponding CSV file; the slide will be skipped...')
            return False
    except Exception as err:
        print(f"Exception during CSV file reading: {err}; the slide will be skipped")
        return False

    # read and scale annotations thath were made in the original WSI resolution
    try:
        ann, _ = read_annotations(os.path.join(roi_dir, roi_fname[0]))
        scale_factor = slide_mpp / target_mpp
        scaled_annPolys = scale(ann, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
    except Exception as err:
        print(f"Error reading or scaling annotations for slide {tiles_folder}: {err}")
        return False

    if generate_plots:
        os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True) 

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

            minx, miny, maxx, maxy = tile.bounds
            ax.add_patch(MatplotlibPolygon([[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]], fill=None, edgecolor='g', linestyle='--'))


        legend_elements = [Patch(facecolor='red', edgecolor='r', alpha=0.5, label='Annotated Tissue'),
                        Patch(facecolor='blue', edgecolor='b', alpha=0.5, label='All Tiles')]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.invert_yaxis()

        plt.savefig(os.path.join(save_dir, "plots", f'{fname.split(".")[0]}.png'))
        plt.close()

    # insert polygons from fnames of tiles into the R-tree spatial index for faster intersection lookup
    index = rtree.index.Index()
    for idx, polygon in enumerate(polygons):
        index.insert(idx, polygon.bounds)

    for tile_fname, tile_polygon in tqdm(tiles_with_polygons, total=len(tiles_with_polygons)):
        process_tile((tile_fname, tile_polygon, scaled_annPolys, index, img_dir, tiles_folder, filtered_tiles_dir, polygons))

    return True


def process_tile(tile_data):
    tile_fname, tile_polygon, scaled_annPolys, index, img_dir, tiles_folder, filtered_tiles_dir, polygons = tile_data    

    try:
        #intersection = scaled_annPolys.intersection(tile_polygon) #for debugging
        #print(f"Tile ID: {tile_fname}, Intersection Area: {intersection.area}, Tile Area: {tile_polygon.area}, Ratio: {intersection.area / tile_polygon.area}")
        intersecting_ids = list(index.intersection(scaled_annPolys.bounds))
        #print(f"Tile bounds: {tile_polygon.bounds}, Intersecting IDs: {intersecting_ids}")
        if len(intersecting_ids) > 0 and index.count(tile_polygon.bounds) > 0:
            if isinstance(scaled_annPolys, Polygon) and tile_polygon.intersects(scaled_annPolys):
                intersection = scaled_annPolys.intersection(tile_polygon)
                #print(f"Intersection area: {intersection.area}, Tile area: {tile_polygon.area}")

                if (intersection.area / tile_polygon.area) >= 0.6 and not os.path.exists(os.path.join(filtered_tiles_dir, tile_fname)):
                    shutil.copy(os.path.join(img_dir, tiles_folder, tile_fname), filtered_tiles_dir)
                    #print(f"{os.path.join(img_dir, tiles_folder, tile_fname)} copied to {filtered_tiles_dir}")

            elif isinstance(scaled_annPolys, MultiPolygon):
                for polygon in scaled_annPolys.geoms:
                    if tile_polygon.intersects(polygon):
                        intersection = polygon.intersection(tile_polygon)
                        #print(f"Intersection area: {intersection.area}, Tile area: {tile_polygon.area}")

                        if (intersection.area / tile_polygon.area) >= 0.6 and not os.path.exists(os.path.join(filtered_tiles_dir, tile_fname)):
                            shutil.copy(os.path.join(img_dir, tiles_folder, tile_fname), filtered_tiles_dir)
                            #print(f"{os.path.join(img_dir, tiles_folder, tile_fname)} copied to {filtered_tiles_dir}")
        else:
            print(f"Length of intersecting_ids is 0: No bounding box intersection found for tile {tile_fname}")
    except Exception as err:
        print(f"There was an error with a polygon, which was not caught by existing checks, skipping this tile. Error: {err}")


def main():

    target_mpp = 256/512
    num_workers = 8
    generate_plots = True

    #roi_dir = f"{ws_path}/data/TCGA-BRCA/BRCA-csv-annotations"
    roi_dir = f"{ws_path}/data/TCGA-CRC/csv_annotations"
    roi_fnames = os.listdir(roi_dir)

    slide_dir = '/mnt/copernicus1/PATHOLOGY/others/public/TCGA/TCGA-CRC-DX-IMGS/data-CRC'
    #slide_dir = "/mnt/copernicus1/PATHOLOGY/others/public/TCGA/TCGA-BRCA-DX-IMGS/data-BRCA"
    slides_fnames = os.listdir(slide_dir)

    # BRCA
    #img_dir = f"{ws_path}/data/TCGA-BRCA/tiles"
    #img_dir = f"{ws_path}/data/TCGA-BRCA/tiles-test"

    # CRC
    img_dir = '/mnt/bulk-ganymede/laura/deep-liver/data/TCGA-CRC/tiles_512x512_05mpp'

    img_fnames = os.listdir(img_dir)

    #save_dir = f'{ws_path}/data/TCGA-BRCA/tiles_512x512_only_tumor-test'
    save_dir = '/mnt/bulk-ganymede/laura/deep-liver/data/TCGA-CRC/tiles_512x512_05mpp-only-tum'
    os.makedirs(save_dir, exist_ok=True)

    correctly_extracted_slide_nr = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_slide, 
                fname, 
                slide_dir, 
                img_dir, 
                save_dir, 
                roi_dir, 
                target_mpp, 
                generate_plots
            ): fname 
            for fname in slides_fnames
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing slides..."):
            slide_name = futures[future]
            try:
                if future.result():
                    correctly_extracted_slide_nr += 1
            except Exception as exc:
                print(f'Slide {slide_name} generated an exception: {exc}')

    print(f'Correctly processed {correctly_extracted_slide_nr} slides.')

if __name__ == "__main__":
    main()
    