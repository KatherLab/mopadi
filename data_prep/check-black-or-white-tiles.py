import os
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import Pool
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import shutil
from dotenv import load_dotenv
from functools import partial

load_dotenv()
ws_path = os.getenv('WORKSPACE_PATH')


def get_black_percentage(image):
    black_pixels = np.sum(np.array(image) == 0)
    total_pixels = image.width * image.height
    return (black_pixels / total_pixels) * 100.0


def process_file_to_remove_black_tiles(args):
    filepath, dest_path = args
    patient_name = filepath.split('/')[-2]

    try:
        img = Image.open(filepath)
        black_percentage = get_black_percentage(img)

        if black_percentage >= 10.0:

            if not os.path.exists(os.path.join(dest_path, patient_name)):
                os.makedirs(os.path.join(dest_path, patient_name), exist_ok=True)

            shutil.move(filepath, os.path.join(dest_path, patient_name, os.path.basename(filepath)))
    except Exception as e:
        print(f"Can't read file {filepath}. Error: {str(e)}")

        if not os.path.exists(os.path.join(dest_path, patient_name)):
            os.makedirs(os.path.join(dest_path, patient_name), exist_ok=True)
        shutil.move(filepath, os.path.join(dest_path, patient_name, os.path.basename(filepath)))


def is_above_or_under_threshold(image_data, threshold, how):
    if how == "under":
        return np.max(image_data) < threshold
    else: 
        return np.mean(image_data) > threshold


def process_image(filepath, dest_path, threshold=40, how="under"):
    try:
        with Image.open(filepath) as img:
            image_data = np.array(img)

        # Check if the image is almost black and if so move it
        if is_above_or_under_threshold(image_data, threshold, how):
            patient_name = filepath.split('/')[-2]
            if not os.path.exists(os.path.join(dest_path, patient_name)):
                os.makedirs(os.path.join(dest_path, patient_name))
                
            destination_filepath = os.path.join(dest_path, patient_name, os.path.basename(filepath))
            shutil.move(filepath, destination_filepath)
            return True
    except Exception as e:
        return False
    return None


if __name__ == "__main__":

    path = f"/mnt/bulk-ganymede/laura/deep-liver/data/TCGA-CRC/tiles_512x512_05mpp"

    # MOVING TILES THAT ARE WEIRD (% OF BLACK  PIXELS > 10)
    dest_path = f"/mnt/bulk-ganymede/laura/deep-liver/data/TCGA-CRC/trash/almost_black"

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    """
    filepaths = [os.path.join(subdir, file) for subdir, _, files in os.walk(path) for file in files if file.endswith('.jpg')]
    args_list = [(filepath, dest_path) for filepath in filepaths]

    with Pool(16) as p:
       list(tqdm(p.imap(process_file_to_remove_black_tiles, args_list), total=len(filepaths), desc="Processing files"))
    """

    image_files = []
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, filename)
                image_files.append(filepath)

    dest_path = f"/mnt/bulk-ganymede/laura/deep-liver/data/TCGA-CRC/trash/almost_white"
    threshold=220

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    partial_process_image = partial(process_image, dest_path=dest_path, threshold=threshold, how="over")

    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(partial_process_image, image_files), total=len(image_files)))

    successful_moves = results.count(True)
    print(f"Successfully moved {successful_moves} tiles.")
