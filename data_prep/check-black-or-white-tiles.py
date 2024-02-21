import os
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import Pool
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import shutil
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv('WORKSPACE_PATH')


def get_black_percentage(image):
    black_pixels = np.sum(np.array(image) == 0)
    total_pixels = image.width * image.height
    return (black_pixels / total_pixels) * 100.0


def process_file(args):
    filepath, dest_path = args
    try:
        img = Image.open(filepath)

        black_percentage = get_black_percentage(img)

        if black_percentage >= 10.0:
            shutil.move(filepath, os.path.join(dest_path, os.path.basename(filepath)))
    except Exception as e:
        print(f"Can't read file {filepath}. Error: {str(e)}")
        shutil.move(filepath, os.path.join(dest_path, os.path.basename(filepath)))


def is_above_or_under_threshold(image_data, threshold=40, how="under"):
    if how == "under":
        return np.max(image_data) < threshold
    else: 
        return np.max(image_data) > threshold


def process_image(filepath, threshold):
    try:
        with Image.open(filepath) as img:
            image_data = np.array(img)

        # Check if the image is almost black and move it if it is
        if is_above_or_under_threshold(image_data, threshold=threshold):
            destination_filepath = os.path.join(dest_path, os.path.basename(filepath))
            shutil.move(filepath, destination_filepath)
            return True
    except Exception as e:
        return False
    return None


if __name__ == "__main__":

    path = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-MSI/TCGA-CRC-only-tumor-tiles-msi-all"

    # MOVING TILES THAT ARE ALMOST BLACK
    """
    dest_path = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-MSI/TCGA-CRC-only-tumor-tiles-msi-white"

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    filepaths = [os.path.join(subdir, file) for subdir, _, files in os.walk(path) for file in files if file.endswith('.jpg')]
    args_list = [(filepath, dest_path) for filepath in filepaths]
    threshold = 40

    # moving all tiles that are almost black
    with Pool(6) as p:
        list(tqdm(p.imap(process_file, args_list, threshold), total=len(filepaths), desc="Processing files"))

    image_files = []
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, filename)
                image_files.append(filepath)

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_image, image_files), total=len(image_files)))

    successful_moves = results.count(True)
    print(f"Successfully moved {successful_moves} tiles.")
    """
    # MOVING TILES THAT ARE TOO WHITE (CONTAIN TOO MUCH BACKGROUND)

    dest_path = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-only-tumor-tiles-msi-white"

    image_files = []
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, filename)
                image_files.append(filepath)

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_image, image_files), total=len(image_files)))

    successful_moves = results.count(True)
    print(f"Successfully moved {successful_moves} images.")

    # MOVING TILES THAT CONTAIN NOT ENOUGH TISSUE
    path = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-MSI/TCGA-CRC-only-tumor-tiles-msi-all"
    dest_path = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-MSI/removed/not-enough-tissue"



