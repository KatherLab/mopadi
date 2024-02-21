import multiprocessing
from functools import partial
from io import BytesIO
from pathlib import Path
from torch.utils.data import Dataset
import lmdb
from PIL import Image
from tqdm import tqdm
import os
import re
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


def open_image(file):
    """
    Opens an image and returns its byte representation.
    """
    img = Image.open(file)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()

def extract_patient_id(file):
    """
    Extracts the patient ID from the image file path.
    """
    file_parts = os.path.basename(os.path.dirname(file)).split("-")
    return "-".join(file_parts[:3])

COORD_PATTERN = re.compile(r"\((\d+),(\d+)\)")

def extract_coordinates(file):
    matches = COORD_PATTERN.search(str(file))
    if matches:
        return (int(matches.group(1)), int(matches.group(2)))
    return None

def worker_function(file_idx, write_metadata, write_coordinates, write_fnames):
    """
    Worker function for processing an image file.
    Opens the image and extracts the patient ID from the file path.
    """
    idx, file = file_idx
    img_data = open_image(file)
    patient_id = extract_patient_id(file) if write_metadata else None
    coords = extract_coordinates(file) if write_coordinates else None
    filename = str(file) if write_fnames else None
    return idx, img_data, patient_id, coords, filename

def process_images(env, paths, n_worker, write_metadata, write_coordinates, write_fnames):
    """
    Processes the images and stores them in LMDB.
    Optionally stores metadata, coordinates or full filenames.
    """
    files = list(enumerate(paths))
    total = 0
    worker_func = partial(worker_function, write_metadata=write_metadata, write_coordinates=write_coordinates, write_fnames=write_fnames)
    with multiprocessing.Pool(n_worker) as pool:
        for idx, img_data, patient_id, coords, fname in tqdm(pool.imap_unordered(worker_func, files)):
            store_image(env, idx, img_data)
            if fname:
                store_fname(env, idx, fname)
            if patient_id:
                store_metadata(env, idx, patient_id)
            if coords:
                store_coordinates(env, idx, coords)
            total += 1
        store_length(env, total)

def store_image(env, idx, img,):
    key = f"{str(idx).zfill(5)}".encode("utf-8")
    with env.begin(write=True) as txn:
        txn.put(key, img)

def store_fname(env, idx, fname):
    filename_key = f"filename_{str(idx).zfill(5)}".encode("utf-8")
    with env.begin(write=True) as txn:
        txn.put(filename_key, fname.encode("utf-8"))

def store_metadata(env, idx, patient_id):
    meta_key = f"meta_{str(idx).zfill(5)}".encode("utf-8")
    with env.begin(write=True) as txn:
        txn.put(meta_key, patient_id.encode("utf-8"))

def store_coordinates(env, idx, coords):
    coord_key = f"coord_{str(idx).zfill(5)}".encode("utf-8")
    coord_str = f"{coords[0]},{coords[1]}".encode("utf-8")
    with env.begin(write=True) as txn:
        txn.put(coord_key, coord_str)

def store_length(env, total):
    with env.begin(write=True) as txn:
        txn.put("length".encode("utf-8"), str(total).encode("utf-8"))

class ImageFolder(Dataset):
    """
    Custom dataset class for handling image folders.
    """
    def __init__(self, folder, exts=["jpg", "png", "tif"]):
        super().__init__()
        self.folder = folder
        self.paths = sorted(p for ext in exts for p in Path(folder).glob(f"**/*.{ext}")
                    if not any(part.startswith(".") for part in p.parts))
        return

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return img


if __name__ == "__main__":
    """
    Converting data to lmdb format dataset for usage during training and generation of results.
    """

    # TCGA CRC 512x512 ------------------------------------------------------------------------------------
    # map_size=10056487220
    # in_path = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-tiles_512x512-only-tumor-tiles"
    # out_path = f"{ws_path}/mopadi/datasets/tcga/tcga_crc_512.lmdb"

    # TCGA CRC 224x224 only with MSIH clini info ----------------------------------------------------------
    # in_path = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-MSI/TCGA-CRC-only-tumor-tiles-msi-train"
    # out_path = f"{ws_path}/mopadi/datasets/tcga/tcga_crc-msi-feb-16-train-lmdb"

    in_path = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-MSI/TCGA-CRC-only-tumor-tiles-msi-val"
    out_path = f"{ws_path}/mopadi/datasets/tcga/tcga_crc-msi-feb-16-val-lmdb"
    map_size = 80 * 1024**3

    # TCGA CRC 224x224 only with BRAF clini info ----------------------------------------------------------
    # in_path = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-only-tumor-BRAF/TCGA-CRC-only-tumor-BRAF-train"
    # out_path = f"{ws_path}/mopadi/datasets/tcga/tcga_crc-braf-train-lmdb"
    # map_size = 40 * 1024**3

    # BRAIN 224x224 ----------------------------------------------------------------------------------------
    # in_path = f"{ws_path}/data/brain"
    # out_path = f"{ws_path}/mopadi/datasets/brain"
    # map_size = 10 * 1024**3

    # JAPAN 224x224 TRAIN -----------------------------------------------------------------------------------
    # in_path = f"{ws_path}/data/japan/train"
    # out_path = f"{ws_path}/mopadi/datasets/japan/japan-lmdb"
    # map_size = 140 * 1024**3

    # JAPAN 224x224 VAL --------------------------------------------------------------------------------------
    # in_path = f"{ws_path}/data/japan/val"
    # out_path = f"{ws_path}/mopadi/datasets/japan/japan-lmdb-val"

    # LUNG 224x224 TRAIN -------------------------------------------------------------------------------------
    # in_path = f"{ws_path}/data/lung/train"
    # out_path = f"{ws_path}/mopadi/datasets/lung/subtypes-lmdb-train"

    # LUNG 224x224 VAL ---------------------------------------------------------------------------------------
    # in_path = f"{ws_path}/data/lung/val"
    # out_path = f"{ws_path}/mopadi/datasets/lung/subtypes-lmdb-val"

    # map_size = 60 * 1024**3


    # whether to write patient IDs as metadata in lmdb dataset
    write_metadata = False

    # whether to write coordinates of the WSI patch as metadata in lmdb dataset, 
    # coordinates are determined from the filename formatted as "(x,y)"
    write_coordinates = False

    write_fnames = True
    
    num_workers = 16
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    dataset = ImageFolder(in_path)
    print(f"Total images: {len(dataset)}")

    # Set the map_size parameter slightly larger than the estimated total size of your 
    # LMDB database to provide some buffer space
    with lmdb.open(out_path, map_size=map_size, readahead=False) as env:
      process_images(env, paths=dataset.paths, n_worker=num_workers, write_metadata=write_metadata, write_coordinates=write_coordinates, write_fnames=write_fnames)
