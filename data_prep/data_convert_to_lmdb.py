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
import json

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
    def __init__(self, folder, test_patients_file = None, exts=["jpg", "png", "tif"]):
        super().__init__()
        self.folder = folder
        self.test_patients = self._load_test_patients(test_patients_file) if test_patients_file else None

        self.paths = sorted(
            [
                p for ext in exts 
                for p in Path(folder).glob(f"**/*.{ext}")
                if 
                #'Tile' in (p.stem) and 
                not any(part.startswith(".") for part in p.parts)
            ]
        )
        unique_patients = ["-".join(str(path).split("/")[-2].split("-")[:3]) for path in self.paths]
        print(f"Total unique patients: {len(set(unique_patients))} and nr of images: {len(self.paths)}")

        if self.test_patients:
            self.paths = [
                p for p in self.paths 
                if "-".join(str(p).split("/")[-2].split("-")[:3]) in self.test_patients #not in self.test_patients
                #and "Liver_hepatocellular_carcinoma" in str(p) or "Cholangiocarcinoma" in str(p)
            ]
            #print(self.paths)
            print(f"Number of images to be put in the lmdb: {len(self.paths)}")

            unique_patients = ["-".join(str(path).split("/")[-2].split("-")[:3]) for path in self.paths]
            print(f"Total unique patients left after filtering: {len(set(unique_patients))}")

            with open(f"{out_path}/patients-ids.txt", 'w') as file:
                for pid in sorted(set(unique_patients)):
                    file.write(f"{pid}\n")

        return

    def _load_test_patients(self, test_patients_file):
        print(f"Data split file given, loading test patients from {test_patients_file}")
        with open(test_patients_file, 'r') as file:
            data = json.load(file)
            test_patients = {patient for patient in data['Test set patients']}
            #test_patients = set()
            #for cancer_type, details in data.items():
            #    patients = details.get("Test set patients", [])
            #    test_patients.update(patients)
            print(f"Number of patients in the set: {len(test_patients)}")
        return test_patients

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
    #map_size=140 * 1024**3
    #in_path = "/mnt/bulk-dgx/laura/mopadi/data/TCGA_CRC_tiles_512x512_only_tumor-all"
    #out_path = "/mnt/bulk-dgx/laura/mopadi/datasets/tcga_crc_512_lmdb-test"
    #test_patients_file = f"/mnt/bulk-mars/laura/diffae/data/TCGA-CRC/new_split/data_info.json"

    # BRAIN 224x224 ----------------------------------------------------------------------------------------
    in_path = f"{ws_path}/data/brain/brain-full-dataset-cache"
    out_path = f"{ws_path}/mopadi/datasets/brain/brain-train-new"
    test_patients_file = f"{ws_path}/data/brain/test_train_split.txt"
    map_size = 140 * 1024**3

    # JAPAN 224x224 --------------------------------------------------------------------------------------
    #in_path = f"/mnt/bulk-mars/laura/diffae/data/japan/new-all"
    #out_path = f"/mnt/bulk-ganymede/laura/diffae/diffae/datasets/japan-lmdb-train-new"
    #test_patients_file = f"/mnt/bulk-mars/laura/diffae/data/japan/test_train_split.txt"
    #map_size = 140 * 1024**3

    # LUNG 224x224 TRAIN -------------------------------------------------------------------------------------
    # in_path = f"{ws_path}/data/lung/train"
    # out_path = f"{ws_path}/mopadi/datasets/lung/subtypes-lmdb-train"

    # LUNG 224x224 VAL ---------------------------------------------------------------------------------------
    # in_path = f"{ws_path}/data/lung/val"
    # out_path = f"{ws_path}/mopadi/datasets/lung/subtypes-lmdb-val"
    # map_size = 60 * 1024**3

    # LIVER SUBTYPES 224x224 TRAIN ---------------------------------------------------------------------------------------
    # in_path = f"{ws_path}/data/liver_types/test"
    # out_path = f"{ws_path}/mopadi/datasets/liver/types-lmdb-test"
    # map_size = 60 * 1024**3

    # CPTAC LUAD EXT VAL ---------------------------------------------------------------------------------------
    #in_path = f"{ws_path}/data/cache-CPTAC-LUAD"
    #out_path = f"{ws_path}/mopadi/datasets/lung/cptac-luad-ext-val-lmdb"
    #map_size = 100 * 1024**3

    # whether to write patient IDs as metadata in lmdb dataset
    write_metadata = False

    # whether to write coordinates of the WSI patch as metadata in lmdb dataset, 
    # coordinates are determined from the filename formatted as "(x,y)"
    write_coordinates = False

    write_fnames = True
    
    num_workers = 8
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    dataset = ImageFolder(in_path, test_patients_file)
    print(f"Total images: {len(dataset)}")

    # Set the map_size parameter slightly larger than the estimated total size of your 
    # LMDB database to provide some buffer space
    with lmdb.open(out_path, map_size=map_size, readahead=False) as env:
      process_images(env, paths=dataset.paths, n_worker=num_workers, write_metadata=write_metadata, write_coordinates=write_coordinates, write_fnames=write_fnames)
