import multiprocessing
from multiprocessing import Manager
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
import zipfile
import h5py
import pickle
import numpy as np
import io
import argparse

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

COORD_PATTERN = re.compile(r"\(([\d\.]+),\s*([\d\.]+)\)")   # gets float coordinates e.g. 254.2541, 255.3485

def extract_coordinates(file):
    matches = COORD_PATTERN.search(str(file))
    if matches:
        return (float(np.float32(matches.group(1))),float(np.float32(matches.group(2))))
    return None

def find_feature_files(patient_id: str, features_dir: str):
    """
    Look in `features_dir` for all .h5 files that contain `patient_id` in the filename.
    Returns a list of matching Paths.
    """
    features_dir = Path(features_dir)
    matching_files = []
    for f in features_dir.glob("*.h5"):
        if patient_id in f.stem:
            matching_files.append(f)
    return matching_files


def build_h5_dict(h5_folder: str, patient_ids: list):
    """
    Build one dictionary: { patient_id: { (x,y): feature_vector, ...
    by scanning each patient's HDF5 file exactly once.
    """
    h5_map = {}
    h5_folder = Path(h5_folder)
    
    for patient_id in patient_ids:
        hf_files = find_feature_files(patient_id, features_dir)
        assert len(hf_files) > 0, f"Features file could not be found for patient {patient_id}"
        if len(hf_files) > 1:
            print(f"Found multiple slides for patient {patient_id}")

        feat_dict = {}
        for hf in hf_files:
            with h5py.File(hf, "r") as f:
                coords = f["coords"][:]
                feats = f["feats"][:]
            
            for c, feat in zip(coords, feats):
                c1, c2 = c
                key = (float(c1), float(c2))
                feat_dict[key] = feat
            
        h5_map[patient_id] = feat_dict
    return h5_map

def open_tile_from_zip(zip_path: str, tile_fname: str):
    """
    Open an image tile from a ZIP archive without extracting it to disk.
    """
    zip_path = Path(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        if tile_fname in zf.namelist(): 
            with zf.open(tile_fname) as img_file:
                img = Image.open(BytesIO(img_file.read())).convert("RGB")
            return img
        else:
            print(f"Tile {tile_fname} not found in {zip_path}")
            return None


def image_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
    """
    Convert a PIL Image to bytes.
    """
    with io.BytesIO() as buffer:
        image.save(buffer, format=format)
        return buffer.getvalue()

def worker_function(file_idx, write_metadata, write_coordinates, write_fnames, feat_maps):
    """
    Worker function for processing an image file.
    """
    idx, (file, tile_fname, pat_id) = file_idx
    img_data = open_tile_from_zip(file, tile_fname)
    assert img_data is not None, f"{tile_fname} could not be found for {pat_id}"
    #patient_id = extract_patient_id(file) if write_metadata else None
    coords = extract_coordinates(tile_fname)
    filename = str(Path(file) / tile_fname) if write_fnames else None

    # = None
    # look up the feature vector from the shared dictionary
    patient_dict = feat_maps.get(pat_id, {})
    #print(patient_dict.keys())
    feats = patient_dict.get(coords, None)
    if feats is None:
        print(f"Features couldn't be found for patient {pat_id} at {coords}")

    return idx, img_data, patient_id, coords, filename, feats

def process_images(env, paths, n_worker, write_metadata, write_coordinates, write_fnames, feat_maps):
    """
    Processes the images and stores them in LMDB.
    Optionally stores metadata, coordinates or full filenames.
    """
    files = list(enumerate(paths))
    total = 0
    worker_func = partial(worker_function, write_metadata=write_metadata, write_coordinates=write_coordinates, write_fnames=write_fnames, feat_maps=feat_maps)
    with multiprocessing.Pool(n_worker) as pool:
        for idx, img_data, patient_id, coords, fname, feats in tqdm(pool.imap_unordered(worker_func, files)):
            store_image(env, idx, img_data)
            if fname:
                store_fname(env, idx, fname)
            if patient_id:
                store_metadata(env, idx, patient_id)
            if coords:
                store_coordinates(env, idx, coords)
            if feats is not None:
                store_features(env, idx, feats)
            total += 1
        store_length(env, total)

def store_image(env, idx, img,):
    key = f"{str(idx).zfill(5)}".encode("utf-8")
    img_bytes = image_to_bytes(img, format='JPEG')
    with env.begin(write=True) as txn:
        txn.put(key, img_bytes)

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

def store_features(env, idx, feat):
    """
    Store feature vector in LMDB under a key like 'feat_00003'.
    We pickle the feature array/vector.
    """
    feat_key = f"feat_{str(idx).zfill(5)}".encode("utf-8")
    feat_data = pickle.dumps(feat, protocol=pickle.HIGHEST_PROTOCOL)
    with env.begin(write=True) as txn:
        txn.put(feat_key, feat_data)

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
        unique_patients = [extract_patient_id(path) for path in self.paths]
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


class ImageFoldersZip(Dataset):
    """
    Custom dataset class for handling image folders.
    """
    def __init__(self, folder, test_patients_file = None, exts=["jpg", "png", "tif"], split="train"):
        super().__init__()
        self.folder = folder
        self.valid_exts = exts
        self.test_patients = self._load_test_patients(test_patients_file) if test_patients_file else None

        # collect zip files
        zip_files = sorted(
            p for p in Path(folder).rglob("*.zip")
            if p.is_file() and not any(part.startswith(".") for part in p.parts)
        )

        self.paths = []    # (zip_path, internal_filename, patient_id) for each image
        patient_ids_list = []

        for zip_path in zip_files:
            # Derive patient ID from the ZIP filename:
            # e.g. "TCGA-3L-AA1B-01Z-00-DX1....zip" -> "TCGA-3L-AA1B"
            patient_id = "-".join(str(zip_path).split('/')[-1].split("-")[:3])
            
            # if test_patients_file was provided, optionally filter
            if self.test_patients is not None and split == "train":
                # skip patients that are in the test set
                if patient_id in self.test_patients:
                    continue
                else:
                    patient_ids_list.append(patient_id)
            elif self.test_patients is not None and split == "test":
                # skip patients that are not in the test set
                if patient_id not in self.test_patients:
                    continue
                else:
                    patient_ids_list.append(patient_id)

            # list the valid image tiles within this ZIP
            with zipfile.ZipFile(zip_path, "r") as zf:
                for name in zf.namelist():
                    if self._is_valid_image(name):
                        self.paths.append((zip_path, name, patient_id))

        self.paths.sort(key=lambda x: f"{x[0]}:{x[1]}")

        print(f"Found {len(zip_files)} valid zip files.")
        print(f"Number of unique patients (after filtering): {len(set(patient_ids_list))} and nr of images: {len(self.paths)}")

        if self.test_patients is not None:
            with open(f"{out_path}/patients-ids.txt", 'w') as file:
                for pat in sorted(set(patient_ids_list)):
                    file.write(f"{pat}\n")

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

    def _is_valid_image(self, filename):
        # check if the file inside the ZIP has a valid extension
        return any(filename.lower().endswith("." + ext) for ext in self.valid_exts)

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

    parser = argparse.ArgumentParser(description="MIL Crossval")
    parser.add_argument('--in_path', type=str, required=True, help='Path to the training/test images/tiles')
    parser.add_argument('--out_path', type=str, required=True, help='Output directory for created dataset')
    parser.add_argument('--test_patients_file', type=str, required=False, help='Path to the json file containing train/test split')
    parser.add_argument('--features_dir', type=str, required=False, default=None, help='Path to the features (h5 files) directory')
    parser.add_argument('--split', type=str, default="train", help='Whether lmdb dataset will be created for train or test')
    parser.add_argument('--map_size', type=int, default=140*1024**3, help='Estimated total size of the LMDB database with some buffer space')

    args = parser.parse_args()

    in_path = args.in_path
    out_path = args.out_path
    test_patients_file = args.test_patients_file
    features_dir = args.features_dir
    split = args.split
    map_size = args.map_size

    # whether to write patient IDs as metadata in lmdb dataset
    write_metadata = False

    # whether to write coordinates of the WSI patch as metadata in lmdb dataset, 
    # coordinates are determined from the filename formatted as "(x,y)"
    write_coordinates = False

    write_fnames = True
    
    num_workers = 32
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    dataset = ImageFoldersZip(folder=in_path, test_patients_file=test_patients_file, split=split)
    print(f"Total images: {len(dataset)}")

    patient_ids_in_dataset = set()
    for p in dataset.paths:
        patient_id = "-".join(str(p).split('/')[-1].split("-")[:3])
        #print(patient_id)
        patient_ids_in_dataset.add(patient_id)

    feat_maps = None
    if features_dir is not None:
        print("Building feature dictionary for all patients...")
        local_dict = build_h5_dict(features_dir, patient_ids_in_dataset)
        with open(os.path.join("temp", "create_lmdb_crc"), "wb") as f:
            pickle.dump(local_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        manager = Manager()
        feat_maps = manager.dict(local_dict)
    else:
        feat_maps = None

    # Set the map_size parameter slightly larger than the estimated total size of your 
    # LMDB database to provide some buffer space
    with lmdb.open(out_path, map_size=map_size, readahead=False) as env:
      process_images(env, paths=dataset.paths, n_worker=num_workers, write_metadata=write_metadata, write_coordinates=write_coordinates, write_fnames=write_fnames, feat_maps=feat_maps)
