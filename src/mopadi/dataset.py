import os, re, io
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import struct
import pickle
import json
import zipfile
import random
import hashlib
import h5py
from itertools import islice
import webdataset as wds
from typing import Dict, Optional, List, Union
from collections import defaultdict, OrderedDict

import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms

from mopadi.utils.dist_utils import *

from dotenv import load_dotenv
load_dotenv()
ws_path = os.getenv('WORKSPACE_PATH')

IMAGE_KEYS = ("png", "jpg", "jpeg", "tif", "tiff")


class SubsetDataset(Dataset):
    def __init__(self, dataset, size):
        assert len(dataset) >= size
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        assert index < self.size
        return self.dataset[index]
    
class TakeIterableDataset(IterableDataset):
    """Take first `size` samples from an iterable dataset."""
    def __init__(self, dataset, size: int):
        assert isinstance(dataset, IterableDataset) or not hasattr(dataset, "__getitem__")
        self.dataset = dataset
        self.size = size

    def __iter__(self):
        # yield exactly `size` samples (will raise StopIteration early if not enough)
        return islice(iter(self.dataset), self.size)
        

def compute_root_dirs_signature(root_dirs):
    """
    Compute a hash signature of root_dirs based on their absolute paths and modification times.
    """
    m = hashlib.sha256()
    for root in sorted(root_dirs):
        abspath = os.path.abspath(root)
        m.update(abspath.encode('utf-8'))
        try:
            mtime = str(os.path.getmtime(abspath))
            m.update(mtime.encode('utf-8'))
        except Exception:
            pass  # If the folder doesn't exist, just skip
    return m.hexdigest()

def calculate_sampling_ratio(cohort_sizes, max_tiles_per_patient, cohort_size_threshold):
    """
    Calculate sampling ratio based on cohort sizes.
    
    Args:
        cohort_sizes (dict): Cohort sizes as {cohort_id: num_tiles}.
    
    Returns:
        dict: Sampling rule per cohort (None for full sampling, 1024 for limited sampling).
    """
    print(f"Will sample {max_tiles_per_patient} per patient for bigger cohorts than {cohort_size_threshold}")
    sampling_ratios = {}
    for cohort, size in cohort_sizes.items():
        if size > cohort_size_threshold:
            sampling_ratios[cohort] = max_tiles_per_patient
        else:
            sampling_ratios[cohort] = None
    return sampling_ratios


def load_test_patients(test_patients_file):
    """Load test patient IDs from JSON file."""
    print(f"Loading test patients from {test_patients_file}")
    with open(test_patients_file, 'r') as file:
        data = json.load(file)
        test_patients = {patient for patient in data['Test set patients']}
    print(f"Number of patients in the test set: {len(test_patients)}")
    return test_patients


def get_tiles_from_zip(zip_path):
    """Extract tile names from a ZIP file."""
    exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        return [name for name in zf.namelist() if name.lower().endswith(exts)]


def load_or_calculate_cohort_sizes(root_dirs, process_only_zips, cache_file, force_recalculate=False):
    """
    Load cohort sizes from a cache or calculate if cache is missing or recalculation is forced.
    This is basically needed so the scanning of directories would not be needed to do multiples times, 
    e.g., when debugging, because it can get time consuming with multiple large cohorts.
    """
    signature = compute_root_dirs_signature(root_dirs)

    if cache_file:
        if os.path.exists(cache_file) and not force_recalculate:
            print(f"Cached cohort sizes file: {cache_file}")
            with open(cache_file, 'r') as file:
                cache = json.load(file)
                cached_sig = cache.get('root_dirs_signature')
                if cached_sig == signature:
                    print(f"Loading cached cohort sizes...")
                    cohort_sizes = cache['cohort_sizes']
                    cohort_sizes_wsi = cache['cohort_sizes_wsi']
                    return cohort_sizes, cohort_sizes_wsi
                else:
                    print(f"Data directories have changed! Cache invalid: {cache_file}")

    print("Calculating cohort sizes...")
    cohort_sizes = defaultdict(int)
    cohort_sizes_wsi = defaultdict(int)

    for root_dir in tqdm(root_dirs):
        print(f"Scanning {root_dir}")
        cohort_id = root_dir.split('-')[-1]
        wsi_count = 0
        for patient_folder in tqdm(os.listdir(root_dir)):
            patient_path = os.path.join(root_dir, patient_folder)

            if process_only_zips and patient_folder.lower().endswith('.zip'):
                num_tiles = len(get_tiles_from_zip(patient_path))
                wsi_count+=1
            elif os.path.isdir(patient_path) and not process_only_zips:
                exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
                num_tiles = len([f for f in os.listdir(patient_path) if f.lower().endswith(exts)])
            else:
                continue

            cohort_sizes[cohort_id] += num_tiles
        cohort_sizes_wsi[cohort_id] = wsi_count

    cache_data = {
    'root_dirs_signature': signature,
    'cohort_sizes': cohort_sizes,
    'cohort_sizes_wsi': cohort_sizes_wsi,
    }

    if cache_file:
        with open(cache_file, 'w') as file:
            json.dump(cache_data, file, indent=1)
        print(f"Calculated cohort sizes saved to {cache_file}")

    return cohort_sizes, cohort_sizes_wsi

def load_tile_paths_cache(root_dirs, cache_pickle_tiles_path):
    if os.path.exists(cache_pickle_tiles_path):
        with open(cache_pickle_tiles_path, "rb") as file:
            cache = pickle.load(file)
            signature = compute_root_dirs_signature(root_dirs)
            if cache.get('root_dirs_signature') == signature:
                print(f"Loaded tile paths from valid cache: {cache_pickle_tiles_path}")
                return cache['tile_paths']
            else:
                print(f"Data directories have changed! Invalidating tile cache: {cache_pickle_tiles_path}")
    return None

def get_tile_paths(root_dirs, test_patients_file, split, max_tiles_per_patient, cohort_size_threshold, process_only_zips, cache_pickle_tiles_path, cache_cohort_sizes_path, force_recalculate_tile_paths=False):
    """
    Get image tile paths, filtering based on train/test split if needed.

    Args:
        root_dir (List[str]): Path to root directory containing patient folders.
        test_patients_file (str): JSON file with test set patient IDs.
        split (str): Either 'train' or 'test' to filter images.

    Returns:
        List[str]: Sorted list of tile paths for the selected split.
    """
    random.seed(42)
    test_patients = load_test_patients(test_patients_file) if test_patients_file else None
    cohort_sizes, cohort_sizes_wsi = load_or_calculate_cohort_sizes(root_dirs, process_only_zips=process_only_zips, cache_file=cache_cohort_sizes_path)
    print(f"Cohort size (total n tiles): {dict(cohort_sizes)}")
    print(f"Cohort size (total n WSIs): {dict(cohort_sizes_wsi)}")

    if (split in ['train', 'none']) and max_tiles_per_patient is not None:
        sampling_ratios = calculate_sampling_ratio(cohort_sizes, max_tiles_per_patient, cohort_size_threshold)
        print(f"\nSampling rations: {sampling_ratios}")

    if cache_pickle_tiles_path: 
        if os.path.exists(cache_pickle_tiles_path) and not force_recalculate_tile_paths:
            tiles = load_tile_paths_cache(root_dirs, cache_pickle_tiles_path)
            if tiles:
                return tiles
            else:
                print("Scanning directories to get all tiles...")

    tile_paths = []
    for root_dir in tqdm(root_dirs):
        with os.scandir(root_dir) as patient_entries:
            #progress_bar = tqdm(list(patient_entries), desc="Initializing tiles retrieval...", leave=False)
            for patient_entry in patient_entries:
                patient_path = patient_entry.path
                patient_id = '-'.join(patient_entry.name.split('-')[:3])
                cohort_id = root_dir.split('-')[-1]

                #progress_bar.set_description(f"Scanning Cohort: {cohort_id}")

                if test_patients is not None and split != 'none':
                    if split == 'train' and patient_id in test_patients:
                        continue  # skip test patients in train split
                    elif split == 'test' and patient_id not in test_patients:
                        continue  # skip non-test patients in test split

                if process_only_zips and patient_entry.name.endswith('.zip'):
                    tile_files = get_tiles_from_zip(patient_path)
                    tile_files = [f"{patient_path}:{name}" for name in tile_files]
                elif patient_entry.is_dir():
                    if os.path.exists(os.path.join(patient_path, "tiles")):
                        exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
                        tile_files = [file for ext in exts for file in glob.glob(os.path.join(patient_path, 'tiles', ext))]
                    else:
                        exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
                        tile_files = [file for ext in exts for file in glob.glob(os.path.join(patient_path, ext))]
                else:
                    continue

                if split in {'train', 'none'} and max_tiles_per_patient is not None:
                    max_tiles = sampling_ratios.get(cohort_id)
                    if len(tile_files) > max_tiles:
                        tile_files = random.sample(tile_files, max_tiles)

                tile_paths.extend(tile_files)

    print(f"\nNumber of tiles found: {len(tile_paths)}")

    cache = {
    'root_dirs_signature': compute_root_dirs_signature(root_dirs),
    'tile_paths': sorted(tile_paths)
    }

    if cache_pickle_tiles_path:
        with open(cache_pickle_tiles_path, "wb") as file:
            pickle.dump(cache, file)
        print(f"Tile paths cached to {cache_pickle_tiles_path}")
    return sorted(tile_paths)

def extract_coords(filename):
    """
    Extract (x, y) coordinates from filename.
    Example: 'tile_(1024.9512, 12811.89).jpg' → (1024.9512, 12811.89)
    """
    match = re.search(r'tile_\(([\d.]+), ([\d.]+)\)\.\w+$', filename)
    if match:
        return np.array([float(match.group(1)), float(match.group(2))], dtype=np.float32)
    # Return a dummy array, not None to avoid errors when creating batches
    return np.array([-1, -1], dtype=np.float32)

class TilesDataset(Dataset):
    def __init__(self, root_dirs, test_patients_file=None, split='none', transform=None, max_tiles_per_patient=None, cohort_size_threshold=1_400_000, process_only_zips=False, cache_pickle_tiles_path=None, cache_cohort_sizes_path=None, force_recalculate_tile_paths=False):
        """
        Args:
            root_dir (str): Path to the root directory containing patient folders.
            transform (callable, optional): Optional transform to apply to images.
        """
        self.root_dirs = root_dirs
        self.transform = transform
        self.tile_paths = get_tile_paths(tuple(root_dirs), test_patients_file, split, max_tiles_per_patient, cohort_size_threshold, process_only_zips, cache_pickle_tiles_path, cache_cohort_sizes_path, force_recalculate_tile_paths)

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, index):
        tile_path = self.tile_paths[index]
        image = Image.open(tile_path).convert("RGB")
        return {"img": image, "filename": os.path.basename(tile_path), 'index': index}

class DefaultTilesDataset(TilesDataset):
    def __init__(self, 
                root_dirs: list, 
                test_patients_file_path: str = None,
                split: str = 'none',
                max_tiles_per_patient: int = None,
                cohort_size_threshold: int = 1_400_000,
                as_tensor: bool = True,
                do_normalize: bool = True,
                do_resize: bool = False,
                img_size: int = 224,
                process_only_zips: bool = False,
                cache_pickle_tiles_path: str = None,
                cache_cohort_sizes_path: str = None,
    ):
        super().__init__(
            root_dirs=root_dirs, 
            test_patients_file=test_patients_file_path, 
            split=split, 
            max_tiles_per_patient=max_tiles_per_patient, 
            cohort_size_threshold=cohort_size_threshold,
            process_only_zips=process_only_zips, 
            cache_pickle_tiles_path=cache_pickle_tiles_path,
            cache_cohort_sizes_path=cache_cohort_sizes_path
            )

        transform_list = []
        if do_resize:
            transform_list.append(transforms.Resize(size=img_size, interpolation=transforms.InterpolationMode.BILINEAR))
        if as_tensor:
            transform_list.append(transforms.ToTensor())
        if do_normalize:
                # this transform is needed for diffusion only, FMs expect different preprocessing
                transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        if transform_list:
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = None

    def __getitem__(self, index):
        tile_path = self.tile_paths[index]

        if ".zip:" in tile_path:
            zip_path, internal_path = tile_path.split(":", 1)
            
            with zipfile.ZipFile(zip_path, 'r') as z:
                with z.open(internal_path) as img_file:
                    image = Image.open(BytesIO(img_file.read())).convert("RGB")
            patient_id = '.'.join(os.path.basename(tile_path).split('.zip')[0].split('.')[:2])
        else: 
            image = Image.open(tile_path).convert("RGB")
            patient_id = '.'.join(os.path.basename(os.path.dirname(tile_path)).split('.')[:2])

        tile_coords = extract_coords(tile_path)

        if self.transform:
            image = self.transform(image)

        return {"img": image, "coords": tile_coords, "filename": tile_path, "index": index}

    def get_images_by_patient_and_fname(self, patient_name, fname):
        # for simple cases, when we have patient folders/zips full of tiles
        for tile_path in self.tile_paths:
            if fname in tile_path:
                print(f"Found: {tile_path}")
                image = Image.open(tile_path)
                if self.transform:
                    image = self.transform(image)
                return {'image': image, 'filename': tile_path}

            # fallback: look deeper within same patient dir
            patient_dir = os.path.dirname(tile_path)
            for file in glob.glob(os.path.join(patient_dir, '**', '*'), recursive=True):
                if fname in os.path.basename(file):
                    print(f"Found: {file}")
                    image = Image.open(file)
                    if self.transform:
                        image = self.transform(image)
                    return {'image': image, 'filename': file}


class ImageTileDatasetWithFeatures(DefaultTilesDataset):
    def __init__(self, 
                root_dirs: list, 
                feature_dirs: list,
                test_patients_file: str = None,
                split: str = 'none',
                max_tiles_per_patient: int = None,
                cohort_size_threshold: int = 1_400_000,
                feat_extractor: str = 'conch',
                as_tensor: bool = True,
                do_normalize: bool = True,
                do_resize: bool = True,
                process_only_zips: bool = False,
                cache_pickle_tiles_path: str = None,
    ):
        super().__init__(root_dirs=root_dirs, test_patients_file=test_patients_file, split=split, max_tiles_per_patient=max_tiles_per_patient, cohort_size_threshold=cohort_size_threshold, process_only_zips=process_only_zips, cache_pickle_tiles_path=cache_pickle_tiles_path)
        self.feature_dirs = feature_dirs
        print("-----Dataset parameters-----")
        print(f"Image directories: {root_dirs}")
        print(f"Feature directories: {feature_dirs}")
        print(f"Feature extractor: {feat_extractor}")
        print(f"Only zipped tiles will be processed: {process_only_zips}")
        print(f"Tile paths will be loaded from (if pkl file exists) or saved to: {cache_pickle_tiles_path} to save scanning folders time.")
        print(f"Dataset split: {split}")
        print(f"Maximum tiles per patient: {max_tiles_per_patient}")
        print(f"Cohort size threshold after which limit the number of tiles: {cohort_size_threshold}")

        transform_list = []
        if do_resize:
            if feat_extractor == 'conch1_5':
                transform_list.append(transforms.Resize(size=448, interpolation=transforms.InterpolationMode.BILINEAR))
            elif feat_extractor == 'conch':
                transform_list.append(transforms.Resize(size=448, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True))
            elif feat_extractor == 'v2':
                transform_list.append(transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True))
            elif feat_extractor == 'uni2':
                transform_list.append(transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BILINEAR, max_size=None, antialias=True))
            elif feat_extractor == 'custom':
                transform_list.append(transforms.Resize(size=512, interpolation=transforms.InterpolationMode.BILINEAR, max_size=None, antialias=True))
            else:
                raise NotImplementedError()
        if as_tensor:
            transform_list.append(transforms.ToTensor())
        if do_normalize:
                transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        if transform_list:
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = None

    def __getitem__(self, index):
        tile_path = self.tile_paths[index]

        if ".zip:" in tile_path:
            zip_path, internal_path = tile_path.split(":", 1)
            
            with zipfile.ZipFile(zip_path, 'r') as z:
                with z.open(internal_path) as img_file:
                    image = Image.open(BytesIO(img_file.read())).convert("RGB")
            patient_id = '.'.join(os.path.basename(tile_path).split('.zip')[0].split('.')[:2])
        else: 
            image = Image.open(tile_path).convert("RGB")
            patient_id = '.'.join(os.path.basename(os.path.dirname(tile_path)).split('.')[:2])

        tile_coords = extract_coords(tile_path)
        
        if ".zip:" in tile_path:
            zip_path, internal_path = tile_path.split(":", 1)
            cohort = os.path.basename(os.path.dirname(zip_path))
        else:
            cohort = os.path.basename(os.path.dirname(os.path.dirname(tile_path)))

        found = False
        for feature_dir in self.feature_dirs:
            if cohort not in feature_dir:
                continue

            feat_path = os.path.join(feature_dir, f"{patient_id}.h5")
            with h5py.File(feat_path, 'r') as f:
                features = torch.tensor(f['feats'][:])
                coords = np.array(f["coords"][:])
                #match_idx = np.where((tile_coords == coords).all(axis=1))[0]

                coords_dict = {tuple(coord): idx for idx, coord in enumerate(coords)}
                match_idx = coords_dict.get(tuple(tile_coords))

                if match_idx is None:
                #if len(match_idx) == 0:
                    raise KeyError(f"Feats not found for {patient_id} tile {tile_coords}")
                #elif len(match_idx) > 1:
                #    print(f'Weirdly multiple feats found (N = {len(match_idx)}) for patient {patient_id} tile {tile_coords}')
                else:
                    found = True
                    break

        if self.transform:
            image = self.transform(image)

        return {"img": image, "feat": features[match_idx], "coords": tile_coords, "filename": tile_path}


class DefaultAttrDataset(Dataset):
    def __init__(
        self,
        root_dirs,
        attr_path,
        id_to_cls,
        test_patients_file_path=None,
        split='none',
        max_tiles_per_patient=None,
        cohort_size_threshold=1_400_000,
        as_tensor=True,
        do_normalize=True,
        do_resize=False,
        img_size=224,
        process_only_zips=False,
        cache_pickle_tiles_path=None,
        cache_cohort_sizes_path=None,
    ):
        self.id_to_cls = id_to_cls
        self.cls_to_id = {v: k for k, v in enumerate(self.id_to_cls)}

        self.tiles_dataset = DefaultTilesDataset(
            root_dirs=root_dirs,
            test_patients_file_path=test_patients_file_path,
            split=split,
            max_tiles_per_patient=max_tiles_per_patient,
            cohort_size_threshold=cohort_size_threshold,
            as_tensor=as_tensor,
            do_normalize=do_normalize,
            do_resize=do_resize,
            img_size=img_size,
            process_only_zips=process_only_zips,
            cache_pickle_tiles_path=cache_pickle_tiles_path,
            cache_cohort_sizes_path=cache_cohort_sizes_path,
        )
        if attr_path is None or os.path.exists(attr_path) is False:
            raise FileNotFoundError(f"Attribute file not found: {attr_path}")
        with open(attr_path) as f:
            self.df = pd.read_csv(f)
            print(self.df)
        self.df = self.df.set_index('FILENAME')

    def get_valid_indices(self):
        """
        Filters the dataset based on which filenames are present in the label table (self.df).
        Returns the valid indices of the dataset.
        """
        valid_indices = []
        tile_paths = self.tiles_dataset.tile_paths  # all tile filepaths
        label_filenames = set(self.df.index)       # set of valid filenames from label table

        for idx, path in enumerate(tile_paths):
            fname = os.path.basename(path)
            if fname in label_filenames:
                valid_indices.append(idx)
        print(f"Number of images found with matching labels: {len(valid_indices)}")
        return valid_indices

    def __len__(self):
        return len(self.tiles_dataset)

    def __getitem__(self, index):
        item = self.tiles_dataset[index]
        fname = os.path.basename(item['filename'])

        if fname not in self.df.index:
            raise KeyError(f"File {fname} not in label table.")
        row = self.df.loc[fname]
        labels = torch.tensor([row.get(cls, 0) for cls in self.id_to_cls], dtype=torch.float32)
        item['labels'] = labels
        return item

def _build_transform(
    *,
    feat_extractor: Optional[str],
    do_resize: bool,
    img_size: int,
    do_normalize: bool,
    as_tensor: bool,
):
    """Compose torchvision transforms, honoring feat_extractor-specific sizes."""
    size = img_size
    if do_resize and feat_extractor is not None:
        if feat_extractor == "conch1_5":
            size = 448
            resize = transforms.Resize(size=size, interpolation=transforms.InterpolationMode.BILINEAR)
        elif feat_extractor == "conch":
            size = 448
            resize = transforms.Resize(size=size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        elif feat_extractor == "v2":
            size = 224
            resize = transforms.Resize(size=size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        elif feat_extractor == "uni2":
            size = 224
            resize = transforms.Resize(size=size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
        elif feat_extractor == "custom":
            resize = transforms.Resize(size=size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
        else:
            resize = transforms.Resize(size=img_size, interpolation=transforms.InterpolationMode.BILINEAR)
    else:
        resize = transforms.Resize(size=img_size, interpolation=transforms.InterpolationMode.BILINEAR) if do_resize else None

    t: List[transforms.Transform] = []
    if resize is not None:
        t.append(resize)
    if as_tensor:
        t.append(transforms.ToTensor())
    if do_normalize:
        # diffusion training expects [-1, 1]
        t.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    return transforms.Compose(t) if t else None

def dict_collate(samples: List[Dict]) -> Dict:
    """
    Collate list of dict samples into dict of tensors. Leaves strings as lists.
    """
    if not samples:
        return {}
    out: Dict[str, Union[torch.Tensor, List]] = {}
    keys = samples[0].keys()
    for k in keys:
        vals = [s[k] for s in samples]
        if isinstance(vals[0], torch.Tensor):
            out[k] = default_collate(vals)
        else:
            out[k] = vals
    return out

def split_key(key: str):
    parts = key.strip("/").rsplit("/", 2)
    if len(parts) == 3:  # cohort/patient/stem
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return "", parts[0], parts[1]
    return "", "", parts[0]

def _pick_img_pil(sample):
    for k in IMAGE_KEYS:
        if k in sample:
            v = sample[k]
            if isinstance(v, Image.Image):
                return v
            else:
                return Image.open(io.BytesIO(v)).convert("RGB")
    raise KeyError(f"No image stream in sample keys: {list(sample.keys())}")

def _as_json(v):
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        return json.loads(v.decode("utf-8"))
    if isinstance(v, str):
        return json.loads(v)
    if isinstance(v, dict):
        return v
    raise TypeError(f"Unsupported JSON type: {type(v)}")

def _patient_from(sample, default_from_key: str):
    pt = sample.get("patient.txt")
    if pt is None: 
        return default_from_key
    return pt.decode("utf-8") if isinstance(pt, (bytes, bytearray)) else str(pt)


def _coords_from(sample):
    """Return (x, y) from coords.json only; fail if missing/invalid."""
    cj = sample.get("coords.json")
    if cj is None:
        raise KeyError(f"coords.json missing for key={sample.get('__key__')}")
    d = _as_json(cj)  # handles bytes/str/dict

    try:
        x = float(d["x"])
        y = float(d["y"])
    except Exception as e:
        raise ValueError(f"Invalid coords.json for key={sample.get('__key__')}: {d!r}") from e
    return (x, y)

class WDSTiles(IterableDataset):
    """
    Streaming tiles from WebDataset shards.
    Yields dict with: img (Tensor), coords (Tensor[2]), filename (str), patient (str), cohort (str), key (str)
    Use .to_loader(...) to obtain a WebLoader (DataLoader-like).
    """
    def __init__(
        self,
        shards: Union[str, List[str]],
        *,
        feat_extractor: str = "custom",
        do_resize: bool = True,
        img_size: int = 224,
        as_tensor: bool = True,
        do_normalize: bool = True,
        coords_regex = r"tile_\(([0-9.+-eE]+),\s*([0-9.+-eE]+)\)",
        pre_shuffle: int = 500,
        post_shuffle: int = 256,
        resampled: bool = False,
        strict: bool = False,  # raise on first issue
    ):
        super().__init__()
        self.shards = shards
        self.transform = _build_transform(feat_extractor=feat_extractor, do_resize=do_resize, img_size=img_size, as_tensor=as_tensor, do_normalize=do_normalize)
        self.coords_regex = coords_regex
        self.pre_shuffle = pre_shuffle
        self.post_shuffle = post_shuffle
        self.resampled = resampled  # if True, ignores shardshuffle
        self.strict = strict

    def pipeline(self):
        handler = wds.handlers.reraise_exception if self.strict else wds.handlers.warn_and_continue
        shardshuffle_val = 0 if self.resampled else 10000

        ds = wds.WebDataset(
                self.shards,
                resampled=self.resampled,
                shardshuffle=shardshuffle_val,
                nodesplitter=wds.split_by_node,
                workersplitter=wds.split_by_worker,
                handler=handler,
            )
        if self.pre_shuffle:
            ds = ds.shuffle(self.pre_shuffle)

        ds = ds.map(self._map_one, handler=handler)

        if self.post_shuffle:
            ds = ds.shuffle(self.post_shuffle)
        return ds
    
    def __iter__(self):
        for s in self.pipeline():
            yield s

    def to_loader(self, batch_size: int, num_workers: int, steps_per_epoch=None):
        ds = self.pipeline()
        ds = ds.batched(batch_size, partial=False, collation_fn=dict_collate)
        loader = wds.WebLoader(
            ds,
            batch_size=None,                 # already batched by .batched()
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        return loader.with_epoch(steps_per_epoch) if steps_per_epoch else loader

    def _map_one(self, sample):
        key = sample["__key__"]
        cohort, patient_key, stem = split_key(key)
        img_pil = _pick_img_pil(sample)
        img = self.transform(img_pil) if self.transform else transforms.ToTensor()(img_pil)
        coords = _coords_from(sample)

        return {
            "img": img,
            "coords": torch.tensor(coords, dtype=torch.float32),
            "filename": f"{stem}.png",
            "patient": patient_key,
            "cohort": cohort,
            "key": key,
        }

def f32pair_key(x, y):
    return struct.pack("<ff", np.float32(x), np.float32(y))

class H5OpenFileCache:
    def __init__(self, max_open=4, feat_key="feats", coords_key="coords"):
        self.max_open = max_open
        self.feat_key = feat_key
        self.coords_key = coords_key
        self.cache = OrderedDict()

    def get(self, h5_path: str):
        if h5_path in self.cache:
            self.cache.move_to_end(h5_path)
            return self.cache[h5_path]

        f = h5py.File(h5_path, "r")
        coords = np.asarray(f[self.coords_key][:], dtype=np.float32)  # (N,2) as float32
        index = {f32pair_key(x, y): i for i, (x, y) in enumerate(coords)}
        payload = {"f": f, "feat_ds": f[self.feat_key], "index": index, "coords": coords}
        self.cache[h5_path] = payload

        if len(self.cache) > self.max_open:
            _, old = self.cache.popitem(last=False)
            try: old["f"].close()
            except Exception: pass
        return payload

class WDSTilesWithFeatures(WDSTiles):
    """
    Streaming tiles with H5 features per patient.
    `feature_dirs` can be either:
      - dict: {"BRCA": "/path/to/brca_h5", "LUAD": "/path/to/luad_h5", ...}
      - list[str]: we will pick the first dir whose path contains the cohort name
    """
    def __init__(
        self,
        shards: Union[str, List[str]],
        feature_dirs: Union[Dict[str, str], List[str]],
        *,
        feat_key: str = "feats",
        coords_key: str = "coords",
        h5_cache_items: int = 8,
        **kwargs,
    ):
        super().__init__(shards, **kwargs)
        self.feature_dirs = feature_dirs
        self.feat_key = feat_key
        self.coords_key = coords_key
        self.cache = H5OpenFileCache(max_open=h5_cache_items, feat_key=feat_key, coords_key=coords_key)
        self._h5_path_cache = {} 

    def pipeline(self):
        base = super().pipeline()
        return base.map(self._add_features, handler=wds.handlers.warn_and_continue)

    def _add_features(self, sample):
        cohort  = sample["cohort"]
        patient = sample["patient"]
        x, y = (float(v) for v in sample["coords"].tolist())

        h5_path = self._find_h5_path_for_patient(cohort, patient)
        if h5_path is None:
            raise FileNotFoundError(f"No H5 for cohort={cohort} patient={patient}")

        payload = self.cache.get(h5_path)
        k = f32pair_key(x, y)
        idx = payload["index"].get(k)

        if idx is None:
            # ultra-rare: final safety with tiny tol in float32 space
            c = payload["coords"]  # already float32
            hits = np.where((np.abs(c[:,0]-np.float32(x)) <= 1e-6) &
                            (np.abs(c[:,1]-np.float32(y)) <= 1e-6))[0]
            if hits.size == 0:
                raise KeyError(f"coords {(x, y)} not found in {h5_path}")
            idx = int(hits[0])

        sample["feat"] = torch.from_numpy(np.asarray(payload["feat_ds"][idx]))
        return sample

    def _resolve_feat_dir(self, cohort: str) -> Optional[str]:
        if isinstance(self.feature_dirs, dict):
            return self.feature_dirs.get(cohort)
        for d in self.feature_dirs:
            if cohort in d:
                return d
        return None

    def _find_h5_path_for_patient(self, cohort: str, patient: str) -> Optional[str]:
        key = (cohort, patient)
        if key in self._h5_path_cache:
            return self._h5_path_cache[key]

        feat_dir = self._resolve_feat_dir(cohort)
        if feat_dir is None:
            self._h5_path_cache[key] = None
            return None

        candidates = [patient, _strip_trailing_hash(patient)]
        seen = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]

        for name in candidates:
            path = os.path.join(feat_dir, f"{name}.h5")
            if os.path.exists(path):
                self._h5_path_cache[key] = path
                return path

        self._h5_path_cache[key] = None
        return None

def _strip_trailing_hash(patient):
    """Strip trailing .<hash> from patient ID if necessary.
    E.g. TCGA-DC-6158-01Z-00-DX1.06cf3c33-83e1-46fd-8717-6c4cddb659d9.f8a1002cd63434ba4e125594d958e55b3e16975b35dbe77cf9b4ab4887951cf8
    becomes  TCGA-DC-6158-01Z-00-DX1.06cf3c33-83e1-46fd-8717-6c4cddb659d9"""
    return ".".join(patient.split('.')[:2])
