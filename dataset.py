import os
from io import BytesIO
from pathlib import Path
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from dist_utils import *
from dotenv import load_dotenv
import re
import pickle
import json
import zipfile
import random
import hashlib

load_dotenv()
ws_path = os.getenv('WORKSPACE_PATH')


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
        

class ImgDataset(Dataset):
    """For making predictions for manipulated images."""
    def __init__(self, image_path, sample=0, transform=None):
        self.transform = transform
        self.image_path = image_path

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_path))
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return (image, self.image_path)


class ImagesBaseDataset(Dataset):
    def __init__(
        self,
        folder,
        exts=["jpg", "tiff", "tif", "png"],
        transform=None,
        do_augment: bool = False,
        do_transform: bool = True,
        do_normalize: bool = True,
    ):
        super().__init__()
        self.folder = Path(folder)

        self.paths = [
            p.relative_to(self.folder) for ext in exts
            for p in self.folder.glob(f'*.{ext}')
        ]

        if transform is None:
            transform = []
            if do_augment:
                transform.append(transforms.RandomHorizontalFlip())
            if do_transform:
                transform.append(transforms.ToTensor())
            if do_normalize:
                transform.append(
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            self.transform = transforms.Compose(transform)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.folder / self.paths[index]
        img = Image.open(path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return {'img': img, 'index': index, 'filename': str(self.paths[index])}
    
    def get_pred_label(self, idx):
        pred_label = np.zeros(len(self.PRED_LABEL), dtype=int)
        for i, label in enumerate(self.PRED_LABEL):
            if self.df[label.strip()].iloc[idx].astype('int') > 0:
                pred_label[i] = self.df[label.strip()].iloc[idx].astype('int')
        return pred_label

    
class TCGADataset(ImagesBaseDataset):
    def __init__(self, images_dir, test_man_amp=None, transform=None):
        self.transform = transform
        self.images_dir = images_dir
        if test_man_amp is not None:
            self.test_man_amp = test_man_amp
            self.patients = [os.path.join(images_dir, name) for name in sorted(os.listdir(images_dir)) if os.path.isdir(os.path.join(images_dir, name))]
            self.patient_subfolders = {
                patient: [
                    os.path.join(patient, subfolder) for subfolder in sorted(os.listdir(patient))
                    if os.path.isdir(os.path.join(patient, subfolder))
                ]
                for patient in self.patients
            }

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_folder = self.patients[idx]
        subfolders = self.patient_subfolders[patient_folder]
        data = {}
        for subfolder in subfolders:
            data[subfolder] = []
            fnames = sorted([os.path.join(subfolder, f) for f in os.listdir(subfolder) if os.path.join(subfolder, f).endswith('png')])
            for fname in fnames:
                if self.test_man_amp in fname:
                    image = Image.open(fname).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    data[subfolder].append(image)
            if len(data[subfolder])==0:
                print(f"Did not find image with {self.test_man_amp} for patient {subfolder}")

        return data

    def get_images_by_patient_and_fname(self, patient_name, coords):
        all_patients = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f))]
        print(f"Looking for {patient_name}")
        for patient in all_patients:
            if patient_name in patient:
                patient_path = os.path.join(self.images_dir, patient)
        tiles = os.listdir(patient_path)
        img_path = None
        for tile in tiles:
            if coords in tile:
                img_path = os.path.join(patient_path, tile)
                fname = tile
                
        if img_path is None:
            return None

        image = Image.open(img_path)
            
        if self.transform:
            image = self.transform(image)

        image_details = {'image':image, 'filename': fname, 'path': img_path}#, 'label': label}
        return image_details
    

class TextureDataset(ImagesBaseDataset):
    def __init__(self, images_dir, transform=None, do_augment=False, do_transform=True, do_normalize=True, ext="tif"):
        super().__init__(images_dir, transform=transform, do_augment=do_augment, do_transform=do_transform, do_normalize=do_normalize)
        self.images_dir = images_dir

        self.id_to_cls = TextureAttrDataset.id_to_cls
        self.cls_to_id = TextureAttrDataset.cls_to_id

        self.image_paths = glob.glob(os.path.join(images_dir, '**', f'*.{ext}'), recursive=True)
        self.PRED_LABEL = self.id_to_cls

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path).convert('RGB')
        
        class_name = os.path.basename(image_path).split('-')[0]
        
        class_id = self.cls_to_id[class_name]
        pred_label = torch.zeros(len(self.id_to_cls), dtype=torch.float32)
        pred_label[class_id] = 1

        if self.transform:
            img = self.transform(img)

        return {'img': img, 'labels': pred_label, 'gt': class_name, 'filename': os.path.basename(image_path)}

    def __len__(self):
        return len(self.image_paths)


class JapanDataset(ImagesBaseDataset):
    def __init__(self, images_dir, path_to_gt_table, transform=None):
        super().__init__(images_dir, transform=transform)
        self.transform = transform
        self.images_dir = images_dir
        self.df = pd.read_csv(path_to_gt_table)
        self.df = self.df.set_index('FILENAME')
        self.PRED_LABEL = self.df.columns

    def __getitem__(self, idx):
        path = os.path.join(self.df.index[idx])
        image = Image.open(path).convert('RGB')
        pred_label = self.get_pred_label(idx)

        if self.transform:
            image = self.transform(image)
        return (image, pred_label)
    

class LungDataset(ImagesBaseDataset):
    def __init__(self, images_dir, path_to_gt_table=None, transform=None):
        super().__init__(images_dir, transform=transform)
        self.transform = transform
        self.images_dir = images_dir
        if path_to_gt_table is not None:
            self.df = pd.read_csv(path_to_gt_table, delim_whitespace=True)
        self.PRED_LABEL = ['Lung_adenocarcinoma', 'Lung_squamous_cell_carcinoma']

    def __getitem__(self, idx):
        fname = self.df['FILENAME'].iloc[idx]
        image = Image.open(os.path.join(self.images_dir, fname))
        pred_label = self.get_pred_label(idx)

        if self.transform:
            image = self.transform(image)

        return {'img': image, 'labels': pred_label, 'filename': fname}

    def get_images_by_patient_and_fname(self, patient_name, fname):
        all_patients = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f))]

        patient_path = None
        for patient in all_patients:
            if patient_name in patient:
                patient_path = os.path.join(self.images_dir, patient)
                break

        if patient_path is None or not os.path.exists(patient_path):
            return None

        tiles = os.listdir(patient_path)
        img_path = None
        filename = None
        for tile in tiles:
            if fname in tile:
                img_path = os.path.join(patient_path, tile)
                filename = tile
                break

        if img_path is None:
            return None

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        image_details = {'image': image, 'filename': filename, 'path': img_path}
        return image_details

# ---------------------------    ^ old scripts

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
    """Extract .jpg tile names from a ZIP file."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        return [name for name in zf.namelist() if name.endswith('.jpg')]


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

            if process_only_zips and patient_folder.endswith('.zip'):
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
    match = re.search(r'tile_\(([\d.]+), ([\d.]+)\)\.jpg', filename)
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
        print(f"Label fnames [:10]: {list(label_filenames)[:10]}")
        print(f"tile_paths [:10]: {list(tile_paths)[:10]}")
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


class AttrDatasetBase(Dataset):
    def __init__(self, path, attr_path, cls_to_id, as_tensor=True, do_augment=True, do_normalize=True, zfill=5):
        print(f"Data will be loaded from: {path}")
        self.data = BaseLMDB(path, zfill)
        self.cls_to_id = cls_to_id

        transform = []
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        with open(attr_path) as f:
            f.readline()  # discard the top line
            self.df = pd.read_csv(f, delim_whitespace=True)

    def pos_count(self, cls_name):
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name):
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        #img_name = row.name

        # img_idx, _ = img_name.split('.')
        img_idx = f'{str(index).zfill(5)}'.encode('utf-8')

        img, fname = self.data[int(img_idx)]
        #img = self.data[int(img_idx)]

        # Extract patient ID from the file path
        pat_id = os.path.join(fname.split("/")[-2], os.path.split(fname)[1])

        if pat_id not in self.df['FILENAME'].values:
            print(f"Patient ID {pat_id} not found in clinical table.")
            raise IndexError(f"Patient ID {pat_id} not found in clinical table.")

        # Retrieve the row corresponding to the patient ID
        row = self.df[self.df['FILENAME'] == pat_id].iloc[0]
        #print(row)

        labels = [0] * len(self.cls_to_id)
        for k, v in row.items():
            if k == 'FILENAME':
                continue
            if k in self.cls_to_id:
                labels[self.cls_to_id[k]] = int(v)
                valid_labels_found = True

        if not valid_labels_found:
            return None

        if self.transform:
            img = self.transform(img)
        return {'img': img, 'index': index, 'labels': torch.tensor(labels), 'filename': fname}


class TextureAttrDataset(AttrDatasetBase):
    id_to_cls = [
        'ADI',
        'BACK',
        'DEB',
        'LYM',
        'MUC',
        'MUS',
        'NORM',
        'STR',
        'TUM'
    ]
    cls_to_id = {v: k for k, v in enumerate(id_to_cls)}

    def __init__(self,
                 path=os.path.expanduser('datasets/texture/texture100k-diff-indexes.lmdb'),
                 attr_path=os.path.expanduser('datasets/texture100k-diff-indexes_anno/list_attr_texture.txt'),
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        
        super().__init__(path, attr_path, self.cls_to_id, do_transform, do_augment, do_normalize)


class TcgaCrcMsiAttrDataset(AttrDatasetBase):
    id_to_cls = [
        'nonMSIH',
        'MSIH',
        'unknown'
    ]
    cls_to_id = {v: k for k, v in enumerate(id_to_cls)}

    def __init__(self,
                 path=os.path.expanduser(f'{ws_path}/mopadi/datasets/crc/tcga_crc_512_lmdb'),
                 attr_path=os.path.expanduser(f'{ws_path}/mopadi/datasets/crc/list_attr_msi_tcga_crc_512.txt'),
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        
        super().__init__(path, attr_path, self.cls_to_id, do_transform, do_augment, do_normalize)
    

class BrainAttrDataset(AttrDatasetBase):
    id_to_cls = [
        'GBM',
        'IDHmut',
    ]
    cls_to_id = {v: k for k, v in enumerate(id_to_cls)}

    def __init__(self,
                 path=os.path.expanduser('datasets/brain/brain-w-fnames'),
                 attr_path=os.path.expanduser(
                     # f'{ws_path}/mopadi/datasets/brain_anno/list_attr.txt'),
                     f'{ws_path}/mopadi/datasets/brain/brain_anno-GBM-IDHmut-new/list_attr.txt'),
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        
        super().__init__(path, attr_path, self.cls_to_id, do_transform, do_augment, do_normalize)


class PanCancerClsDataset(AttrDatasetBase):
    id_to_cls = [
        'Mesothelioma',
        'Uterine_Carcinosarcoma',
        'Kidney_renal_papillary_cell_carcinoma',
        'Cholangiocarcinoma',
        'Pheochromocytoma_and_Paraganglioma',
        'Adrenocortical_carcinoma',
        'Brain_Lower_Grade_Glioma',
        'Ovarian_serous_cystadenocarcinoma', 
        'Bladder_Urothelial_Carcinoma',
        'Breast_invasive_carcinoma', 
        'Liver_hepatocellular_carcinoma',
        'Kidney_renal_clear_cell_carcinoma',
        'Uveal_Melanoma',
        'Rectum_adenocarcinoma',
        'Testicular_Germ_Cell_Tumors', 
        'Thymoma',
        'Pancreatic_adenocarcinoma',
        'Lung_adenocarcinoma',
        'Prostate_adenocarcinoma',
        'Esophageal_carcinoma',
        'Cervical_squamous_cell_carcinoma_and_endocervical_adenocarcinoma',
        'Lymphoid_Neoplasm_Diffuse_Large_B-cell_Lymphoma', 
        'Colon_adenocarcinoma',
        'Stomach_adenocarcinoma', 
        'Skin_Cutaneous_Melanoma', 
        'Head_and_Neck_squamous_cell_carcinoma', 
        'Kidney_Chromophobe', 
        'Glioblastoma_multiforme', 
        'Thyroid_carcinoma', 
        'Lung_squamous_cell_carcinoma',
        'Sarcoma', 
        'Uterine_Corpus_Endometrial_Carcinoma'
    ]
    cls_to_id = {v: k for k, v in enumerate(id_to_cls)}

    def __init__(self,
                 path=f'{ws_path}/mopadi/datasets/pancancer/japan-lmdb',
                 attr_path=f'{ws_path}/mopadi/datasets/pancancer/list_attr_train.txt',
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        
        super().__init__(path, attr_path, self.cls_to_id, do_transform, do_augment, do_normalize)


class LungClsDataset(AttrDatasetBase):
    id_to_cls = [
        'Lung_adenocarcinoma', 
        'Lung_squamous_cell_carcinoma'
    ]
    cls_to_id = {v: k for k, v in enumerate(id_to_cls)}

    def __init__(self,
                 path = f'{ws_path}/mopadi/datasets/pancancer/japan-lmdb-train-new',
                 attr_path = f'{ws_path}/mopadi/datasets/pancancer/list_classes_all.txt',
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        
        super().__init__(path, attr_path, self.cls_to_id, do_transform, do_augment, do_normalize)

        self.attr_path = attr_path
        self.lmdb_path = path
        self.valid_fnames = self.load_valid_filenames(attr_path)

    def load_valid_filenames(self, attr_path):
        """
        Loads the attr_path file which contains filenames and their corresponding class indicators.
        Returns a set of valid filenames based on the target cancer types. All in one attr file (test & train!)
        """
        valid_filenames = set()

        with open(attr_path, 'r') as f:
            total_img_nr = f.readline()

            class_headers = f.readline().strip().split()[1:]
            target_indices = [i for i, class_name in enumerate(class_headers) if class_name in self.cls_to_id]

            for line in f:
                parts = line.strip().split()
                filename = parts[0]
                class_flags = parts[1:]

                for idx in target_indices:
                    if class_flags[idx] == '1':
                        valid_filenames.add(filename)
                        break
        return valid_filenames


class LiverCancerTypesClsDataset(AttrDatasetBase):
    id_to_cls = [
        'Cholangiocarcinoma', 
        'Liver_hepatocellular_carcinoma'
    ]
    cls_to_id = {v: k for k, v in enumerate(id_to_cls)}

    def __init__(self,
                 path = f'{ws_path}/mopadi/datasets/pancancer/japan-lmdb-train-new',
                 attr_path = f'{ws_path}/mopadi/datasets/pancancer/list_classes_all.txt',
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        
        super().__init__(path, attr_path, self.cls_to_id, do_transform, do_augment, do_normalize)

        self.attr_path = attr_path
        self.lmdb_path = path
        self.valid_fnames = self.load_valid_filenames(attr_path)

    def load_valid_filenames(self, attr_path):
        """
        Loads the attr_path file which contains filenames and their corresponding class indicators.
        Returns a set of valid filenames based on the target cancer types. All in one attr file (test & train!)
        """
        valid_filenames = set()

        with open(attr_path, 'r') as f:
            total_img_nr = f.readline()

            class_headers = f.readline().strip().split()[1:]
            target_indices = [i for i, class_name in enumerate(class_headers) if class_name in self.cls_to_id]

            for line in f:
                parts = line.strip().split()
                filename = parts[0]
                class_flags = parts[1:]

                for idx in target_indices:
                    if class_flags[idx] == '1':
                        valid_filenames.add(filename)
                        break
        return valid_filenames
