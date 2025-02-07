import os
from io import BytesIO
from pathlib import Path
from collections import defaultdict
import lmdb
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
import pickle

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
    """
    def __init__(self, images_dir, path_to_gt_table=None, transform=None):
        super().__init__(images_dir, transform=transform)
        self.transform = transform
        self.images_dir = images_dir
        if path_to_gt_table is not None:
            self.df = pd.read_csv(path_to_gt_table)

            self.df["full_path"] = self.df.apply(lambda row: str(os.path.join(self.images_dir, row['PATIENT_FULL'], row['FILENAME'])), axis=1)
            self.df = self.df.set_index('full_path')
            self.PRED_LABEL = self.df.columns.difference(['PATIENT_FULL', 'FILENAME', 'full_path', 'Unnamed: 0'])

    def __getitem__(self, idx):
        path = self.df.index[idx]
        img = super().__getitem__(idx)['img']
        pred_label = self.get_pred_label(idx)

        return (img, pred_label)
    """
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

  
class DatasetBase(Dataset):
    def __init__(self, path, lmdb_class, as_tensor=True, do_augment=True, do_normalize=True, do_resize=True, zfill=5):
        self.data = lmdb_class(path, zfill)
        self.length = len(self.data)

        transform_list = []
        if do_augment:
            transform_list.append(transforms.RandomHorizontalFlip())
        if do_resize:
            #transform_list.append(transforms.Resize(size=448, interpolation=transforms.InterpolationMode.BILINEAR))       # conch v1.5
            transform_list.append(transforms.Resize(size=448, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True))
        if as_tensor:
            transform_list.append(transforms.ToTensor())
        if do_normalize:
            #transform_list.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))    # conch v1.5
            transform_list.append(transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))) 
        
        if transform_list:
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        #data_item, fname = self.data[index]
        data_item = self.data[index]
        img = data_item if isinstance(data_item, Image.Image) else data_item['img']
        feat = data_item.get('feat', None)
        feat = torch.from_numpy(feat)
        if self.transform:
            img = self.transform(img)
        return {'img': img, 'feat': feat, 'index': index}


class BaseLMDB(Dataset):
    def __init__(self, path, zfill: int = 5):
        self.zfill = zfill
        self.env = self._open_lmdb_env(path)
        self.length = self._get_length()
            
    def _open_lmdb_env(self, path):
        env = lmdb.open(path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not env:
            raise IOError('Cannot open lmdb dataset', path)
        return env
    
    def _get_length(self):
        with self.env.begin(write=False) as txn:
            return int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{str(index).zfill(self.zfill)}'.encode('utf-8')
            try:
                filename_key = f"filename_{str(index).zfill(5)}".encode("utf-8")
                filename = txn.get(filename_key).decode("utf-8")
            except:
                filename = None

            img_bytes = txn.get(key)

            if img_bytes is None:
                raise KeyError(f"Image with key {key} not found in LMDB.")

            feat_key = f"feat_{str(index).zfill(self.zfill)}".encode('utf-8')
            feat_bytes = txn.get(feat_key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer).convert('RGB')
        feat = pickle.loads(feat_bytes) if feat_bytes is not None else None
        return {'img': img, 'filename': filename, 'feat': feat}


class TcgaCRCwoMetadata(DatasetBase):
    def __init__(self,
                 path=None,
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 do_resize: bool = True,
                 zfill: int = 5,
                 ):

        super().__init__(path, lmdb_class=BaseLMDB, as_tensor=as_tensor, do_augment=do_augment, do_normalize=do_normalize, do_resize=do_resize, zfill=zfill)


class TcgaBRCA512lmdbwoMetadata(Dataset):
    def __init__(self,
                 path=None,
                 split=None,
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.data = BaseLMDB(path, zfill=5)
        self.length = len(self.data)

        if split is None:
            self.offset = 0
        elif split == 'train':
            self.length = 85204
            self.offset = 0
        elif split == 'test':
            self.length = self.length - 85204
            self.offset = 85204
        else:
            raise NotImplementedError()

        transform = []
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        print(type(self.data[index]))
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class PanCancerLmdb(DatasetBase):
    def __init__(self,
                 path=None,
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 zfill=5):

        super().__init__(path, lmdb_class=BaseLMDB, as_tensor=as_tensor, do_augment=do_augment, do_normalize=do_normalize, zfill=zfill)


class LungLmdb(DatasetBase):
    def __init__(self,
                 path=os.path.expanduser('datasets/lung/subtypes-lmdb-train'),
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 zfill=5):

        super().__init__(path, lmdb_class=BaseLMDB, as_tensor=as_tensor, do_augment=do_augment, do_normalize=do_normalize, zfill=zfill)


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

    def view_all_patient_ids(self):
        """
        Method to view all patient IDs present in the LMDB dataset.
        """
        patient_ids = set()
        for index in tqdm(range(len(self.data))):
            try:
                _, fname = self.data[index]
                pat_id = "-".join(fname.split("/")[-2].split("-")[:3])
                patient_ids.add(pat_id)
            except Exception as e:
                print(f"Error processing index {index}: {e}")

        print(f"Total unique patient IDs: {len(patient_ids)}")
        return patient_ids

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
