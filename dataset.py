import os
from io import BytesIO
from pathlib import Path
from collections import defaultdict
import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import pandas as pd
import numpy as np
import os
from dist_utils import *
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv('WORKSPACE_PATH')


class ImagesBaseDataset(Dataset):
    def __init__(
        self,
        folder,
        exts=["jpg", "tiff", "tif"],
        transform=None,
        do_augment: bool = True,
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
    def __init__(self, images_dir, path_to_gt_table=None, sample=0, transform=None):
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
        all_dir = f'{ws_path}/data/TCGA-CRC/TCGA-CRC-MSI/TCGA-CRC-only-tumor-tiles-msi-all'
        all_patients = [f for f in os.listdir(all_dir) if os.path.isdir(os.path.join(all_dir, f))]

        for patient in all_patients:
            if patient_name in patient:
                patient_path = os.path.join(all_dir, patient)
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
    def __init__(self, images_dir, path_to_gt_table, sample=0, transform=None):
        super().__init__(images_dir, transform=transform)
        self.transform = transform
        self.images_dir = images_dir
        self.df = pd.read_csv(path_to_gt_table)
        self.df = self.df.set_index('FILENAME')
        self.PRED_LABEL = self.df.columns

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.df.index[idx].split("-")[0], self.df.index[idx])
        img = Image.open(path).convert('RGB')
        pred_label = self.get_pred_label(idx)

        if self.transform:
            img = self.transform(img)

        return (img, pred_label)
    

class JapanDataset(ImagesBaseDataset):
    def __init__(self, images_dir, path_to_gt_table, sample=0, transform=None):
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


class BrainDataset(ImagesBaseDataset):
    def __init__(self, images_dir, path_to_gt_table, sample=0, transform=None):
        super().__init__(images_dir, transform=transform)
        self.transform = transform
        self.images_dir = images_dir
        self.df = pd.read_csv(path_to_gt_table)
        self.df = self.df.set_index('FILENAME')
        self.PRED_LABEL = self.df.columns

    def __getitem__(self, idx):
        fname = self.df.index[idx]
        if 'GBM' in fname:
            folder = 'GBM'
        else:
            # folder = 'G4A'
            folder = "IDHmut"

        image = Image.open(os.path.join(self.images_dir, folder, fname))
        pred_label = self.get_pred_label(idx)

        if self.transform:
            image = self.transform(image)

        return {'img': image, 'labels': pred_label, 'filename': fname}
    

class LungDataset(ImagesBaseDataset):
    def __init__(self, images_dir, path_to_gt_table, transform=None):
        super().__init__(images_dir, transform=transform)
        self.transform = transform
        self.images_dir = images_dir
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
        matching_patients = self.df[self.df['FILENAME'].str.contains(patient_name) & self.df['FILENAME'].str.contains(fname)]
        assert len(matching_patients) == 1, "Expected exactly one matching row, but found {}".format(len(matching_patients))
        
        image_details = []
        for idx, row in matching_patients.iterrows():
            fname = row['FILENAME']
            image_path = os.path.join(self.images_dir, fname)
            image = Image.open(image_path)
            
            if self.transform:
                image = self.transform(image)

            if os.path.exists(image_path):
                label = self.get_pred_label(idx)
                image_details.append({'image':image, 'filename': fname, 'path': image_path, 'label': label})
        return image_details[0]

  
class DatasetBase(Dataset):
    def __init__(self, path, lmdb_class, as_tensor=True, do_augment=True, do_normalize=True, zfill=5):
        self.data = lmdb_class(path, zfill)
        self.length = len(self.data)

        transforms_list = []
        if do_augment:
            transforms_list.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transforms_list.append(transforms.ToTensor())
        if do_normalize:
            transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
      
        if transforms_list:
            self.transform = transforms.Compose(transforms_list)
        else:
            self.transform = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        data_item, fname = self.data[index]
        # data_item = self.data[index]
        img = data_item if isinstance(data_item, Image.Image) else data_item['img']
        if self.transform:
            img = self.transform(img)
        return {'img': img, 'index': index, 'filename': fname}


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

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        return img, filename


class BaseLMDBwithMetadata(BaseLMDB):
    def __init__(self, path, zfill: int = 5, patient_id = "", coords=""):
        self.patient_id = patient_id
        self.coords = coords
        super().__init__(path, zfill)
        self.patient_coord_to_index = {}
        self.patient_to_coords = defaultdict(list)

        for i in range(self.length):
            data_item = self.__getitem__(i)
            patient_id = data_item['patient_id']
            coords = data_item['coords']

            key = (patient_id, coords)
            self.patient_coord_to_index[key] = i
            self.patient_to_coords[patient_id].append(coords)

    def __getitem__(self, index):
        key_img = f'{str(index).zfill(self.zfill)}'.encode('utf-8')
        key_meta = f'meta_{str(index).zfill(self.zfill)}'.encode('utf-8')
        key_coord = f'coord_{str(index).zfill(self.zfill)}'.encode('utf-8')
        
        with self.env.begin(write=False) as txn:
            img_bytes = txn.get(key_img)
            patient_id = txn.get(key_meta).decode('utf-8')
            coords = txn.get(key_coord).decode('utf-8')

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        return {'img': img, 'patient_id': patient_id, "coords": coords}

    def get_coords_for_patient(self, patient_id):
        return self.patient_to_coords.get(patient_id, [])

    def get_indices_by_patient_and_coords(self, patient_id, coords):
        """Retrieve indices for a given patient_id and coords"""
        key = (patient_id, coords)
        return self.patient_coord_to_index.get(key, None)

    def get_patient_id_for_index(self, index):
        """Retrieve the patient ID for a given global index"""
        meta_key = f"meta_{str(index).zfill(5)}".encode("utf-8")
        with self.env.begin(write=False) as txn:
            patient_id = txn.get(meta_key)
            if patient_id:
                return patient_id.decode("utf-8")
        return None

    def get_all_patient_ids(self):
        patient_ids = set()
        for i in range(self.length):
            patient_id = self.get_patient_id_for_index(i)
            if patient_id:
                patient_ids.add(patient_id)
        return list(patient_ids)
    

class TextureLMDB(DatasetBase):
    def __init__(self,
                 path=os.path.expanduser('datasets/texture/Texture100k.lmdb'),
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 zfill: int = 5,
                ):
        
        super().__init__(path, lmdb_class=BaseLMDB, as_tensor=as_tensor, do_augment=do_augment, do_normalize=do_normalize, zfill=zfill)


class TcgaCRCwoMetadata(DatasetBase):
    def __init__(self,
                 path=os.path.expanduser('datasets/tcga/tcga_crc-224x224-msi.lmdb'),
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 zfill: int = 5,
                 ):

        super().__init__(path, lmdb_class=BaseLMDB, as_tensor=as_tensor, do_augment=do_augment, do_normalize=do_normalize, zfill=zfill)


class TcgaCRCwMetadata(DatasetBase):
    def __init__(self,
                 path=os.path.expanduser('datasets/tcga/tcga_crc-224x224-msi-val.lmdb'),
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 zfill: int = 5):

        super().__init__(path, lmdb_class=BaseLMDBwithMetadata, as_tensor=as_tensor, do_augment=do_augment, do_normalize=do_normalize, zfill=zfill)

    def __getitem__(self, index):
        assert index < self.length
        data_item = self.data[index]
        img = data_item['img']
        if self.transform:
            img = self.transform(img)
        return {'img': img, 'index': index, 'patient_id': data_item['patient_id'], "coords": data_item["coords"]}
    
    def get_by_patient_coords(self, patient_id, coords):
        """Retrieve data item by patient ID and coordinates."""
        if (patient_id, coords) in self.data.patient_coord_to_index:
            index = self.data.patient_coord_to_index[(patient_id, coords)]
            return self.__getitem__(index)
        else:
            raise ValueError(f"No data found for patient_id={patient_id} and coords={coords}")


class BrainLmdb(DatasetBase):
    def __init__(self,
                 path=os.path.expanduser('datasets/brain/brain-lmdb'),
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 zfill=5):

        super().__init__(path, lmdb_class=BaseLMDB, as_tensor=as_tensor, do_augment=do_augment, do_normalize=do_normalize, zfill=zfill)


class PanCancerLmdb(DatasetBase):
    def __init__(self,
                 path=os.path.expanduser('datasets/japan/japan-lmdb'),
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

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.name
        img_idx, _ = img_name.split('.')
        # img, fname = self.data[int(img_idx)]
        img = self.data[int(img_idx)]
        labels = [0] * len(self.cls_to_id)
        for k, v in row.items():
            labels[self.cls_to_id[k]] = int(v)

        if self.transform:
            img = self.transform(img)
        return {'img': img, 'index': index, 'labels': torch.tensor(labels)}#, 'filename': fname}


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
        'MSIH'
    ]
    cls_to_id = {v: k for k, v in enumerate(id_to_cls)}

    def __init__(self,
                 path=os.path.expanduser(f'{ws_path}/mopadi/datasets/tcga/tcga_crc-msi-dec-20-train-lmdb'),
                 attr_path=os.path.expanduser(f'{ws_path}/mopadi/datasets/tcga/tcga_crc_anno_only_msi/list_attr_tcga_crc_224x224-msi.txt'),
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


class TCGACRCBRAFAttrDataset(AttrDatasetBase):
    id_to_cls = [
        'WT',
        'MUT',
    ]
    cls_to_id = {v: k for k, v in enumerate(id_to_cls)}

    def __init__(self,
                 path=os.path.expanduser('datasets/tcga/tcga_crc-braf-train-lmdb'),
                 attr_path=os.path.expanduser('datasets/tcga/tcga_crc_anno_only_braf/list_attr_tcga_crc_224x224-braf.txt'),
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
                 path=os.path.expanduser('datasets/japan/japan-lmdb'),
                 attr_path=os.path.expanduser('datasets/japan/japan_anno/list_attr.txt'),
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
                 path=os.path.expanduser('datasets/lung/subtypes-lmdb-train'),
                 attr_path=os.path.expanduser('datasets/lung/lung_anno_train/list_attr.txt'),
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        
        super().__init__(path, attr_path, self.cls_to_id, do_transform, do_augment, do_normalize)