#%%
import os

import sys
sys.path.append("/mnt/bulk-mars/laura/diffae/mopadi")

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from configs.templates import *
from configs.templates_cls import *
from linear_clf.train_linear_cls import ClsModel
from dotenv import load_dotenv
from torch.utils.data import Sampler
pd.set_option('display.max_columns', None)

from dotenv import load_dotenv
load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


#%%

def write_results(log_dir, classes, ground_truth_df, valid_filenames):
    """
    Takes paths of npy files with obtained results, and writes final results 
    to a file patient-preds.csv in the form compatible with wanshi-utils.
    """
    y_pred = np.load(os.path.join(log_dir, "y_pred.npy"))

    pred_df = pd.DataFrame(y_pred, columns=classes)
    pred_df["pred"] = pred_df.idxmax(axis=1)

    ground_truth_df = pd.read_csv(ground_truth_df, header=1, delimiter=' ')
    ground_truth_df["CLASS"] = ground_truth_df[classes].idxmax(axis=1)
    ground_truth_df.drop(columns=classes, inplace=True)

    # Filter ground_truth_df to only include rows with filenames in valid_filenames
    ground_truth_df = ground_truth_df[ground_truth_df['FILENAME'].isin(valid_filenames)]
    print(ground_truth_df)
    ground_truth_df.reset_index(inplace=True)
    ground_truth_df.rename(columns={'index': 'lmdb_index'}, inplace=True)

    combined_df = pd.concat([ground_truth_df, pred_df], axis=1)
    for class_name in classes:
        combined_df[f"CLASS_{class_name}"] = combined_df[class_name]

    combined_df.drop(columns=classes, inplace=True)
    combined_df.to_csv(os.path.join(log_dir, "patches-preds-new.csv"))
    print(f"Results have been written to {os.path.join(log_dir, 'patches-preds-new.csv')}")
#%%

def get_valid_indices(dataset):
    """
    Filters the dataset based on the valid filenames from the attr_path file.
    Returns the valid indices of the LMDB dataset.
    """
    valid_indices = []
    with lmdb.open(dataset.lmdb_path, readonly=True) as env:
        with env.begin(write=False) as txn:
            length_value = txn.get("length".encode("utf-8"))
            if length_value is not None:
                length = int(length_value.decode("utf-8"))
            print(f"Length of LMDB dataset: {length}")
            for idx in tqdm(range(length), desc="Checking fnames in the lmdb"):
                filename_key = f"filename_{str(idx).zfill(5)}".encode("utf-8")
                filename = txn.get(filename_key).decode("utf-8")
                fname = filename.split("/")[-2] + "/" + filename.split("/")[-1]
                if fname in dataset.valid_fnames:
                    valid_indices.append(idx)
    print(f"Number of respective classes images found: {len(valid_indices)}")
    return valid_indices
#%%

if __name__ == "__main__":
#%%
    # Pancancer
    gt_table_dir = f"{ws_path}/mopadi/datasets/pancancer/list_classes_all.txt"
    checkpoint_dir = f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/last.ckpt"    
    dataset_dir = f"{ws_path}/mopadi/datasets/pancancer/japan-lmdb-test"
    data_split_info = "/mnt/bulk-mars/laura/diffae/data/japan/test_train_split.txt"

    # Liver
    #cls_checkpoint_dir = f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/liver_cancer_types_linear_cls/last.ckpt"
    #log_dir = f"{ws_path}/mopadi/checkpoints/pancancer/linear_liver_cancer_types_results"
    #classes = ['Cholangiocarcinoma', 'Liver_hepatocellular_carcinoma']
    #test_dataset = LiverCancerTypesClsDataset(path=dataset_dir)

    # Lung
    cls_checkpoint_dir = f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/lung_subtypes_linear_cls/last.ckpt"
    log_dir = f"{ws_path}/mopadi/checkpoints/pancancer/linear_lung_subtypes_results"
    cls_conf = lung_linear_cls()  
    test_dataset = LungClsDataset(path=dataset_dir)
    classes = ['Lung_adenocarcinoma', 'Lung_squamous_cell_carcinoma']

    #---------------------------------------------------------------------------------

    valid_filenames=test_dataset.valid_fnames
    #valid_indices = get_valid_indices(test_dataset)
#%%
    #write_results(log_dir=log_dir, classes=classes, ground_truth_df=gt_table_dir, valid_filenames=test_dataset.valid_fnames)
    y_pred = np.load(os.path.join(log_dir, "y_pred.npy"))
    print(y_pred)
#%%
    pred_df = pd.DataFrame(y_pred, columns=classes)
    pred_df["pred"] = pred_df.idxmax(axis=1)
    print(len(pred_df))
#%%
    with open(data_split_info, 'r') as f:
        info = json.load(f)
    test_patients = set()
    for cancer_type, details in info.items():
        test_patients.update(details.get("Test set patients", []))
#%%
    ground_truth_df = pd.read_csv(gt_table_dir, header=1, delimiter=' ')

    #filter to only include rows with filenames of patients with relevant classes
    ground_truth_df = ground_truth_df[ground_truth_df['FILENAME'].isin(valid_filenames)]
#%%
    #now filter to get only test patients
    ground_truth_df['PATIENT'] = ground_truth_df['FILENAME'].apply(lambda x: "-".join(x.split("-")[:3]))
    test_ground_truth_df = ground_truth_df[ground_truth_df['PATIENT'].isin(test_patients)]
    test_ground_truth_df["CLASS"] = ground_truth_df[classes].idxmax(axis=1)
    test_ground_truth_df.reset_index(inplace=True)

    columns_to_keep = ['FILENAME', 'CLASS']
    df_filtered = test_ground_truth_df[columns_to_keep]

#%%
    combined_df = pd.concat([df_filtered, pred_df], axis=1)

    for class_name in classes:
        combined_df[f"CLASS_{class_name}"] = combined_df[class_name]

    print(combined_df)

    combined_df.drop(columns=classes, inplace=True)
    combined_df.to_csv(os.path.join(log_dir, "patches-preds.csv"))
    print(f"Results have been written to {os.path.join(log_dir, 'patches-preds.csv')}")

#%%
