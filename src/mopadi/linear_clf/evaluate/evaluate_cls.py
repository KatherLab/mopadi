import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from mopadi.configs.templates import *
from mopadi.configs.templates_cls import *
from mopadi.linear_clf.train_linear_cls import ClsModel

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


def write_results(log_dir, classes, ground_truth_df, save_path):
    """
    Takes paths of npy files with obtained results, and writes final results 
    to a file patient-preds.csv in the form compatible with wanshi-utils.
    """
    y_pred = np.load(os.path.join(log_dir, "y_pred.npy"))

    pred_df = pd.DataFrame(y_pred, columns=classes)
    pred_df["pred"] = pred_df.idxmax(axis=1)

    ground_truth_df = pd.read_csv(ground_truth_df)
    ground_truth_df["CLASS"] = ground_truth_df[classes].idxmax(axis=1)
    ground_truth_df.drop(columns=classes, inplace=True)
    try:
        ground_truth_df.drop(columns=["Unnamed: 0"], inplace=True)
    except:
        pass
    
    combined_df = pd.concat([ground_truth_df, pred_df], axis=1)
    for class_name in classes:
        combined_df[f"CLASS_{class_name}"] = combined_df[class_name]

    combined_df.drop(columns=classes, inplace=True)

    combined_df.to_csv(os.path.join(save_path, "patches-preds-new-june-2025.csv"))


def make_gt_table(root_dir, save_dir):
    """
    Makes a ground truth table for datasets, where
    classes are inferred from subdirectories names.

    For example, the resulting CSV file looks like this:

        FILENAME	           STR LYM TUM BACK	ADI	DEB	MUC	MUS	NORM
    0	STR-TCGA-CWFPNFSC.tif	1	0	0	0	0	0	0	0	0
    """

    filenames = []
    subfolder_names = []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".tif"):
                continue
            #filenames.append(subdir + "/" + file)
            filenames.append(file)
            subfolder_names.append(os.path.basename(subdir))
    # print(filenames)
    
    df = pd.DataFrame({"FILENAME": filenames, "Subfolder": subfolder_names})

    for category in subfolder_names:
        df[category] = df["Subfolder"].apply(lambda x: 1 if x == category else 0)

    df.drop("Subfolder", axis=1, inplace=True)
    df.to_csv(save_dir, index=False)


def make_gt_table_tcga(wsi_dir, save_dir, clini_table_path, target_label):
    """
    Makes a ground truth table for TCGA datasets with clini tables. 

    For example, the resulting CSV file looks like this:

        PATIENT        FILENAME	            WT  MUT
    0	TCGA-4N-A93T   Tile_(8960,6720).jpg	 1	 0
    """
    clini_table = pd.read_excel(clini_table_path)
    patient_list = []
    filename_list = []

    for subdir, _, files in os.walk(wsi_dir):
        if not str(subdir).startswith("."):
            patient_name = os.path.basename(subdir)
        else:
            continue
        for file in files:
            if not file.startswith("."):
                patient_list.append(patient_name)
                filename_list.append(file)

    # since patients" folders are named like this TCGA-4N-A93T-01Z-00-DX1, 
    # but in clini table TCGA-4N-A93T, need to adjust the list acccordingly
    edited_patient_list = [("-").join(id.split("-")[:3]) for id in patient_list]

    df = pd.DataFrame({"PATIENT_FULL": patient_list,"PATIENT": edited_patient_list, "FILENAME": filename_list})
    unique_labels = clini_table[target_label].dropna().unique()
    merged_df = pd.merge(df, clini_table, on="PATIENT", how="left")

    for label in unique_labels:
        merged_df[f"{label}"] = merged_df[target_label].apply(lambda x: 1 if x == label else 0)

    keep_columns = ["PATIENT_FULL", "FILENAME"]
    keep_columns.extend(unique_labels)

    merged_df[keep_columns].to_csv(save_dir, index=False)
    

class Tester():
    def __init__(self, log_folder, model_config, checkpoint_path, cls_checkpoint_dir, cls_conf, device="cuda:0"):
        
        self.log_folder = log_folder

        self.model = LitModel(model_config)
        state = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state["state_dict"], strict=False)
        self.model.ema_model.to(device)

        self.cls_model = ClsModel(cls_conf)
        state = torch.load(cls_checkpoint_dir, map_location="cpu", weights_only=False)
        print("latent step:", state["global_step"])
        self.cls_model.load_state_dict(state["state_dict"], strict=False)
        self.cls_model.to(device)

    def test(self, loader):
        label_list, pred_list = [], []
        with torch.no_grad():
            self.model.ema_model.eval()
            self.cls_model.ema_classifier.eval()
            batch = next(iter(loader))
            print(batch)
            for data in tqdm(loader, desc="Predicting"):
                img = data["img"].cuda().float()
                label = data["labels"].float()
                latent = self.model.ema_model.encoder(img)
                latent = self.cls_model.normalize(latent)
                output = self.cls_model.ema_classifier.forward(latent)                
                pred = torch.sigmoid(output)
                
                label_list.append(label.cpu().numpy())
                pred_list.append(pred.cpu().numpy())
        
        pred = np.squeeze(np.array(pred_list))
        label = np.squeeze(np.array(label_list))
        np.save(os.path.join(self.log_folder, "y_pred.npy"), pred)
        np.save(os.path.join(self.log_folder, "y_true.npy"), label)


if __name__ == "__main__":

    # Texture 100k ----------------------------------------------------------------------------------------------------------

    gt_table_dir = f"{ws_path}/mopadi/datasets/texture/texture-val-ground-truth-new.csv"
    checkpoint_dir= f"{ws_path}/mopadi/checkpoints/texture100k/last.ckpt"
    cls_checkpoint_dir = f"{ws_path}/mopadi/checkpoints/texture100k/texture100k_clf/last.ckpt"
    log_dir = f"{ws_path}/mopadi/checkpoints/texture100k/cls_evaluation/new"
    images_dir = f"{ws_path}/data/texture100k/CRC-VAL-HE-7K"
    classes = [
        'ADI',
        'BACK',
        'DEB',
        'LYM',
        'MUC',
        'MUS',
        'NORM',
        'STR',
        'TUM',
    ]

    make_gt_table(root_dir=images_dir, save_dir=gt_table_dir)

    # configurations
    conf = texture100k_autoenc()
    cls_conf = texture100k_linear_cls()
    cls_conf.id_to_cls = classes
    test_dataset = DefaultAttrDataset(root_dirs=[images_dir], attr_path=gt_table_dir, id_to_cls = classes)

    tester = Tester(model_config=conf, checkpoint_path=checkpoint_dir, cls_checkpoint_dir=cls_checkpoint_dir,
                    cls_conf=cls_conf, log_folder=log_dir)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    tester.test(test_loader)
    write_results(log_dir=log_dir, classes=classes, ground_truth_df=gt_table_dir, save_path=log_dir)

    # TCGA CRC BRAF -------------------------------------------------------------------------------------------------------
    """
    clini_table_dir = f"{ws_path}/data/TCGA-CRC/clini-tables/TCGA-CRC-DX_CLINI.xlsx"
    gt_table_dir = f"{ws_path}/mopadi/datasets/tcga/tcga-crc-braf-val-ground-truth.csv"
    checkpoint_dir= f"{ws_path}/mopadi/checkpoints/exp09_tcga_224-diff-cls/last.ckpt"
    cls_checkpoint_dir = f"{ws_path}/mopadi/checkpoints/exp09_tcga_224-diff-cls/tcga_crc_224_autoenc_cls_braf/last.ckpt"
    log_dir = f"{ws_path}/mopadi/checkpoints/exp09_tcga_224-diff-cls/tcga_crc_224_autoenc_cls_braf"
    images_dir = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-only-tumor-BRAF/TCGA-CRC-only-tumor-BRAF-val"

    # configurations
    conf = tcga_crc_autoenc()
    cls_conf = tcga_crc_autoenc_cls_braf()

    make_gt_table_tcga(root_dir=images_dir, save_dir=gt_table_dir, clini_table_path=clini_table_dir, target_label="BRAF")

    tester = Tester(log_folder=log_dir, model_config=conf, checkpoint_path=checkpoint_dir, cls_checkpoint_dir=cls_checkpoint_dir,
                    cls_conf=cls_conf)
    
    test_dataset = TCGADataset(images_dir=images_dir,
                               path_to_gt_table=gt_table_dir,
                               transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    tester.test(test_loader)
    """

    # TCGA CRC MSI  -------------------------------------------------------------------------------------------------------
    """
    clini_table_dir = f"{ws_path}/data/TCGA-CRC/clini-tables/TCGA-CRC-DX_CLINI.xlsx"
    gt_table_dir = f"{ws_path}/mopadi/datasets/tcga/tcga-crc-msi-val-ground-truth.csv"
    checkpoint_dir= f"{ws_path}/mopadi/checkpoints/tcga_crc_224x224/last.ckpt"
    # cls_checkpoint_dir = f"{ws_path}/mopadi/checkpoints/tcga_crc_224x224/tcga_crc_224_autoenc_cls_msi/best_model-epoch=epoch=05-step=step=16120-loss=loss=0.2929.ckpt"
    cls_checkpoint_dir = f"{ws_path}/mopadi/checkpoints/tcga_crc_224x224/tcga_crc_224_autoenc_cls_msi-nonlinear/best_model-epoch=epoch=05-step=step=16120-loss=loss=0.1005.ckpt"
    log_dir = f"{ws_path}/mopadi/checkpoints/tcga_crc_224x224/tcga_crc_224_autoenc_cls_msi-nonlinear/test_best_model"
    images_dir = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-MSI/TCGA-CRC-only-tumor-tiles-msi-val"

    # configurations
    conf = tcga_crc_autoenc()
    cls_conf = tcga_crc_autoenc_cls_msi()    

    tester = Tester(log_folder=log_dir, model_config=conf, checkpoint_path=checkpoint_dir, cls_checkpoint_dir=cls_checkpoint_dir,
                    cls_conf=cls_conf)
    
    test_dataset = TCGADataset(images_dir=images_dir,
                               path_to_gt_table=gt_table_dir,
                               transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    tester.test(test_loader)

    write_results(log_dir=log_dir, classes=["nonMSIH", "MSIH"], ground_truth_df=gt_table_dir, save_path=log_dir)
    """

    # Japan PanCancer TCGA------------------------------------------------------------------------------------------------
    """
    gt_table_dir = f"{ws_path}/mopadi/datasets/japan/japan_anno_val/test_gt_table.txt"
    checkpoint_dir = f"{ws_path}/mopadi/checkpoints/pancancer/last.ckpt"
    cls_checkpoint_dir = f"{ws_path}/mopadi/checkpoints/pancancer/pancancer_cls/last.ckpt"
    log_dir = f"{ws_path}/mopadi/checkpoints/pancancer/evaluation"
    images_dir = f"{ws_path}/data/japan/val"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    make_gt_table(images_dir, gt_table_dir)

    # configurations
    conf = pancancer_autoenc()
    cls_conf = pancancer_autoenc_cls()    

    tester = Tester(log_folder=log_dir, model_config=conf, checkpoint_path=checkpoint_dir, cls_checkpoint_dir=cls_checkpoint_dir,
                    cls_conf=cls_conf)
    
    test_dataset = JapanDataset(images_dir=images_dir,
                               path_to_gt_table=gt_table_dir,
                               transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    tester.test(test_loader)

    classes = [
        "Rectum_adenocarcinoma",
        "Testicular_Germ_Cell_Tumors",
        "Lung_adenocarcinoma",
        "Kidney_renal_clear_cell_carcinoma", 
        "Pancreatic_adenocarcinoma", 
        "Glioblastoma_multiforme",
        "Head_and_Neck_squamous_cell_carcinoma",
        "Thyroid_carcinoma", 
        "Bladder_Urothelial_Carcinoma", 
        "Thymoma", 
        "Skin_Cutaneous_Melanoma", 
        "Cervical_squamous_cell_carcinoma_and_endocervical_adenocarcinoma",
        "Kidney_Chromophobe", 
        "Pheochromocytoma_and_Paraganglioma", 
        "Liver_hepatocellular_carcinoma", 
        "Prostate_adenocarcinoma", 
        "Uterine_Carcinosarcoma", 
        "Lymphoid_Neoplasm_Diffuse_Large_B-cell_Lymphoma",
        "Ovarian_serous_cystadenocarcinoma", 
        "Cholangiocarcinoma", 
        "Uveal_Melanoma", 
        "Kidney_renal_papillary_cell_carcinoma", 
        "Colon_adenocarcinoma", 
        "Esophageal_carcinoma", 
        "Mesothelioma", 
        "Uterine_Corpus_Endometrial_Carcinoma", 
        "Stomach_adenocarcinoma", 
        "Lung_squamous_cell_carcinoma", 
        "Sarcoma", 
        "Breast_invasive_carcinoma", 
        "Brain_Lower_Grade_Glioma", 
        "Adrenocortical_carcinoma",
    ]

    write_results(log_dir=log_dir, classes=classes, ground_truth_df=gt_table_dir, save_path=log_dir)
    """
