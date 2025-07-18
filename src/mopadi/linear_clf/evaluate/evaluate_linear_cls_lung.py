# %%
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
from mopadi.train_linear_cls import ClsModel


load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")

# %%
def write_results(log_dir, classes, ground_truth_df, save_path):
    """
    Takes paths of npy files with obtained results, and writes final results 
    to a file patient-preds.csv in the form compatible with wanshi-utils.
    """
    y_pred = np.load(os.path.join(log_dir, "y_pred.npy"))

    pred_df = pd.DataFrame(y_pred, columns=classes)
    pred_df["pred"] = pred_df.idxmax(axis=1)
    print(pred_df)

    ground_truth_df = pd.read_csv(ground_truth_df, header=1, delimiter=' ')
    print(ground_truth_df.columns)
    ground_truth_df["CLASS"] = ground_truth_df[classes].idxmax(axis=1)
    ground_truth_df.drop(columns=classes, inplace=True)
    try:
        ground_truth_df.drop(columns=["Unnamed: 0"], inplace=True)
    except:
        pass
    ground_truth_df.reset_index(inplace=True)
    ground_truth_df.rename(columns={'index': 'lmdb_index'}, inplace=True)

    print(ground_truth_df)
    
    combined_df = pd.concat([ground_truth_df, pred_df], axis=1)
    for class_name in classes:
        combined_df[f"CLASS_{class_name}"] = combined_df[class_name]

    combined_df.drop(columns=classes, inplace=True)

    combined_df.to_csv(os.path.join(save_path, "patches-preds.csv"))
    

class Tester():
    def __init__(self, log_folder, model_config, checkpoint_path, cls_checkpoint_dir, cls_conf, device="cuda:0"):
        
        self.log_folder = log_folder

        self.model = LitModel(model_config)
        state = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state["state_dict"], strict=False)
        self.model.ema_model.to(device)

        self.cls_model = ClsModel(cls_conf)
        state = torch.load(cls_checkpoint_dir, map_location="cpu")
        print("latent step:", state["global_step"])
        self.cls_model.load_state_dict(state["state_dict"], strict=False)
        self.cls_model.to(device)

    def test(self, loader):
        label_list, pred_list = [], []
        with torch.no_grad():
            self.model.ema_model.eval()
            self.cls_model.ema_classifier.eval()
            for img_data in tqdm(loader, desc="Predicting"):
                data = img_data["img"]
                label = img_data["labels"]
                data, label = data.cuda().float(), label.cuda().float()
                latent = self.model.ema_model.encoder(data)
                latent = self.cls_model.normalize(latent)
                output = self.cls_model.ema_classifier.forward(latent)                
                pred = torch.sigmoid(output)
                
                label_list.append(label.cpu().numpy())
                pred_list.append(pred.cpu().numpy())
        
        pred = np.squeeze(np.array(pred_list))
        label = np.squeeze(np.array(label_list))
        np.save(os.path.join(self.log_folder, "y_pred.npy"), pred)
        np.save(os.path.join(self.log_folder, "y_true.npy"), label)
# %%

if __name__ == "__main__":
    # %%
    # Lung Cancer Types------------------------------------------------------------------------------------------------
    gt_table_dir = f"{ws_path}/mopadi/datasets/lung/lung_anno_val/list_attr.txt"
    checkpoint_dir = f"{ws_path}/mopadi/checkpoints/pancancer/last.ckpt"
    cls_checkpoint_dir = f"{ws_path}/mopadi/checkpoints/pancancer/lung_subtypes/last.ckpt"
    log_dir = f"{ws_path}/mopadi/checkpoints/pancancer/lung_subtypes/evaluation"
    dataset_dir = f"{ws_path}/mopadi/datasets/lung/subtypes-lmdb-val"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # configurations
    conf = pancancer_autoenc()
    cls_conf = liver_cancer_types_autoenc_cls()    

    tester = Tester(log_folder=log_dir, model_config=conf, checkpoint_path=checkpoint_dir, cls_checkpoint_dir=cls_checkpoint_dir,
                    cls_conf=cls_conf)
    # %%
    test_dataset = LungClsDataset(path=dataset_dir, attr_path=gt_table_dir)
    # %%
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    tester.test(test_loader)

    classes = [
        'Lung_adenocarcinoma', 
        'Lung_squamous_cell_carcinoma'
    ]
    # %%
    write_results(log_dir=log_dir, classes=classes, ground_truth_df=gt_table_dir, save_path=log_dir)

# %%
