# %%
import os
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

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")

# %%
def write_results(log_dir, classes, ground_truth_df, save_path):
    """
    Takes paths of npy files with obtained results, and writes final results 
    to a file patient-preds.csv in the form compatible with wanshi-utils.
    Not compatible if custom sampler was used!
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


class CustomSampler(Sampler):
    def __init__(self, data_source, valid_indices):
        """
        :param data_source: The dataset object
        :param valid_indices: List of indices that are valid and should be sampled
        """
        self.data_source = data_source
        self.valid_indices = valid_indices

    def __iter__(self):
        return iter(self.valid_indices)

    def __len__(self):
        return len(self.valid_indices)
    

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
            for idx in tqdm(range(length)):
                filename_key = f"filename_{str(idx).zfill(5)}".encode("utf-8")
                filename = txn.get(filename_key).decode("utf-8")
                fname = filename.split("/")[-2] + "/" + filename.split("/")[-1]
                if fname in dataset.valid_fnames:
                    valid_indices.append(idx)
    print(f"Number of respective classes images found: {len(valid_indices)}")
    return valid_indices
# %%

if __name__ == "__main__":
    """
    From the given LMDB dataset only classes defined in the dataset class (e.g., LiverCancerTypesClsDataset)
    are taken. Therefore, LMDB dataset need to have filename_key, which is then matched to the gt_table 
    to get the ground truth class of that image.
    """

    # %%
    # Pancancer
    gt_table_dir = os.path.expanduser('datasets/pancancer/list_classes_all.txt')
    checkpoint_dir = f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/last.ckpt"
    dataset_dir = f"{ws_path}/mopadi/datasets/pancancer/japan-lmdb-test"

    # Liver
    #cls_checkpoint_dir = f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/liver_cancer_types_linear_cls/last.ckpt"
    #log_dir = f"{ws_path}/mopadi/checkpoints/pancancer/linear_liver_cancer_types_results"
    #cls_conf = liver_cancer_types_cls()
    #test_dataset = LiverCancerTypesClsDataset(path=dataset_dir)

    # Lung
    cls_checkpoint_dir = f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/lung_subtypes_linear_cls/last.ckpt"
    log_dir = f"{ws_path}/mopadi/checkpoints/pancancer/linear_lung_subtypes_results"
    cls_conf = lung_linear_cls()  
    test_dataset = LungClsDataset(path=dataset_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # configurations
    conf = pancancer_autoenc() 

    tester = Tester(log_folder=log_dir, 
                    model_config=conf, 
                    checkpoint_path=checkpoint_dir, 
                    cls_checkpoint_dir=cls_checkpoint_dir,
                    cls_conf=cls_conf)
    # %%

    valid_indices = get_valid_indices(test_dataset)

    sampler = CustomSampler(test_dataset, valid_indices)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, sampler=sampler)
    tester.test(test_loader)

# %%
