#%%
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

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")
#%%
    
class Tester():
    def __init__(self, log_folder, model_config, checkpoint_path, cls_checkpoint_dir, cls_conf, device="cuda:0"):
        
        self.log_folder = log_folder

        # load diffusion autoencoder
        self.model = LitModel(model_config)
        state = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state["state_dict"], strict=False)
        self.model.ema_model.to(device)

        # load latent space classifier
        self.cls_model = ClsModel(cls_conf)
        state = torch.load(cls_checkpoint_dir, map_location="cpu")
        print("latent step:", state["global_step"])
        self.cls_model.load_state_dict(state["state_dict"], strict=False)
        self.cls_model.to(device)

    def test(self, loader):
        with torch.no_grad():
            self.model.ema_model.eval()
            self.cls_model.ema_classifier.eval()
            for data, fname in tqdm(loader, desc="Predicting"):
                # print(fname)
                data = data.cuda().float()

                # encode the image into the latent space of the diffusion model
                latent = self.model.ema_model.encoder(data)
                latent = self.cls_model.normalize(latent)

                output = self.cls_model.ema_classifier.forward(latent)             
                pred = torch.sigmoid(output)
                
                directory_path = os.path.dirname(fname[0])
                with open(os.path.join(directory_path, "predictions-manipulated-formatted.txt"), "a") as f:
                    f.write(f"Filename: {fname}\n")
                    formatted_pred = [f"{number:.3f}" for number in pred.cpu().numpy().flatten()]
                    f.write(f"Pred (hcc, cca): {formatted_pred}\n")
                    f.write(f"\n") 

#%%
if __name__ == "__main__":
    #%%
    # LIVER CANCER TYPES -------------------------------------------------------------------------------------------------------
    #gt_table_dir = f"{ws_path}/mopadi/datasets/liver/types-lmdb-test-anno/list_attr.txt"
    checkpoint_dir = f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/last.ckpt"
    cls_checkpoint_dir = f'{ws_path}/mopadi/checkpoints/pancancer/autoenc/liver_cancer_types_linear_cls/last.ckpt'
    log_dir = f"{ws_path}/mopadi/checkpoints/pancancer/liver_cancer_types/evaluation"

    # configurations
    conf = pancancer_autoenc()
    cls_conf = liver_cancer_types_cls()    
    # %%
    tester = Tester(log_folder=log_dir, model_config=conf, checkpoint_path=checkpoint_dir, cls_checkpoint_dir=cls_checkpoint_dir,
                    cls_conf=cls_conf)
    # %%
    path = f"/mnt/bulk-mars/laura/diffae/results/liver-newest/TCGA-3X-AAVA-01Z-00-DX1"
    fnames = os.listdir(path)
    lvls = [0.2, 0.4, 0.6, 0.8, 1.0]
    #%%
    for fname in tqdm(fnames):
        #if os.path.exists(os.path.join(path, fname, "predictions-manipulated-formatted.txt")):
        #    print("Skipping...")
        #    continue
        
        image_path = os.path.join(path, fname, "_".join(fname.split("_")[:4]) + f"_original.png")

        if not os.path.isfile(image_path):
            print(f"skipped {image_path}")
            continue
        
        test_dataset = ImgDataset(image_path=image_path,
                                    transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]))

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
        tester.test(test_loader)

        for target_class in ['Liver_hepatocellular_carcinoma', 'Cholangiocarcinoma']:

            for lvl in lvls:
                image_path = os.path.join(path, fname, "_".join(fname.split("_")[:4]) + f"_manipulated_to_{target_class}_amplitude_{lvl}.png")
                print(image_path)
                if not os.path.isfile(image_path):
                    print(f"skipped {image_path}")
                    continue
                test_dataset = ImgDataset(image_path=image_path,
                                            transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]))

                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
                tester.test(test_loader)

# %%
