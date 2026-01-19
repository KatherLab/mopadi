import os
import pandas as pd
import numpy as np
from PIL import Image
from dotenv import load_dotenv

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from mopadi.configs.templates import *
from mopadi.configs.templates_cls import *
from mopadi.linear_clf.train_linear_cls import ClsModel

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


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
    
class Tester():
    def __init__(self, model_config, checkpoint_path, cls_checkpoint_dir, cls_conf, device="cuda:0"):

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

        self.device = device

    def test(self, loader):
        with torch.no_grad():
            self.model.ema_model.eval()
            self.cls_model.ema_classifier.eval()
            for data, fname in tqdm(loader, desc="Predicting"):
                # print(fname)
                data = data.to(self.device).float()

                # encode the image into the latent space of the diffusion model
                latent = self.model.ema_model.encoder(data)
                latent = self.cls_model.normalize(latent)

                output = self.cls_model.ema_classifier.forward(latent)             
                pred = torch.sigmoid(output)
                
                directory_path = os.path.dirname(fname[0])
                with open(os.path.join(directory_path, "predictions-manipulated-formatted-new.txt"), "a") as f:
                    f.write(f"Filename: {fname}\n")
                    formatted_pred = [f"{number:.3f}" for number in pred.cpu().numpy().flatten()]
                    f.write(f"Pred (hcc, cca): {formatted_pred}\n")
                    f.write(f"\n") 

# legacy of old config system
def liver_cancer_types_cls():
    conf = pancancer_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = "liver_cancer_types"
    conf.manipulate_znormalize = True
    conf.feats_infer_path = f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/latent.pkl"
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 150_000
    conf.save_every_samples = 1_000
    conf.pretrain = PretrainConfig(
        "Pancancer",
        f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/last.ckpt",
    )
    conf.id_to_cls = ['Cholangiocarcinoma', 'Liver_hepatocellular_carcinoma']
    conf.name = "liver_cancer_types_linear_cls"
    return conf

if __name__ == "__main__":

    # LIVER CANCER TYPES -------------------------------------------------------------------------------------------------------
    checkpoint_dir = f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/last.ckpt"
    cls_checkpoint_dir = f'{ws_path}/mopadi/checkpoints/pancancer/liver_cancer_types_linear_cls/last.ckpt'
    path = f"{ws_path}/mopadi/checkpoints/pancancer/linear_liver_cancer_types_results/manipulate_results/TCGA-3X-AAVA-01Z-00-DX1-paper"

    # configurations
    conf = pancancer_autoenc()
    cls_conf = liver_cancer_types_cls()    

    tester = Tester(model_config=conf, checkpoint_path=checkpoint_dir, cls_checkpoint_dir=cls_checkpoint_dir, cls_conf=cls_conf)

    fnames = os.listdir(path)
    lvls = [0.2, 0.4, 0.6, 0.8, 1.0]

    for fname in tqdm(fnames):
        
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
