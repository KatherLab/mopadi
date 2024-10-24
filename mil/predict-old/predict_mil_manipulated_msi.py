import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from mil.utils import *

from configs.templates import *
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


class Tester():
    def __init__(self,
                #log_folder, 
                model_config, 
                autoenc_path, 
                mil_path, 
                latent_infer_path, 
                man_amp,
                target_dict,
                device="cuda:0",
                dim = 512,
                num_heads = 8,
                num_seeds = 4,
                num_classes = 2,
):
        self.device = device
        self.man_amp = man_amp
        self.target_dict = target_dict

        # load diffusion autoencoder
        self.model = LitModel(model_config)
        state = torch.load(autoenc_path, map_location="cpu")
        self.model.load_state_dict(state["state_dict"], strict=False)
        self.model.ema_model.to(self.device)

        # load the classifier
        self.cls_model = Classifier(dim, num_heads, num_seeds, num_classes)
        weights = torch.load(mil_path)
        self.cls_model.load_state_dict(weights)
        self.cls_model = self.cls_model.to(self.device)
        self.latent_state = torch.load(latent_infer_path)


    def test(self, loader):
        with torch.no_grad():
            self.model.eval()

            for data in tqdm(loader, desc="Predicting"):

                feats = []
                for fname in data.keys():
                    if len(data[fname])>0:
                        img = data[fname][0]
                    else:
                        continue
                    feats.append(self.model.encoder(img.to(self.device)))

                if feats:
                    feats_patient = torch.cat(feats, dim=0)
                else:
                    continue

                # latent = normalize(latent, self.latent_state, device=self.device)
                logits = self.cls_model(feats_patient.unsqueeze(dim=0).to(self.device))
                # _, predicted_labels = torch.max(logits, dim=1)
                pred = F.softmax(logits, dim=1)  

                parent_directory = os.path.dirname(fname)

                with open(os.path.join(parent_directory, "predictions.txt"), "a") as f:
                    f.write(f"Manipulation amplitude: {self.man_amp}\n")  
                    f.write(f"Pred ({self.target_dict[0]}, {self.target_dict[1]}): {[pred.cpu().numpy()]}\n")
                    f.write(f"\n")


if __name__ == "__main__":

    autoenc_path = "checkpoints/tcga_crc_224x224/last.ckpt"
    mil_path = "checkpoints/msi-final-ppt-512/PMA_mil.pth"
    latent_infer_path = "checkpoints/tcga_crc_224x224/latent.pkl"
    results_dir = f"{ws_path}/results-manipulated/msi-final-ppt-512"
    conf = tcga_crc_autoenc()
    target_dict = {0: "nonMSIH", 1: "MSIH"}
    test_man_amps = ["original", "0,40", "0,80", "1,20", "1,60", "2,00"]

    for man_amp in test_man_amps:

        tester = Tester(model_config=conf, 
                        autoenc_path=autoenc_path, 
                        mil_path=mil_path,
                        latent_infer_path=latent_infer_path,
                        man_amp=man_amp,
                        target_dict=target_dict,
                        )

        data = TCGADataset(
                       images_dir=results_dir,
                        test_man_amp=man_amp,
                        transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]))

        test_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1)
        tester.test(test_loader)
