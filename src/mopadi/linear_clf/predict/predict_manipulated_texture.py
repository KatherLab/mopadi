import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from linear_clf.train_linear_cls import ClsModel
from mil.utils import *

from configs.templates import *
from configs.templates_cls import *
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


class Tester():
    def __init__(self,
                model_config,
                cls_config,
                autoenc_path, 
                cls_checkpoint_dir, 
                man_amp,
                device="cuda:0",
):
        self.device = device
        self.man_amp = man_amp

        # load diffusion autoencoder
        self.model = LitModel(model_config)
        state = torch.load(autoenc_path, map_location="cpu")
        self.model.load_state_dict(state["state_dict"], strict=False)
        self.model.ema_model.to(self.device)

        # load the classifier
        self.cls_model = ClsModel(cls_conf)
        state = torch.load(cls_checkpoint_dir, map_location="cpu")
        print("latent step:", state["global_step"])
        self.cls_model.load_state_dict(state["state_dict"], strict=False)
        self.cls_model.to(device)


    def test(self, image, save_dir):

        target_list = TextureAttrDataset.id_to_cls

        with torch.no_grad():
            self.model.ema_model.eval()
            self.cls_model.ema_classifier.eval()

            latent = self.model.ema_model.encoder(image.cuda().float())
            latent = self.cls_model.normalize(latent)
            output = self.cls_model.ema_classifier.forward(latent)                
            pred = torch.softmax(output, dim=1)
            _, max_index = torch.max(pred, 1)
            predicted_class = target_list[max_index.item()]

            with open(os.path.join(save_dir, "predictions.txt"), "a") as f:
                f.write(f"Manipulation amplitude: {self.man_amp}\n")  
                f.write(f"Pred {target_list}: {[pred.cpu().numpy()]}\n")
                f.write(f"Pred class: {predicted_class}\n")
                f.write(f"\n")


if __name__ == "__main__":

    autoenc_path = "checkpoints/texture100k/last.ckpt"
    cls_checkpoint_dir = "checkpoints/texture100k/texture100k_clf/last.ckpt"
    results_dir = "images/example_norm_to_tumor"

    test_man_amps = ["original", "0.2", "0.4", "0.6", "0.8", "1.0"]

    conf = texture100k_autoenc()
    cls_conf = texture100k_linear_cls()

    transform = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform)

    for man_amp in test_man_amps:

        for fname in os.listdir(results_dir):
            if man_amp in fname:
                print(fname)
                image = Image.open(os.path.join(results_dir, fname)).convert('RGB')
                image = transform(image)

                tester = Tester(model_config=conf, 
                                cls_config=cls_conf,
                                autoenc_path=autoenc_path, 
                                cls_checkpoint_dir=cls_checkpoint_dir,
                                man_amp=man_amp)

                tester.test(image.unsqueeze(0), results_dir)

