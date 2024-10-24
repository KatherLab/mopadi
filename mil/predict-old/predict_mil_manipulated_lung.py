import os
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from mil.utils import *

from configs.templates import *
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


class LungDataset(Dataset):
    def __init__(self, results_dir, test_man_amp, transform=None):
        self.transform = transform
        self.results_dir = results_dir
        self.test_man_amp = test_man_amp
        self.patients = [os.path.join(results_dir, name) for name in sorted(os.listdir(results_dir)) if os.path.isdir(os.path.join(results_dir, name))]
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
            fnames = sorted([os.path.join(subfolder, f) for f in os.listdir(subfolder) if os.path.join(subfolder, f).endswith("png")])
            for fname in fnames:
                if self.test_man_amp in fname:
                    image = Image.open(fname).convert("RGB")
                    if self.transform:
                        image = self.transform(image)
                    data[subfolder].append(image)
            if len(data[subfolder])==0:
                print(f"Did not find image with {self.test_man_amp} for patient {subfolder}")

        return data


class Tester():
    def __init__(self,
                model_config, 
                autoenc_path, 
                mil_path, 
                latent_infer_path, 
                device="cuda:1",
                dim = 512,
                num_heads = 8,
                num_seeds = 4,
                num_classes = 2,
):
        self.device = device

        # load diffusion autoencoder
        self.model = LitModel(model_config)
        state = torch.load(autoenc_path, map_location="cpu")
        self.model.load_state_dict(state["state_dict"], strict=False)
        self.model.ema_model.eval()
        self.model.ema_model.to(self.device)

        # load the classifier
        self.cls_model = Classifier(dim, num_heads, num_seeds, num_classes)
        weights = torch.load(mil_path)
        self.cls_model.load_state_dict(weights)
        self.cls_model = self.cls_model.to(self.device)
        self.latent_state = torch.load(latent_infer_path)


    def test(self, loader, man_amp):
        with torch.no_grad():
            self.model.eval()

            for data in tqdm(loader, desc="Predicting"):

                feats = []
                for fname in data.keys():
                    if len(data[fname])>0:
                        img = data[fname][0]
                    else:
                        continue
                    print(f"Img size: {img.size()}")
                    return
                    feats.append(self.model.encode(img.to(self.device)))

                if feats:
                    feats_patient = torch.cat(feats, dim=0)
                else:
                    continue

                

                logits = self.cls_model(feats_patient.unsqueeze(0).to(self.device))
                # _, predicted_labels = torch.max(logits, dim=1)
                #pred = F.softmax(logits, dim=1)
                pred = torch.sigmoid(logits) 

                parent_directory = os.path.dirname(fname)

                with open(os.path.join(parent_directory, "predictions.txt"), "a") as f:
                    f.write(f"-----------------------------------\n")
                    f.write(f"Manipulation amplitude: {man_amp}\n")  
                    f.write(f"Pred (LUSC, LUAD): {[f'{p:.3f}' for p in pred.squeeze().cpu().numpy()]}\n")
                    f.write(f"\n")


if __name__ == "__main__":

    autoenc_path = "checkpoints/pancancer/last.ckpt"
    mil_path = "checkpoints/lung/lung-subtypes-crossval-layernorm/full_model/PMA_mil.pth"
    latent_infer_path = "checkpoints/pancancer/latent.pkl"
    results_dir = "checkpoints/lung/lung-subtypes-crossval-layernorm/full_model/manip-tiles-together"
    test_man_amps = ["original", "0,010", "0,020"]#, "0,04", "0.045", "0,050", "0,055", "0,06"]

    conf = pancancer_autoenc()

    tester = Tester(model_config=conf, 
                    autoenc_path=autoenc_path, 
                    mil_path=mil_path,
                    latent_infer_path=latent_infer_path,
                    )

    for man_amp in test_man_amps:

        data = LungDataset(
                        results_dir=results_dir,
                        test_man_amp=man_amp,
                        transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]))

        test_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1)
        tester.test(test_loader, man_amp)
