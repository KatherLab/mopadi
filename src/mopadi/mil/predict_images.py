import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from mopadi.train_diff_autoenc import LitModel
from mopadi.configs.templates import tcga_crc_autoenc
from mopadi.configs.templates_cls import crc_pretrained_mil
from mopadi.mil.set_transformer import PMA, SAB

from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")

class Classifier(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, num_classes, ln=False):
        super().__init__()
        self.dim = dim
        self.layer_norm = nn.LayerNorm(normalized_shape=dim)
        self.pool = nn.Sequential(
            PMA(dim, num_heads, num_seeds, ln),
            SAB(dim, dim, num_heads, ln),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(dim, num_classes),
        )
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.pool(x).max(1).values
        return self.classifier(x)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

def to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Return probabilities from logits for CE(>=2 classes) or BCE(1 class)."""
    return torch.sigmoid(logits) if logits.size(-1) == 1 else torch.softmax(logits, dim=-1)

@torch.no_grad()
def predict_pil(lit_model: LitModel, mil_classifier: nn.Module, pil_img: Image.Image, device="cuda:0"):
    """PIL -> encoder -> MIL -> probs"""
    enc_owner = getattr(lit_model, "ema_model", lit_model)
    encoder = enc_owner.encoder

    x = preprocess(pil_img.convert("RGB")).unsqueeze(0).to(device)  # [1,3,H,W]
    feats = encoder(x)                                              # [1,D] or similar
    if feats.ndim != 2:
        feats = feats.view(1, -1)
    bag = feats.unsqueeze(1)                                        # [1,1,D]
    logits = mil_classifier(bag)                                    # [1,C]
    probs = to_probs(logits).squeeze(0).cpu()
    return probs

def find_images_in_tile_dir(tile_dir: str):
    """
    Return:
      - original_path: str or None
      - list of (amp_str, cf_path) for counterfactuals
    """
    original_path = None
    counterfactuals = []

    # originals typically contain '_0_original_'
    for fn in os.listdir(tile_dir):
        if "_0_original_" in fn and fn.lower().endswith(".png"):
            original_path = os.path.join(tile_dir, fn)
            break

    # counterfactuals typically contain 'manip_to_*_amp_0,060.png' (comma or dot)
    amp_re = re.compile(r"amp_([0-9]+[,.][0-9]+)")
    for fn in os.listdir(tile_dir):
        if "manip_to_" in fn and "amp_" in fn and fn.lower().endswith(".png"):
            m = amp_re.search(fn)
            amp = m.group(1) if m else "NA"
            counterfactuals.append((amp, os.path.join(tile_dir, fn)))

    # Sort CFs by numeric amplitude if possible
    def amp_key(a):
        s = a[0].replace(",", ".")
        try:
            return float(s)
        except:
            return 1e9
    counterfactuals.sort(key=amp_key)
    return original_path, counterfactuals

def main(
    base_dir: str,
    autoenc_ckpt: str,
    mil_ckpt: str,
    device: str = "cuda:0",
    class_names=("nonMSIH", "MSIH"),
):
    lit_model = LitModel(tcga_crc_autoenc())
    state = torch.load(autoenc_ckpt, map_location="cpu")
    lit_model.load_state_dict(state["state_dict"], strict=False)
    lit_model.to(device).eval()

    conf_cls = crc_pretrained_mil()
    mil_classifier = Classifier(conf_cls.dim, conf_cls.num_heads, conf_cls.num_seeds, conf_cls.num_classes)
    mil_classifier.load_state_dict(torch.load(mil_ckpt, map_location="cpu"))
    mil_classifier.to(device).eval()

    # base_dir/<PATIENT>/<Tile_(x,y)>/*.png
    for patient in sorted(os.listdir(base_dir)):
        pat_dir = os.path.join(base_dir, patient)
        if not os.path.isdir(pat_dir):
            continue

        for tile_folder in sorted(os.listdir(pat_dir)):
            tile_dir = os.path.join(pat_dir, tile_folder)
            if not os.path.isdir(tile_dir):
                continue

            original_path, cf_list = find_images_in_tile_dir(tile_dir)
            if original_path is None and not cf_list:
                continue

            out_txt = os.path.join(tile_dir, "predictions-new.txt")
            with open(out_txt, "w") as f:
                f.write(f"Patient: {patient}\nTile: {tile_folder}\n\n")

                if original_path is not None:
                    try:
                        orig_img = Image.open(original_path).convert("RGB")
                        probs = predict_pil(lit_model, mil_classifier, orig_img, device=device)  # [C]

                        if probs.ndim == 0:  # safety
                            probs = probs.unsqueeze(0)
                        prob_list = [float(p) for p in probs.tolist()]
                        if len(class_names) == len(prob_list):
                            f.write("Original image prediction:\n")
                            for cn, p in zip(class_names, prob_list):
                                f.write(f"  {cn}: {p:.3f}\n")
                        else:
                            f.write(f"Original probs: {prob_list}\n")
                    except Exception as e:
                        f.write(f"Original prediction failed: {e}\n")
                else:
                    f.write("Original image not found.\n")

                f.write("\nCounterfactual predictions:\n")

                for amp, cf_path in cf_list:
                    try:
                        cf_img = Image.open(cf_path).convert("RGB")
                        probs = predict_pil(lit_model, mil_classifier, cf_img, device=device)
                        prob_list = [float(p) for p in probs.tolist()]
                        if len(class_names) == len(prob_list):
                            f.write(f"  amp={amp} | {os.path.basename(cf_path)}\n")
                            for cn, p in zip(class_names, prob_list):
                                f.write(f"    {cn}: {p:.3f}\n")
                        else:
                            f.write(f"  amp={amp} | {os.path.basename(cf_path)} | probs={prob_list}\n")
                    except Exception as e:
                        f.write(f"  amp={amp} | {os.path.basename(cf_path)} | FAILED: {e}\n")

            print(f"Wrote: {out_txt}")

if __name__ == "__main__":

    BASE_DIR = f"{ws_path}/mopadi/checkpoints/crc-paper/mil_classifier_isMSIH/counterfactuals"
    AUTOENC_CKPT = "/home/laura/.cache/huggingface/hub/models--KatherLab--MoPaDi/snapshots/5d8e775e24473c5d8f4c0c57fd5c865c3c2a4aab/crc_512_model/autoenc.ckpt"
    MIL_CKPT = "/home/laura/.cache/huggingface/hub/models--KatherLab--MoPaDi/snapshots/5d8e775e24473c5d8f4c0c57fd5c865c3c2a4aab/crc_512_model/mil_msi_classifier.pth"

    CLASS_NAMES = ("nonMSIH", "MSIH")

    main(BASE_DIR, AUTOENC_CKPT, MIL_CKPT, device="cuda:0", class_names=CLASS_NAMES)
