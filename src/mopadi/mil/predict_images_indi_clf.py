import os
import re
import torch
from tqdm import tqdm
from PIL import Image

try:
    from stamp.modeling.lightning_model import LitVisionTransformer
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "STAMP, which is used for independent classifier validation, is not installed."
        " Follow instructions at https://github.com/KatherLab/STAMP to install it."
    ) from e

from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


def to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Return probabilities from logits for CE(>=2 classes) or BCE(1 class)."""
    return torch.sigmoid(logits) if logits.size(-1) == 1 else torch.softmax(logits, dim=-1)

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

@torch.no_grad()
def extract_feats(pil_img: Image.Image, extractor, device: str = "cuda") -> torch.Tensor:
    """Extract features for a single PIL image."""
    x = extractor.transform(pil_img.convert("RGB")).unsqueeze(0).to(device)  # [1,3,H,W]
    feats = extractor.model(x)  # [1,D]
    return feats

@torch.no_grad()
def predict_pil_stamp(extractor, stamp_model, pil_img: Image.Image, device: str = "cuda"):
    """Run on-the-fly feat extraction → STAMP classifier"""
    feats = extract_feats(pil_img, extractor, device=device)  # [1,D]
    bag = feats.unsqueeze(1)
    B, N, _ = bag.shape

    # dummy coords & bag_sizes for single-tile inference
    coords = torch.zeros((B, N, 2), dtype=torch.float32, device=device)  # [B, N, 2]
    mask = torch.zeros((B, N), dtype=torch.bool, device=device)

    logits = stamp_model(bags=bag, coords=coords, mask=mask)
    #print(logits)
    probs = to_probs(logits).squeeze(0).cpu()
    return probs


def main(
    base_dir: str,
    stamp_ckpt: str,
    encoder: str,
    device: str = "cuda:0",
    class_names=("nonMSIH", "MSIH"),
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print(f"Loading STAMP ViT classifier from {stamp_ckpt}")
    lit = LitVisionTransformer.load_from_checkpoint(stamp_ckpt, map_location="cpu")
    stamp_model  = lit.vision_transformer.to(device).eval()

    if encoder.lower() == "conch":
        from stamp.preprocessing.extractor.conch import conch
        print("Loading CONCH extractor (MahmoodLab/conch)")
        extractor = conch()
        extractor.model.to(device).eval()
    elif encoder.lower() == "virchow2":
        from stamp.preprocessing.extractor.virchow2 import virchow2
        print("Loading Virchow2 extractor")
        extractor = virchow2()
        extractor.model.to(device).eval()
    elif encoder.lower() == "uni2":
        from stamp.preprocessing.extractor.uni2 import uni2
        print("Loading UNI2 extractor")
        extractor = uni2()
        extractor.model.to(device).eval()

    for patient in tqdm(sorted(os.listdir(base_dir))):
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

            out_txt = os.path.join(tile_dir, f"predictions_stamp_{encoder}.txt")
            with open(out_txt, "w") as f:
                f.write(f"Patient: {patient}\nTile: {tile_folder}\n\n")

                if original_path is not None:
                    try:
                        img = Image.open(original_path).convert("RGB")
                        probs = predict_pil_stamp(extractor, stamp_model, img, device=device)
                        prob_list = [float(p) for p in probs.tolist()]
                        if len(prob_list) == len(class_names):
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
                    cf_img = Image.open(cf_path).convert("RGB")
                    probs = predict_pil_stamp(extractor, stamp_model, cf_img, device=device)
                    prob_list = [float(p) for p in probs.tolist()]
                    if len(prob_list) == len(class_names):
                        f.write(f"  amp={amp} | {os.path.basename(cf_path)}\n")
                        for cn, p in zip(class_names, prob_list):
                            f.write(f"    {cn}: {p:.3f}\n")
                    else:
                        f.write(f"  amp={amp} | {os.path.basename(cf_path)} | probs={prob_list}\n")


if __name__ == "__main__":

    BASE_DIR = f"{ws_path}/mopadi/checkpoints/crc-paper/mil_classifier_isMSIH/counterfactuals"
    #STAMP_CKPT = f"{ws_path}/indi_clf/conch/model.ckpt"
    #STAMP_CKPT = f"{ws_path}/indi_clf/uni2/model.ckpt"
    STAMP_CKPT = f"{ws_path}/indi_clf/v2/model.ckpt"

    lit = LitVisionTransformer.load_from_checkpoint(STAMP_CKPT, map_location="cpu")
    print("Checkpoint categories:", getattr(lit, "categories", None))
    print("Hparams keys:", lit.hparams.keys())
    if "categories" in lit.hparams:
        print("Hparams categories:", lit.hparams["categories"])

    CLASS_NAMES = tuple(str(c) for c in lit.hparams["categories"])

    main(BASE_DIR, STAMP_CKPT, encoder="virchow2", device="cuda:0", class_names=CLASS_NAMES)
