"""
Compute LPIPS between original and reconstructed tiles.

Reconstructed images are sequential PNGs (reconstructed_image_XXXXX.png)
that map 1:1 to rows in the CSV by row index.
Originals are found by matching Patient (prefix) + Filename under tiles_path.

Usage:
    python compute_lpips.py \
        --recon_dir /path/to/reconstructed_images-new/TCGA-CRC \
        --tiles_path /path/to/TCGA-CRC/512x512_tumor_test \
        --output results_lpips_crc.csv \
        [--device cuda:0]
"""

import argparse
import os
import glob
import pandas as pd
import torch
import lpips
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def find_original(tiles_path, patient_id, filename):
    # Search at depth 1: tiles_path/PatientFolder*/filename
    for match in glob.glob(os.path.join(tiles_path, patient_id + "*", filename)):
        if os.path.exists(match):
            return match
    # Search at depth 2: tiles_path/*/PatientFolder*/filename (e.g. JAPAN cancer type subdir)
    for match in glob.glob(os.path.join(tiles_path, "*", patient_id + "*", filename)):
        if os.path.exists(match):
            return match
    # Fallback: flat structure
    candidate = os.path.join(tiles_path, filename)
    if os.path.exists(candidate):
        return candidate
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_dir", required=True, help="Directory with reconstructed_image_XXXXX.png and CSV")
    ap.add_argument("--tiles_path", required=True, help="Root directory of original tiles")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    csv_path = os.path.join(args.recon_dir, "autoencoder-evaluation-results-final.csv")
    df = pd.read_csv(csv_path)

    loss_fn = lpips.LPIPS(net="alex").to(args.device)
    loss_fn.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    lpips_scores = []
    missing = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="LPIPS"):
        recon_path = os.path.join(args.recon_dir, f"reconstructed_image_{idx:05d}.png")

        filename = row["Filename"]

        if "Patient" in df.columns:
            orig_path = find_original(args.tiles_path, str(row["Patient"]), filename)
        elif "Class" in df.columns:
            orig_path = os.path.join(args.tiles_path, str(row["Class"]), filename)
        else:
            orig_path = os.path.join(args.tiles_path, filename)

        if orig_path is None or not os.path.exists(orig_path) or not os.path.exists(recon_path):
            lpips_scores.append(None)
            missing += 1
            continue

        orig = to_tensor(Image.open(orig_path).convert("RGB")).unsqueeze(0).to(args.device)
        recon = to_tensor(Image.open(recon_path).convert("RGB")).unsqueeze(0).to(args.device)

        with torch.no_grad():
            score = loss_fn(orig, recon).item()
        lpips_scores.append(score)

    df["LPIPS"] = lpips_scores
    df.to_csv(args.output, index=False)

    valid = [s for s in lpips_scores if s is not None]
    print(f"Done. {len(valid)}/{len(df)} pairs computed (missing: {missing})")
    print(f"Mean LPIPS: {sum(valid)/len(valid):.4f}")


if __name__ == "__main__":
    main()
