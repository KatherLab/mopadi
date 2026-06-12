import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from cmcrameri import cm
from PIL import Image
from torchvision import transforms
from torchvahadane import TorchVahadaneNormalizer
from skimage import color
from dotenv import load_dotenv
import torch
import h5py

from mopadi.dataset import DefaultTilesDataset
from mopadi.configs.templates import tcga_crc_autoenc
from mopadi.configs.templates_cls import crc_pretrained_mil
from mopadi.mil.manipulate.manipulator_mil import ImageManipulator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


def to_model_tensor(img_uint8):
    x = torch.from_numpy(img_uint8.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
    return x

def ensure_uint8_rgb(img) -> np.ndarray:
    """
    Accept NumPy or torch.Tensor, return HxWx3 uint8 RGB array.
    Handles:
    - HxW, HxWx1, HxWx3
    - CHW (3,H,W) or (1,H,W)
    - batched (1,3,H,W) or (1,H,W,3)
    """
    # torch.Tensor -> np
    if isinstance(img, torch.Tensor):
        t = img.detach().cpu()
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]                    # (C,H,W) or (H,W,3)
        if t.ndim == 3 and t.shape[0] in (1, 3):  # CHW -> HWC
            t = t.permute(1, 2, 0)
        img = t.numpy()

    # must be np.ndarray from here
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(img)}")

    # squeeze 1-length batch dim if present
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]  # now (3,H,W) or (H,W,3)

    # CHW (3,H,W) -> HWC
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    # grayscale -> RGB
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    # dtype/range
    if img.dtype in (np.float32, np.float64):
        # assume either [0,1] or [0,255]; scale if in [0,1]
        vmax = float(np.nanmax(img)) if img.size else 1.0
        if vmax <= 1.0 + 1e-3:
            img = (img * 255.0)
        img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # final sanity
    if not (img.ndim == 3 and img.shape[-1] == 3):
        raise AssertionError(f"Expected HxWx3 uint8, got {img.shape}")
    return img


def rgb_to_gray3(img_uint8: np.ndarray) -> np.ndarray:
    img_uint8 = ensure_uint8_rgb(img_uint8)
    g = color.rgb2gray(img_uint8.astype(np.float32) / 255.0)  # float64 0..1
    g3 = np.stack([g,g,g], axis=-1)
    return (np.clip(g3 * 255, 0, 255)).astype(np.uint8)

@torch.no_grad()
def predict_logit(manipulator, img) -> torch.Tensor:
    """
    Returns logits [num_classes] using the same path as training:
    encoder(x) -> make 1-tile bag -> full MIL classifier (LN+PMA+SAB+MLP).
    """
    img_uint8 = ensure_uint8_rgb(img)

    x = transforms.ToTensor()(img_uint8)                # [3,H,W] in [0,1]
    x = manipulator.normalizer(x).unsqueeze(0).to(manipulator.device)   # [1,3,H,W]

    enc_owner = getattr(manipulator.model, "ema_model", manipulator.model)

    encoder = getattr(enc_owner, "encoder", None)
    encoder.eval()
    manipulator.classifier.eval()

    if encoder is None:
        raise AttributeError("manipulator.model (or ema_model) has no .encoder")

    feats = encoder(x)                                   # [1,D] (MoPaDi enc) or similar
    if feats.ndim != 2:                                  # safety: flatten if needed
        feats = feats.view(feats.size(0), -1)

    bag = feats.unsqueeze(1)                             # [1,1,D] — a single-tile bag
    logits = manipulator.classifier(bag)                         # [1,C], full MIL path
    return logits.squeeze(0)


# --- 1b) Vahadane normalize (TorchVahadane backend) -------------------------
def vahadane_normalize(source_uint8, target_uint8):
    norm = TorchVahadaneNormalizer(device='cuda', staintools_estimate=True)
    norm.fit(ensure_uint8_rgb(target_uint8))
    out = norm.transform(ensure_uint8_rgb(source_uint8))
    #print("after vahadane:", out.min(), out.max(), out.dtype, out.mean())
    return ensure_uint8_rgb(out)   # <- ensure NumPy uint8 for downstream


# --- 2) Decomposition A: style-structure swap (Vahadane) --------------------
def decompose_style_vs_morph_vahadane(ctx, x_uint8, xcf_uint8):
    # total effect on RGB
    s0   = predict_logit(ctx, x_uint8)
    s_cf = predict_logit(ctx, xcf_uint8)
    d_total = s_cf - s0

    # hybrids using Vahadane
    x_style = vahadane_normalize(x_uint8, xcf_uint8)   # structure x, style x_cf
    x_morph = vahadane_normalize(xcf_uint8, x_uint8)   # structure x_cf, style x

    s_style = predict_logit(ctx, x_style)
    s_morph = predict_logit(ctx, x_morph)

    d_style = s_style - s0
    d_morph = s_morph - s0
    interaction = d_total - (d_style + d_morph)

    phi_style = d_style + 0.5 * interaction
    phi_morph = d_morph + 0.5 * interaction

    return dict(
        d_total=d_total,
        phi_style=phi_style,
        phi_morph=phi_morph,
        s0=s0, s_cf=s_cf, s_style=s_style, s_morph=s_morph,
        x_style=x_style, x_morph=x_morph
    )


# --- 3) Decomposition B: grayscale proxy -----------------------------------
def decompose_gray_proxy(ctx, x_uint8, xcf_uint8):
    s0   = predict_logit(ctx, x_uint8)
    s_cf = predict_logit(ctx, xcf_uint8)
    d_total = s_cf - s0

    gx   = rgb_to_gray3(x_uint8)
    gcf  = rgb_to_gray3(xcf_uint8)
    s0g  = predict_logit(ctx, gx)
    scfg = predict_logit(ctx, gcf)
    d_gray = scfg - s0g   # morphology-biased lower bound

    return dict(d_total=d_total, morph_lb=d_gray, style_resid=d_total - d_gray)


# --- 4) Optional: tensor -> uint8 for visual checks -------------------------
def tensor_to_uint8_rgb(img):
    # Torch tensor → NumPy
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    # Float [0,1] → scale up
    if img.dtype in (np.float32, np.float64):
        if img.max() <= 1.0 + 1e-3:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    # Grayscale → 3 channels
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    return img

def to_pil(a):
    return Image.fromarray(ensure_uint8_rgb(a))

# ---- visualize ------------------------------------------------
def visualize_decomposition(ctx, x_uint8, xcf_uint8, save_dir=None, prefix="viz"):
    """
    ctx: your ImageManipulator instance
    x_uint8, xcf_uint8: HxWx3 uint8 arrays
    """
    # run your existing function to get hybrids
    out = decompose_style_vs_morph_vahadane(ctx, x_uint8, xcf_uint8)
    x_style = ensure_uint8_rgb(out["x_style"])
    x_morph = ensure_uint8_rgb(out["x_morph"])
    x0      = ensure_uint8_rgb(x_uint8)
    xcf     = ensure_uint8_rgb(xcf_uint8)

    # 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.ravel()
    panels = [
        ("Original (x)", x0),
        ("Counterfactual (x_cf)", xcf),
        ("Style-hybrid (structure x, style x_cf)", x_style),
        ("Morph-hybrid (structure x_cf, style x)", x_morph),
    ]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"{prefix}_grid.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        # also save raw images if you like
        to_pil(x0).save(os.path.join(save_dir, f"{prefix}_orig.png"))
        to_pil(xcf).save(os.path.join(save_dir, f"{prefix}_cf.png"))
        to_pil(x_style).save(os.path.join(save_dir, f"{prefix}_styleHybrid.png"))
        to_pil(x_morph).save(os.path.join(save_dir, f"{prefix}_morphHybrid.png"))
        print(f"Saved visualization to: {fig_path}")

    return out  # you still get logits etc. back

def rgb_to_gray3(img):
    # Convert anything (torch, np, batched CHW) to HxWx3 uint8
    img_uint8 = ensure_uint8_rgb(img)
    # Convert to [0,1] float
    f = img_uint8.astype(np.float32) / 255.0
    # Grayscale in [0,1]
    g = color.rgb2gray(f)   # shape [H,W]
    # Stack to 3 channels for consistency
    g3 = np.stack([g, g, g], axis=-1)
    return (g3 * 255).astype(np.uint8)

def visualize_gray_proxy(x_uint8, xcf_uint8, save_dir=None, prefix="viz_gray"):
    gx, gcf = rgb_to_gray3(x_uint8), rgb_to_gray3(xcf_uint8)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(gx);  axes[0].set_title("Original (gray)"); axes[0].axis("off")
    axes[1].imshow(gcf); axes[1].set_title("Counterfactual (gray)"); axes[1].axis("off")
    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{prefix}_panel.png"), dpi=150, bbox_inches="tight")

def explain_decomposition(decomp_out, class_names=None, top_k=None):
    """
    Pretty-print decomposition results per class.
    
    Args:
        decomp_out: dict with keys ["d_total","phi_style","phi_morph",...], each a tensor [num_classes]
        class_names: optional list of class names (len = num_classes)
        top_k: if set, only show the top_k classes ranked by |d_total|
    """
    d_total   = decomp_out["d_total"].detach().cpu().numpy()
    phi_style = decomp_out["phi_style"].detach().cpu().numpy()
    phi_morph = decomp_out["phi_morph"].detach().cpu().numpy()

    n_classes = len(d_total)
    idxs = np.arange(n_classes)

    # Rank by absolute total change
    if top_k is not None:
        sorted_idxs = np.argsort(-np.abs(d_total))[:top_k]
    else:
        sorted_idxs = idxs

    explanations = []
    for i in sorted_idxs:
        cname = class_names[i] if class_names is not None else f"class {i}"
        total, style, morph = d_total[i], phi_style[i], phi_morph[i]

        # Who dominates?
        if abs(style) > abs(morph):
            driver = "style (stain)"
        elif abs(morph) > abs(style):
            driver = "morphology"
        else:
            driver = "both equally"

        # Conflict or agreement?
        if np.sign(style) != np.sign(morph):
            note = " (style and morph in conflict)"
        else:
            note = ""

        explanations.append(
            f"{cname}: Δ={total:+.3f} → driven by {driver}"
            f" (style={style:+.3f}, morph={morph:+.3f}){note}"
        )

    return explanations

def dataset_tensor_to_uint8(img_t: torch.Tensor,
                            was_normalized: bool = True) -> np.ndarray:
    """
    Convert a dataset tensor to HxWx3 uint8 RGB.
    - Accepts (3,H,W) or (1,3,H,W)
    - If was_normalized=True, assumes Normalize(mean=0.5,std=0.5) was applied.
    """
    assert isinstance(img_t, torch.Tensor)
    t = img_t.detach().cpu().float()
    if t.ndim == 4:
        assert t.size(0) == 1, f"expected a single image, got {tuple(t.shape)}"
        t = t[0]                     # (3,H,W)

    # undo Normalize(mean=0.5, std=0.5) -> back to [0,1]
    if was_normalized:
        t = t * 0.5 + 0.5
    t = t.clamp(0, 1)

    # CHW -> HWC and scale to 0..255
    arr = (t * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr

def log_decomposition(out, lines, save_dir, prefix="decomposition"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{prefix}.txt")

    # Format tensors nicely
    def fmt(t): return np.round(t.detach().cpu().numpy(), 4).tolist()

    with open(path, "a", encoding="utf-8") as f:
        f.write("=== Raw tensors ===\n")
        f.write(f"d_total: {fmt(out['d_total'])}\n")
        f.write(f"phi_style: {fmt(out['phi_style'])}\n")
        f.write(f"phi_morph: {fmt(out['phi_morph'])}\n")
        f.write(f"s0: {fmt(out['s0'])}\n")
        f.write(f"s_cf: {fmt(out['s_cf'])}\n")
        f.write(f"s_style: {fmt(out['s_style'])}\n")
        f.write(f"s_morph: {fmt(out['s_morph'])}\n\n")

        f.write("=== Human-readable explanation ===\n")
        for line in lines:
            print(line)           # still print to console
            f.write(line + "\n")
        f.write("\n")
    print(f"Decomposition log saved to {path}")


if __name__ == "__main__":

    autoenc_model_path = hf_hub_download(
        repo_id="KatherLab/MoPaDi",
        filename="crc_512_model/autoenc.ckpt",
    )
    print(f"Autoencoder's checkpoint downloaded to: {autoenc_model_path}")

    clf_model_path = hf_hub_download(
        repo_id="KatherLab/MoPaDi",
        filename="crc_512_model/mil_msi_classifier.pth",
    )
    print(f"Classifier's checkpoint downloaded to: {clf_model_path}")

    save_folder = "hybrids-lvl6-new2"
    manipulator = ImageManipulator(
        autoenc_config = tcga_crc_autoenc(),
        autoenc_path = autoenc_model_path, 
        mil_path = clf_model_path, 
        conf_cls = crc_pretrained_mil(),
        dataset=None,
        device="cuda:0",
    )

    for patient_name in tqdm(os.listdir(f"{ws_path}/mopadi/checkpoints/crc-paper/mil_classifier_isMSIH/counterfactuals")):

        pat_dir = f"{ws_path}/mopadi/checkpoints/crc-paper/mil_classifier_isMSIH/counterfactuals/{patient_name}"
        tile_dirs = [os.path.join(pat_dir, f) for f in os.listdir(pat_dir)]
        print(f"Number of manipulated tiles found: {len(tile_dirs)}")

        for tile in tile_dirs:

            save_dir = os.path.join(tile, save_folder)
            if os.path.exists(save_dir):
                print(f"Skipping {tile}, already done.")
                continue

            tile_dir_contents = os.listdir(tile)
            for f in tile_dir_contents:
                if "0,06" in f:
                    image_name = os.path.join(tile, f)
                elif "original" in f:
                    ori_image_name = os.path.join(tile, f)
                else:
                    continue
            
            if "cam" in ori_image_name or "cam" in ori_image_name:
                print("found:")
                print(ori_image_name)
                print(image_name)

            ori = Image.open(ori_image_name)
            manip = Image.open(image_name)

            if "manip_to_nonMSIH" in image_name:
                target_class = "nonMSIH"
                ori_class = "MSIH"

            #x_uint8   = dataset_tensor_to_uint8(ori, was_normalized=False)
            #xcf_uint8 = tensor_to_uint8_rgb(manip, was_normalized=False)
            x_uint8 = np.array(ori.convert("RGB"), dtype=np.uint8)
            xcf_uint8 = np.array(manip.convert("RGB"), dtype=np.uint8)

            out = decompose_style_vs_morph_vahadane(manipulator, x_uint8, xcf_uint8)

            class_names = ["nonMSIH", "MSIH"]
            lines = explain_decomposition(out, class_names=class_names, top_k=5)

            log_decomposition(out, lines, save_dir, prefix="decomposition")

            out = visualize_decomposition(manipulator, x_uint8, xcf_uint8, save_dir=os.path.join(tile, save_folder), prefix="case001")

            # grayscale sanity check
            # visualize_gray_proxy(x_uint8, xcf_uint8, save_dir=save_folder, prefix="case001")
