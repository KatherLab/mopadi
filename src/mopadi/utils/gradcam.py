import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
from dotenv import load_dotenv

from mopadi.configs.templates import tcga_crc_autoenc
from mopadi.configs.templates_cls import crc_pretrained_mil
from mopadi.configs.templates_cls import *
from mopadi.mil.utils import *

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")

# take the last Conv2d inside the encoder
def find_last_conv(module: nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found in encoder.")
    return last

def list_conv_layers(model):
    layers, names = [], []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            layers.append(m)
            names.append(name)
    return layers, names

def resolve_target_layer(model, target):
    if isinstance(target, nn.Module):
        return target

    if isinstance(target, int):
        convs, names = list_conv_layers(model)
        if target < 0 or target >= len(convs):
            raise IndexError(f"target index {target} out of range (found {len(convs)} convs)")
        return convs[target]

    if isinstance(target, str):
        # exact match first
        for name, m in model.named_modules():
            if name == target:
                return m
        # partial match fallback
        candidates = [m for name, m in model.named_modules() if target in name]
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            raise ValueError(f"Ambiguous target '{target}', matches multiple modules")
        raise ValueError(f"No module named '{target}' found")

    raise TypeError(f"Unsupported target type: {type(target)}")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
])

def to_uint8_rgb(arr01):
    arr01 = np.clip(arr01, 0, 1)
    return (arr01*255.0).round().astype(np.uint8)

class GradCAM:
    def __init__(self, encoder: nn.Module, target_layer: nn.Module):
        self.encoder = encoder
        self.target_layer = target_layer
        self._acts = None
        self._grads = None
        self._fh = target_layer.register_forward_hook(self._save_acts)
        self._bh = target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, module, inp, out):
        # out: [B, C, H, W]
        self._acts = out.detach()

    def _save_grads(self, module, grad_in, grad_out):
        # grad_out[0]: [B, C, H, W]
        self._grads = grad_out[0].detach()

    def remove_hooks(self):
        self._fh.remove()
        self._bh.remove()

    def compute_cam(self):
        """
        Return CAM in shape [B, 1, H, W], ReLU normalized per-sample to 0..1.
        Requires self._acts and self._grads to be filled by a backward pass.
        """
        assert self._acts is not None and self._grads is not None, "Run forward+backward before compute_cam()"
        B, C, H, W = self._acts.shape
        # weights: GAP over spatial for each channel k
        weights = self._grads.mean(dim=(2,3), keepdim=True)     # [B, C, 1, 1]
        cam = (weights * self._acts).sum(dim=1, keepdim=True)   # [B, 1, H, W]
        cam = F.relu(cam)
        # normalize each sample to 0..1
        cam_min = cam.view(B,-1).min(dim=1)[0].view(B,1,1,1)
        cam_max = cam.view(B,-1).max(dim=1)[0].view(B,1,1,1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

def gradcam_single_tile(lit_model, mil_classifier, pil_img: Image.Image,
                        class_idx=None, device="cuda:0", target_layer=None,
                        overlay_alpha=0.35):
    """
    - lit_model: LitModel
    - mil_classifier (expects [B, N, D]); we feed N=1
    - pil_img: raw tile (PIL)
    - class_idx: if None -> use predicted class
    Returns: dict with 'logits', 'prob', 'class_idx', 'cam_uint8', 'overlay_uint8'
    """
    lit_model.eval()
    mil_classifier.eval()

    enc_owner = getattr(lit_model, "ema_model", lit_model)
    encoder = enc_owner.encoder

    if target_layer is None:
        target_layer = find_last_conv(encoder)
    else:
        target_module = resolve_target_layer(encoder, li) 

    cam_engine = GradCAM(encoder, target_module)

    x = preprocess(pil_img).unsqueeze(0).to(device)   # [1,3,H,W]

    # Forward all the way: encoder -> feature -> MIL (as 1-tile bag)
    x.requires_grad_(True)
    feats = encoder(x) 
    if feats.ndim == 2:
        bag = feats.unsqueeze(1)
    else:
        bag = feats.view(1, 1, -1)

    logits = mil_classifier(bag)
    probs = torch.softmax(logits, dim=1)

    if class_idx is None:
        class_idx = int(probs.argmax(dim=1).item())

    loss = logits[0, class_idx]
    lit_model.zero_grad(set_to_none=True)
    mil_classifier.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)

    # compute CAM on the hooked encoder layer
    cam = cam_engine.compute_cam()[0,0].detach().cpu().numpy()   # [Hc, Wc] in 0..1
    cam_engine.remove_hooks()

    # upscale CAM to input size
    H, W = pil_img.size[1], pil_img.size[0]
    cam_up = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)  # 0..1
    cam_u8 = to_uint8_rgb(cam_up)  # HxW uint8 (single channel replicated to RGB below)
    cam_rgb = np.stack([cam_u8, cam_u8, cam_u8], axis=-1)

    # overlay on the *unnormalized* RGB image (0..1)
    base = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    heat = cv2.applyColorMap((cam_up*255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1] / 255.0
    overlay = (1 - overlay_alpha) * base + overlay_alpha * heat
    overlay_u8 = to_uint8_rgb(overlay)

    return {
        "logits": logits.detach().cpu(),
        "prob": probs.detach().cpu().numpy().round(3),
        "class_idx": class_idx,
        "cam_uint8": cam_rgb,          # grayscale in 3 channels
        "overlay_uint8": overlay_u8,   # colored overlay
    }


if __name__ == "__main__":

    device = "cuda:0"

    # downloaded from Huggingface
    autoenc_model_path = "/home/laura/.cache/huggingface/hub/models--KatherLab--MoPaDi/snapshots/5d8e775e24473c5d8f4c0c57fd5c865c3c2a4aab/crc_512_model/autoenc.ckpt"
    clf_model_path = "/home/laura/.cache/huggingface/hub/models--KatherLab--MoPaDi/snapshots/5d8e775e24473c5d8f4c0c57fd5c865c3c2a4aab/crc_512_model/mil_msi_classifier.pth"

    model = LitModel((tcga_crc_autoenc()))
    state = torch.load(autoenc_model_path, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()
    model.to(device).eval()

    conf_cls = crc_pretrained_mil()

    mil_classifier = Classifier(conf_cls.dim, conf_cls.num_heads, conf_cls.num_seeds, conf_cls.num_classes)
    weights = torch.load(clf_model_path)
    mil_classifier.load_state_dict(weights)
    mil_classifier = mil_classifier.to(device)
    mil_classifier.eval()

    tile_path = f"{ws_path}/mopadi/checkpoints/crc-paper/mil_classifier_isMSIH/counterfactuals/TCGA-DC-6160/Tile_(10752,12288)/hybrids-lvl6-new/case001_morphHybrid.png"
    save_path = os.path.dirname(tile_path)

    tile = Image.open(tile_path).convert("RGB")

    for li in [19]:
        out = gradcam_single_tile(model, mil_classifier, tile, device=device, target_layer=li)
        Image.fromarray(out["overlay_uint8"]).save(os.path.join(save_path, f"cam_overlay_L{li}_style_hybrid.png"))
        #Image.fromarray(out["cam_uint8"]).save(os.path.join(save_path, f"cam_gray_L{li}.png"))

    print("Pred class:", out["class_idx"])
    print("Probs:", out["prob"])
