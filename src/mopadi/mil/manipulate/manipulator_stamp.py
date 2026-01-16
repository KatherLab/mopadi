import torch
from torchvision import transforms
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from skimage.metrics import structural_similarity, mean_squared_error
import cv2
from pathlib import Path

from mopadi.configs.templates import *
from mopadi.configs.templates_cls import *
from mopadi.mil.utils import *
from mopadi.model.extractor import (
    FeatureExtractorConch, FeatureExtractorConch15,
    FeatureExtractorVirchow2, FeatureExtractorUNI2
)


class ImageManipulatorSTAMP:
    def __init__(self, autoenc_config, autoenc_path, mil_path, dataset, device="cuda:0"):
        self.device = device
        self.feat_extractor = autoenc_config.feat_extractor
        self.model = self._load_model(autoenc_config, autoenc_path)
        self.classifier = self._load_cls_model(mil_path)
        self.data = dataset
        print("Both models loaded successfully.")

    def _load_model(self, model_config, checkpoint_path):
        model = LitModel(model_config)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=False)
        model.eval()
        model.to(self.device)
        model.feat_extractor
        if self.feat_extractor == 'conch':
            model.feat_extractor = FeatureExtractorConch(device=self.device)
        elif self.feat_extractor == 'conch1_5':
            model.feat_extractor = FeatureExtractorConch15(device=self.device)
        elif self.feat_extractor == 'v2':
            model.feat_extractor = FeatureExtractorVirchow2(device=self.device)
        elif self.feat_extractor == 'uni2':
            model.feat_extractor = FeatureExtractorUNI2(device=self.device)
        return model
    
    def _load_cls_model(self, mil_path):
        from stamp.modeling.models.vision_tranformer import VisionTransformer

        ckpt = torch.load(mil_path,  weights_only=False, map_location="cpu")

        hp = ckpt["hyper_parameters"]
        sd = ckpt["state_dict"]

        cls_model = VisionTransformer(
            dim_output=len(hp["categories"]),
            dim_input=hp["dim_input"],
            dim_model=hp["dim_model"],
            n_layers=hp["n_layers"],
            n_heads=hp["n_heads"],
            dim_feedforward=hp["dim_feedforward"],
            dropout=hp["dropout"],
            use_alibi=hp["use_alibi"],
        )
        print(f"Categories of the trained STAMP model: {hp['categories']}")

        model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        missing, unexpected = cls_model.load_state_dict(model_sd, strict=True)
        print("STAMP vision transformer loaded. missing:", missing, "unexpected:", unexpected)

        cls_model = cls_model.to(self.device)
        cls_model.eval()
        return cls_model


    def manipulate_latent_feats_stamp_slide_level(
        self,
        vit_model,      # VisionTransformer (nn.Module)
        feats,          # [tile, feature] or [1, tile, feature]
        coords,         # [tile, 2] or [1, tile, 2]
        man_amp,
        cls_id,
        mask=None,      # [tile] or [1, tile] bool, or None
    ):
        vit_model.eval()
        vit_model.zero_grad(set_to_none=True)

        device = next(vit_model.parameters()).device

        # Ensure shapes: [1, tile, feature] and [1, tile, 2]
        if feats.dim() == 2:
            feats_bag = feats.unsqueeze(0)
        else:
            feats_bag = feats

        if coords.dim() == 2:
            coords_bag = coords.unsqueeze(0)
        else:
            coords_bag = coords

        if mask is not None and mask.dim() == 1:
            mask_bag = mask.unsqueeze(0)
        else:
            mask_bag = mask

        # Dtype policy:
        # - grads for manipulation direction: float32 is safer
        feats_bag = feats_bag.to(device=device, dtype=torch.float32).detach().clone().requires_grad_(True)
        coords_bag = coords_bag.to(device=device, dtype=torch.float32)

        # Forward: logits [1, num_classes]
        logits = vit_model(feats_bag, coords=coords_bag, mask=mask_bag)

        # Objective: increase target class logit (or prob; logit is fine)
        loss = logits[:, cls_id].sum()
        loss.backward()

        grad = feats_bag.grad
        if grad is None:
            raise RuntimeError("Gradients are None. Backward did not populate feats grad.")
        if torch.all(grad == 0):
            raise RuntimeError("Gradients are zero. Check cls_id, forward path, and requires_grad.")

        # Normalize per-tile direction in feature space
        direction = F.normalize(grad, dim=-1)

        # Keep your original scaling convention
        norm_man_amp = man_amp * math.sqrt(feats_bag.size(-1))
        manipulated = feats_bag + norm_man_amp * direction

        # Return in the original dtype/shape (commonly float16, no batch)
        out = manipulated.detach()
        if feats.dim() == 2:
            out = out.squeeze(0)

        return out.to(dtype=feats.dtype)
    
    def manipulate_latent_feats_stamp_tile_level(
        self,
        vit_model,
        feats,        # [N, F] or [1,N,F]
        coords,       # [N, 2] or [1,N,2]
        tile_indices,
        man_amp,
        cls_id,
    ):
        vit_model.eval()
        vit_model.zero_grad(set_to_none=True)

        device = next(vit_model.parameters()).device

        # Ensure shapes [1, N, *]
        feats_bag = feats.unsqueeze(0) if feats.dim() == 2 else feats
        coords_bag = coords.unsqueeze(0) if coords.dim() == 2 else coords

        # Ensure dtype/device: grads in fp32; coords fp32
        feats_bag = feats_bag.to(device=device, dtype=torch.float32).detach().clone().requires_grad_(True)
        coords_bag = coords_bag.to(device=device, dtype=torch.float32)

        logits = vit_model(feats_bag, coords=coords_bag, mask=None)  # [1, C]
        loss = logits[:, cls_id].sum()
        loss.backward()

        grad = feats_bag.grad[0]  # [N, F]

        # Output in original dtype
        feats_out = feats.detach().clone().to(device)

        if not torch.is_tensor(tile_indices):
            tile_indices = torch.tensor(tile_indices, dtype=torch.long, device=device)
        else:
            tile_indices = tile_indices.to(device)

        for idx in tile_indices:
            g = grad[idx]
            direction = F.normalize(g, dim=-1)
            norm_amp = man_amp * math.sqrt(g.numel())
            feats_out[idx] = feats_out[idx] + norm_amp * direction.to(feats_out.dtype)

        return feats_out


    def manipulate_patients_images(
                        self, 
                        patient_name=None, 
                        patient_features=None,
                        save_path=os.path.join(os.getcwd(), "results"), 
                        man_amps=[0.1, 0.2, 0.3], 
                        T_step=100, 
                        T_inv=200,
                        patient_class=None,
                        target_dict=None,
                        num_top_tiles=15,
                        ):

        target_class = next(cls for cls in target_dict if cls != patient_class)
        target_cls_id = target_dict[target_class]

        if patient_features is None:
            try:
                patient_features, coords, fnames = self.data.get_patient_features(patient_name)
                coords = torch.as_tensor(coords, device=self.device, dtype=torch.float32)
                patient_features = patient_features.float().squeeze(0).to(self.device)

            except Exception as e:
                print(f"Exception occurred while getting patient features: {e}")
                return
    
        top_tiles = stamp_top_tiles(model=self.classifier, feats=patient_features, coords=coords, topk=num_top_tiles, device=self.device)
        top_tiles = top_tiles['topk_idx']

        for man_amp in tqdm(man_amps, desc="Manipulating at different amplitudes"):

            for top_idx in tqdm(top_tiles, total=len(top_tiles.cpu().tolist()), desc='Manipulating top tiles'):
                idx = int(top_idx.item())
                top_tile_feats_manip = self.manipulate_latent_feats_stamp_tile_level(vit_model=self.classifier,
                                                                                feats=patient_features,
                                                                                coords=coords,
                                                                                tile_indices=[idx],
                                                                                man_amp=man_amp, 
                                                                                cls_id=target_cls_id)

                single_tile_feats_manip = top_tile_feats_manip[idx].unsqueeze(0).unsqueeze(0)
                single_tile_coords = coords[idx].unsqueeze(0).unsqueeze(0)

                # original (pre-manipulation)
                orig_tile_feats = patient_features[idx].unsqueeze(0).unsqueeze(0)
                logits_orig = self.classifier(orig_tile_feats, coords=single_tile_coords, mask=None)
                pred_original = torch.softmax(logits_orig, dim=1)

                logits_manipulated = self.classifier(single_tile_feats_manip, coords=single_tile_coords, mask=None)
                pred_manipulated = torch.softmax(logits_manipulated, dim=1)

                try:
                    tile_path = fnames[idx]
                    fname = os.path.basename(fnames[idx]).split(".png")[0]
                except IndexError:
                    print(f"Features for the top tile {top_idx.item()} for patient {patient_name} not found. Patient has {patient_features.size()} features.")
                    continue

                out_dir = os.path.join(save_path, os.path.basename(fname).split(".png")[0])
                Path(out_dir).mkdir(parents=True, exist_ok=True)

                inverted_target_dict = {v: k for k, v in target_dict.items()}

                with open(os.path.join(out_dir, "predictions.txt"), "a") as f:
                    f.write(f"Original tile-level prediction ({inverted_target_dict[0]}, {inverted_target_dict[1]}): {[f'{p:.3f}' for p in pred_original.squeeze().detach().cpu().numpy()]}\n")
                    f.write(f"\n")

                original_out_path = os.path.join(out_dir, f"{fname}_0_original_{patient_class}.png")
                save_manip_path = os.path.join(out_dir, f"{fname}_manip_to_{target_class}_amp_{'{:.3f}'.format(man_amp).replace('.', ',')}.png")

                original_img = self.data.get_tile(tile_path).unsqueeze(0).to(self.device)

                features = patient_features[idx].unsqueeze(0)  # [1, F]

                stochastic_latent = self.model.encode_stochastic(original_img.to(self.device), features.to(self.device), T=T_inv)
                manipulated_img = self.model.render(stochastic_latent, single_tile_feats_manip.squeeze(0), T=T_step)

                original_img_rgb = convert2rgb(original_img)
                self.save_image(original_img_rgb, original_out_path)

                manipulated_img_rgb = convert2rgb(manipulated_img[0])
                self.save_image(manipulated_img_rgb, save_manip_path)

                # make predictions for the manipulated rendered image
                img_feats = self.model.feat_extractor.extract_feats(manipulated_img.to(self.device), need_grad=False)
                logits = self.classifier(
                    img_feats.clone().unsqueeze(0),          # [1,1,F]
                    coords=single_tile_coords,        # [1,1,2]
                    mask=None,
                )
                pred_rendered = torch.softmax(logits, dim=1)

                with open(os.path.join(out_dir, "predictions.txt"), "a") as f:
                    f.write(f"Manipulation amplitude: {man_amp}\n")  
                    f.write(f"Pred manip feat ({inverted_target_dict[0]}, {inverted_target_dict[1]}): {[f'{p:.3f}' for p in pred_manipulated.detach().squeeze().cpu().numpy()]}\n")
                    f.write(f"Pred rendered img ({inverted_target_dict[0]}, {inverted_target_dict[1]}): {[f'{p:.3f}' for p in pred_rendered.detach().squeeze().cpu().numpy()]}\n")
                    f.write(f"--------------------------------------------------\n")


    def save_image(self, image_tensor, save_path):
        image = transforms.ToPILImage()(image_tensor.cpu().squeeze(0))
        image.save(save_path)

def convert2rgb(img):
    convert_img = img.clone().detach()
    if convert_img.min() < 0:
        # transform pixel values into the [0, 1] range
        convert_img = (convert_img + 1) / 2
    return convert_img.cpu()


def compute_structural_similarity(reconstructed_img, image_original, out_file_dir=None):
    """
    Compute the Structural Similarity Index (SSIM) [1], Mean Square Error and Multi-scale SSIM [2]
    between the reconstructed image and the original image.

    Args:
        reconstructed_img (PIL Image): Reconstructed image
        image_original (PIL Image): Original image
        out_file_dir (str): Directory to save the computed metrics
    
    References:
    [1] https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
    [2] https://lightning.ai/docs/torchmetrics/stable/image/multi_scale_structural_similarity.html
    """

    reconstructed_img_np = np.array(reconstructed_img)
    image_original_np = np.array(image_original)
    #print(f"Shape of the original image: {image_original_np.shape}, and range: [{image_original_np.min()}, {image_original_np.max()}]")
    #print(f"Shape of the reconstructed image: {reconstructed_img_np.shape}, and range: [{reconstructed_img_np.min()}, {reconstructed_img_np.max()}]")

    # SSIM
    (ssim, diff) = structural_similarity(image_original_np, reconstructed_img_np, full=True, data_range=image_original_np.max() - image_original_np.min(), multichannel = True, channel_axis = 2)
    print("Image Similarity: {:.3f}%".format(ssim * 100))
    diff = (diff * 255).astype("uint8")
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # MEAN SQUARE ERROR (MSE)
    # convert to grayscale
    image_original_gray = cv2.cvtColor(image_original_np, cv2.COLOR_BGR2GRAY)
    reconstructed_img_gray = cv2.cvtColor(reconstructed_img_np, cv2.COLOR_BGR2GRAY)

    image_original_normalized = image_original_gray.astype("float32") / 255.0
    reconstructed_img_normalized = reconstructed_img_gray.astype("float32") / 255.0

    mse = mean_squared_error(image_original_normalized, reconstructed_img_normalized)
    print(f"MSE: {mse:.4f}")

    # MULTI-SCALE SSIM
    image_original_tensor = torch.tensor(image_original_normalized).unsqueeze(0).unsqueeze(0).float()
    reconstructed_img_tensor = torch.tensor(reconstructed_img_normalized).unsqueeze(0).unsqueeze(0).float()

    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()
    ms_ssim_res = ms_ssim(reconstructed_img_tensor,  image_original_tensor) * 100
    print(f"MultiScale Structural Similarity: {ms_ssim_res:.3f}%")
    print("-----------------------------------------------")

    if out_file_dir is not None:
        with open(os.path.join(out_file_dir, "ssim_info.txt"), 'a') as f:
            f.write("\nImage Similarity: {:.3f}%".format(ssim * 100))
            f.write(f"\nMSE: {mse:.3f}")
            f.write(f"\nMultiScale Structural Similarity: {ms_ssim_res:.3f}")
            f.write("\n------------------------------------------------\n")

    return diff_gray, ssim, mse, ms_ssim_res
