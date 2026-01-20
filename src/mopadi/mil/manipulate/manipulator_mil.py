import torch
from torchvision import transforms
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from skimage.metrics import structural_similarity, mean_squared_error
import cv2

from mopadi.configs.templates import *
from mopadi.configs.templates_cls import *
from mopadi.mil.utils import *


class ImageManipulator:
    def __init__(self, autoenc_config, autoenc_path, mil_path, conf_cls, dataset=None, device="cuda:0"):
        self.device = device
        self.model = self._load_model(autoenc_config, autoenc_path)
        self.classifier = self._load_cls_model(conf_cls, mil_path)
        self.data = dataset
        print("Both models loaded successfully.")
        self.normalizer = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def _load_model(self, model_config, checkpoint_path):
        model = LitModel(model_config)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=False)
        model.eval()
        model.to(self.device)
        return model
    
    def _load_cls_model(self, 
                       conf_cls,
                       mil_path,
                    ):
        cls_model = cls_model.to(self.device)
        cls_model.eval()
        return cls_model


    def manipulate_latent_feats(self, model, feats, man_amp, cls_id):
        feats = feats.detach().clone().requires_grad_(True)
        model.zero_grad()

        scores = model(feats)

        model.train()

        loss = scores[:, cls_id].sum([])
        # backward pass to get gradients
        loss.backward()

        # set the model back to eval mode
        model.eval()

        if feats.grad is None:
            raise RuntimeError("Gradients are None. Ensure backward pass is performed correctly.")
        if torch.all(feats.grad == 0):
            raise RuntimeError("Gradients are zero. Ensure the backward pass is computing gradients correctly.")

        normalized_class_direction = F.normalize(feats.grad, dim=-1)
        
        norm_man_amp = man_amp * math.sqrt(feats.size(-1))
        manipulated_feats = feats + norm_man_amp * normalized_class_direction

        return manipulated_feats

    def manipulate_one_image(self,
                            img,                       # PIL.Image or Tensor (3,H,W) in [0,1] or [0,255]
                            source_class: str,         # e.g. "MSIH" or "nonMSIH"
                            target_dict: dict,         # {"nonMSIH":0, "MSIH":1}  (or your 9 classes)
                            man_amps=(0.1, 0.2, 0.3),  # amplitudes to try
                            T_step=100,
                            T_inv=200,
                            out_dir="./one_image_results",
                            basename="case"):
        """
        Encodes feats for a single tile, manipulates toward the *other* class,
        renders images, and writes predictions to <out_dir>/predictions.txt.
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # Decide target class ID
        if len(target_dict) == 2:
            target_class = next(k for k in target_dict if k != source_class)
        else:
            # for multi-class pass target explicitly; here we take the highest non-source logit later
            target_class = None

        source_id = target_dict[source_class]
        target_id = None if target_class is None else target_dict[target_class]

        x = _to_device_uint(img, self.device)
        enc_owner = getattr(self.model, "ema_model", self.model)
        encoder = enc_owner.encoder
        with torch.no_grad():
            cond = encoder(x)
        bag = cond.unsqueeze(1)

        with torch.no_grad():
            logits_ori = self.classifier(bag)
            probs_ori = _to_probs(logits_ori)
            c = probs_ori.argmax(dim=1).item()

        # For multi-class with unspecified target: choose best non-source
        if target_id is None and logits_ori.size(-1) > 2:
            mask = torch.ones_like(logits_ori[0], dtype=torch.bool)
            mask[source_id] = False
            target_id = int(torch.argmax(logits_ori[0][mask]).item())
            # map masked index back to true index
            target_id = [i for i in range(logits_ori.size(-1)) if i != source_id][target_id]
            target_class = [k for k,v in target_dict.items() if v == target_id][0]

        with torch.no_grad():
            xT = self.model.encode_stochastic(x, cond, T=T_inv)

        def _save_rgb(t, path):
            im = t.clone()
            if im.min() < 0:
                im = (im + 1)/2
            im = (im.clamp(0,1) * 255).byte()[0].permute(1,2,0).cpu().numpy()
            Image.fromarray(im).save(path)

        ori_path = os.path.join(out_dir, f"{basename}_0_original_{source_class}.png")
        with torch.no_grad():
            # render identity (optional): render(xT, cond) ~= original
            x_id = self.model.render(xT, cond, T=T_step)
        _save_rgb(x_id, ori_path)

        inv_target = {v:k for k,v in target_dict.items()}
        with open(os.path.join(out_dir, "predictions.txt"), "w") as f:
            f.write(f"Classes: {list(target_dict.keys())}\n")
            f.write(f"Original probs: {[f'{p:.3f}' for p in probs_ori[0].cpu().tolist()]}\n")
            f.write(f"Argmax class: {inv_target[c]}\n")
            f.write("\n")

        results = []
        for a in man_amps:
            # ---- Manipulate latent feats toward target ----
            manip_feats = self.manipulate_latent_feats(
                model=self.classifier,
                feats=bag.clone().detach(),              # [1,1,D]
                man_amp=a,
                cls_id=target_id
            )                                            # [1,1,D]

            # ---- Predict on *manipulated features only* (no rendering)
            with torch.no_grad():
                logits_feat = self.classifier(manip_feats)            # [1,C]
                probs_feat = _to_probs(logits_feat)                   # [1,C]

            # ---- Render manipulated image and predict on it
            with torch.no_grad():
                x_manip = self.model.render(xT, manip_feats.squeeze(1), T=T_step)  # [1,3,H,W] in [-1,1]
                # encode again to get fair image-level prediction
                cond_img = encoder(x_manip)
                bag_img = cond_img.unsqueeze(1)
                logits_img = self.classifier(bag_img)
                probs_img = _to_probs(logits_img)

            # ---- Save image + log
            out_img_path = os.path.join(out_dir, f"{basename}_manip_to_{target_class}_amp_{str(a).replace('.', ',')}.png")
            _save_rgb(x_manip, out_img_path)

            with open(os.path.join(out_dir, "predictions.txt"), "a") as f:
                f.write(f"amp={a}  (target={target_class})\n")
                f.write(f"  probs(manip_feats): {[f'{p:.3f}' for p in probs_feat[0].cpu().tolist()]}\n")
                f.write(f"  probs(rendered_img): {[f'{p:.3f}' for p in probs_img[0].cpu().tolist()]}\n")
                f.write("-"*48 + "\n")

            results.append({
                "amp": a,
                "probs_feat": probs_feat.cpu(),
                "probs_img": probs_img.cpu(),
                "image_path": out_img_path,
            })

        return {
            "original_probs": probs_ori.cpu(),
            "target_class": target_class,
            "target_id": target_id,
            "results": results,
            "original_image_path": ori_path,
        }


    def manipulate_patients_images(
                        self, 
                        patient_name=None, 
                        patient_features=None,
                        metadata=None,
                        save_path=os.path.join(os.getcwd(), "results"), 
                        man_amps=[0.1, 0.2, 0.3], 
                        T_step=100, 
                        T_inv=200,
                        patient_class=None,
                        target_dict=None,
                        target_class=None,
                        num_top_tiles=15,
                        filename=None,
                        manip_tiles_separately=True,
                        ):
        
        if target_class is None:
            target_class = next(cls for cls in target_dict if cls != patient_class)
        target_cls_id = target_dict[target_class]

        try:
            top_tiles = get_top_tiles(model=self.classifier, feats=patient_features, k=num_top_tiles, cls_id=target_dict[patient_class], device=self.device)
            print(f"Top {num_top_tiles} tiles obtained for manipulation.")
        except Exception as e:
            print(f"Exception occured while getting top tiles: {e}")
            return

        #print(metadata)
            
        # if to select a specific top tile from a specific patient to manipulate
        if filename is not None:
            selected_index = None
            for i, top_idx in tqdm(enumerate(top_tiles), total=len(top_tiles.cpu().tolist()), desc='Selecting top tiles'):
                try:
                    fname = metadata[top_idx.item()]
                    print(fname)
                except IndexError:
                    print(f"Features for the top tile {top_idx.item()} for patient {patient_name} not found. Patient has {len(metadata)} features.")
                    continue
                
                if fname == filename:
                    selected_index = i
                    break
            
            if selected_index is None:
                print("Selected filename tile was not found!")
                return

            top_tiles = top_tiles[selected_index].unsqueeze(0)  

        if not manip_tiles_separately:
            top_tiles_feats = patient_features.squeeze(0).index_select(0, top_tiles.cpu())

        for man_amp in tqdm(man_amps, desc="Manipulating at different amplitudes"):

            if not manip_tiles_separately:
                manipulated_semantic_latent = self.manipulate_latent_feats(model=self.classifier, 
                                                                           feats=top_tiles_feats.unsqueeze(0).to(self.device),
                                                                           man_amp=man_amp, 
                                                                           cls_id=target_cls_id
                                                                           )
            results = []

            for i, top_idx in tqdm(enumerate(top_tiles), total=len(top_tiles.cpu().tolist()), desc='Manipulating top tiles'):
                res = {}
                if manip_tiles_separately:
                    top_tiles_feats = patient_features.squeeze(0).index_select(0, top_idx.unsqueeze(0).cpu())
                    manipulated_semantic_latent = self.manipulate_latent_feats(model=self.classifier, 
                                                                            feats=top_tiles_feats.unsqueeze(0).to(self.device), 
                                                                            man_amp=man_amp, 
                                                                            cls_id=target_cls_id
                                                                            )

                    logits_original = self.classifier(top_tiles_feats.unsqueeze(0).to(self.device))
                    pred_original = F.softmax(logits_original, dim=-1) # fo multiclass
                    #pred_original = torch.sigmoid(logits_original) # for binary

                    print(f"Patient being processed: {patient_name}; idx: {top_idx.item()}; metadata: {metadata[top_idx.item()]}")
                    try:
                        fname = metadata[top_idx.item()]
                    except IndexError:
                        print(f"Features for the top tile {top_idx.item()} for patient {patient_name} not found. Patient has {len(metadata)} features.")
                        continue
                    #print(f"Filename: {fname}")

                    out_dir = os.path.join(save_path, fname.split(".")[0])
                    if not os.path.exists(out_dir):
                        Path(out_dir).mkdir(parents=True)

                    inverted_target_dict = {v: k for k, v in target_dict.items()}

                    with open(os.path.join(out_dir, "predictions.txt"), "a") as f:
                        f.write(f"Original image prediction ({inverted_target_dict[0]}, {inverted_target_dict[1]}): {[f'{p:.3f}' for p in pred_original.squeeze().detach().cpu().numpy()]}\n")
                        f.write(f"\n")

                # saving paths -------------------------------------------
                try:
                    fname = metadata[top_idx.item()].split(".")[0]
                except IndexError:
                    print(f"Features for the top tile {top_idx.item()} for patient {patient_name} not found. Patient has {len(metadata)} features.")
                    continue

                out_dir = os.path.join(save_path, fname.split(".")[0])
                
                if not os.path.exists(out_dir):
                    Path(out_dir).mkdir(parents=True)

                original_out_path = os.path.join(out_dir, f"{fname}_0_original_{patient_class}.png")
                save_manip_path = os.path.join(out_dir, f"{fname}_manip_to_{target_class}_amp_{'{:.3f}'.format(man_amp).replace('.', ',')}.png")
                
                # if the image has been already manipulated with the same amplitude, skip it
                if os.path.exists(save_manip_path):
                    print(f"Image {save_manip_path} already exists, skipping...")
                    continue

                patient_details = self.data.get_images_by_patient_and_fname(patient_name, fname)
                if patient_details is None:
                    print(f"Patient {patient_name} with filename {fname} not found in the dataset.")
                    continue
                #print(patient_details["filename"])
                original_img = patient_details["image"][None]

                ori_feats = self.model.encode(original_img.to(self.device))
                #print(ori_feats)
                #print(top_tiles_feats)
                #assert torch.allclose(ori_feats.to(self.device), top_tiles_feats.to(self.device), atol=1e-05)

                features = patient_features.index_select(1, top_idx.cpu()).squeeze(0)

                if not manip_tiles_separately:
                    manip_features = manipulated_semantic_latent.squeeze(0)[i].unsqueeze(0)
                else:
                    manip_features = manipulated_semantic_latent.squeeze(0)

                stochastic_latent = self.model.encode_stochastic(original_img.to(self.device), features.to(self.device), T=T_inv)
                res["ori_feats"] = ori_feats
                res["stochastic_latent"] = stochastic_latent
                res["manip_feat"] = manip_features
                res["ori_img"] = original_img
                res["ori_img_path"] = original_out_path

                manipulated_img = self.model.render(stochastic_latent, manip_features, T=T_step)[0]

                original_img_rgb = convert2rgb(original_img)
                self.save_image(original_img_rgb, original_out_path)

                manipulated_img_rgb = convert2rgb(manipulated_img)
                self.save_image(manipulated_img_rgb, save_manip_path)

                res["manip_img"] = manipulated_img
                res["manip_img_rgb"] = manipulated_img_rgb
                res["manip_img_path"] = save_manip_path

                if manip_tiles_separately:
                    # make predictions for the manipulated rendered image
                    # print(manipulated_img.size())
                    img_feats = self.model.encode(manipulated_img.unsqueeze(0).to(self.device))
                    logits = self.classifier(img_feats.unsqueeze(0))
                    pred_rendered = torch.sigmoid(logits) 

                    # make also prediction for only manipulated semantinc latent features
                    logits = self.classifier(manipulated_semantic_latent.to(self.device))
                    #pred = torch.sigmoid(logits) 
                    pred = F.softmax(logits, dim=1)  
                    with open(os.path.join(out_dir, "predictions.txt"), "a") as f:
                        f.write(f"Manipulation amplitude: {man_amp}\n")  
                        f.write(f"Pred manip feat ({inverted_target_dict[0]}, {inverted_target_dict[1]}): {[f'{p:.3f}' for p in pred.detach().squeeze().cpu().numpy()]}\n")
                        f.write(f"Pred rendered img ({inverted_target_dict[0]}, {inverted_target_dict[1]}): {[f'{p:.3f}' for p in pred_rendered.detach().squeeze().cpu().numpy()]}\n")
                        f.write(f"--------------------------------------------------\n")
                results.append(res)
            
        return results  


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

def _to_device_uint(img, device):
    # Accept PIL Image or CHW tensor in [0,1]; normalize like training
    if not torch.is_tensor(img):
        x = transforms.ToTensor()(img)                      # [3,H,W] in [0,1]
    else:
        x = img.float()
        if x.ndim == 3 and x.max() > 1.0:                  # already 0..255?
            x = (x / 255.0).clamp(0, 1)
        if x.ndim == 4:
            x = x[0]
    x = transforms.Normalize((0.5,)*3, (0.5,)*3)(x)
    return x.unsqueeze(0).to(device)                        # [1,3,H,W]

def _to_probs(logits):
    return torch.sigmoid(logits) if logits.size(-1) == 1 else F.softmax(logits, dim=-1)
