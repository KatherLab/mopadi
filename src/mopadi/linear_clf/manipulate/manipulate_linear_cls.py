from skimage.metrics import structural_similarity, mean_squared_error
import cv2

import torch
from torchvision import transforms
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from mopadi.linear_clf.train_linear_cls import ClsModel
from mopadi.configs.templates import *
from mopadi.configs.templates_cls import *


class ImageManipulator:
    def __init__(self, autoenc_config, autoenc_path, cls_config, cls_path, latent_infer_path=None, device="cuda:0"):
        self.device = device
        self.image_dataset = None
        self.cls_config = cls_config
        self.model = self.load_model(autoenc_config, autoenc_path)
        self.classifier = self.load_cls_model(cls_config, cls_path, latent_infer_path)
        self.normalizer = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        print("Both models loaded successfully.")

        print(f"Valid target classes: {cls_config.id_to_cls}")
        self.cls_to_id = {v: k for k, v in enumerate(cls_config.id_to_cls)}

    def load_model(self, model_config, checkpoint_path):
        model = LitModel(model_config)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=False)
        model.eval()
        model.to(self.device)
        return model
    
    def load_cls_model(self, cls_config, cls_path, latent_infer_path=None):
        cls_model = ClsModel(cls_config)
        state = torch.load(cls_path, weights_only=False)
        print('latent step:', state['global_step'])
        cls_model.load_state_dict(state['state_dict'], strict=False)
        cls_model.to(self.device)
        return cls_model

    def manipulate_dataset(self, 
                        image_index, 
                        target_class, 
                        save_path=os.path.join(os.getcwd(), "results"), 
                        manipulation_amplitude=0.3, 
                        T_step=100, 
                        T_inv=200,
                        dataset=None):
        assert self.image_dataset is not None or dataset is not None, "Data is missing."
        self.image_dataset = dataset

        if target_class not in self.cls_config.id_to_cls:
            raise ValueError(f"Invalid target class. Valid options are: {self.cls_config.id_to_cls}")

        batch = self.image_dataset[image_index]["img"][None]  # batch shape needs to be torch.Size([1, 3, 224, 224])

        save_fname = self.image_dataset[image_index]["filename"].split(".")[0] + "_original.png"
        save_fname_manip = self.image_dataset[image_index]["filename"].split(".")[0] + f"_manipulated_to_{target_class}_amplitude_{manipulation_amplitude}.png"
        
        return self.manipulate(batch, target_class, manipulation_amplitude, save_path, save_fname, save_fname_manip, T_step, T_inv)

    def manipulate_image(self, 
                        img, 
                        target_class,
                        manipulation_amplitude, 
                        save_path=os.path.join(os.getcwd(), "results"), 
                        save_fname_ori = "original.png",
                        save_fname_manip = None,
                        T_step=100, 
                        T_inv=200):

        transform = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform = transforms.Compose(transform)
        img = transform(img).unsqueeze(0)

        return self.manipulate(img, target_class, manipulation_amplitude, save_path, save_fname_ori, save_fname_manip, T_step, T_inv)

    def manipulate(self,
                   img,
                   target_class, 
                   manipulation_amplitude,
                   save_path, 
                   save_fname_ori,
                   save_fname_manip,
                   T_step=100, 
                   T_inv=200):

        if save_fname_manip is None:
            save_fname_manip = f"manipulated_to_{target_class}_amplitude_{manipulation_amplitude}.png"

        if target_class not in self.cls_config.id_to_cls:
            raise ValueError(f"Invalid target class. Valid options are: {self.cls_config.id_to_cls}")

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # Image encoding and manipulation steps
        results = {}
        semantic_latent = self.model.encode(img.to(self.device))
        stochastic_latent = self.model.encode_stochastic(
            img.to(self.device), semantic_latent, T=T_inv
        )
        results["ori_feats"] = semantic_latent
        results["stochastic_latent"] = stochastic_latent

        cls_id = self.cls_to_id[target_class]
        target_list = self.cls_config.id_to_cls

        if not self.cls_config.linear:
            semantic_latent = semantic_latent.detach()
            semantic_latent.requires_grad = True
            self.classifier.classifier.zero_grad()
            scores = self.classifier.classifier(semantic_latent)
            scores[:,cls_id].sum().backward()
            normalized_class_direction = F.normalize(semantic_latent.grad, dim=1)
            
        else:
            class_direction = self.classifier.classifier.weight[cls_id] 
            normalized_class_direction = F.normalize(class_direction[None, :], dim=1)
        
        normalized_semantic_latent = self.classifier.normalize(semantic_latent)
        normalized_manipulation_amp = manipulation_amplitude * math.sqrt(512)
        normalized_manipulated_semantic_latent = (
            normalized_semantic_latent
            + normalized_manipulation_amp * normalized_class_direction
        )

        manipulated_semantic_latent = self.classifier.denormalize(
            normalized_manipulated_semantic_latent
        )

        results["manip_feat"] = manipulated_semantic_latent

        # Render Manipulated image
        manipulated_img = self.model.render(stochastic_latent, manipulated_semantic_latent, T=T_step)[0]

        # Save the original image
        out_path = os.path.join(save_path, save_fname_ori)

        original_img_rgb = convert2rgb(img)
        results["ori_img"] = original_img_rgb
        results["ori_img_path"] = out_path
        save_image(original_img_rgb, str(out_path))
        #print(f"Original image saved to: {out_path}")

        # Save manipulated image
        save_manip_path = os.path.join(save_path, save_fname_manip)

        manipulated_img_rgb = convert2rgb(manipulated_img)
        results["manip_img"] = manipulated_img
        results["manip_img_rgb"] = manipulated_img_rgb
        results["manip_img_path"] = save_manip_path
        self.save_image(manipulated_img_rgb, save_manip_path)
        #print(f"Manipulated image saved to: {save_manip_path}")


        with torch.no_grad():
            self.classifier.ema_classifier.eval()

            # normalize rendered img 
            manip_img_norm = self.normalizer(manipulated_img.unsqueeze(0).cuda().float())

            # extract features from manipulated img
            manip_img_latent = self.model.ema_model.encoder(manip_img_norm)

            # normalize extracted features
            manip_img_latent_norm = self.classifier.normalize(manip_img_latent)

            # pass through the classifier
            output = self.classifier.ema_classifier.forward(manip_img_latent_norm)                
            pred = torch.softmax(output, dim=1)
            _, max_index = torch.max(pred, 1)
            predicted_class = target_list[max_index.item()]

            results["target_list"] = target_list
            results["preds"] = f"{[f'{p:.3f}' for p in pred.cpu().numpy().flatten()]}"
            results["pred_class"] = predicted_class

        return results

    def save_image(self, image_tensor, save_path):
        image = transforms.ToPILImage()(image_tensor.cpu().squeeze(0))
        image.save(save_path)

    def predict_path(self, img_path: str) -> Tuple[str, np.ndarray]:
        img = Image.open(img_path).convert("RGB")
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        x = tfm(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats  = self.model.ema_model.encoder(x)
            feats  = self.classifier.normalize(feats)
            logits = self.classifier.ema_classifier(feats)
            probs  = torch.sigmoid(logits).cpu().numpy().flatten()

        pred_lbl = self.cls_config.id_to_cls[int(probs.argmax())]
        return pred_lbl, probs


def compute_structural_similarity(reconstructed_img, image_original_tensor, out_file_dir=None):
    reconstructed_img_np = reconstructed_img.permute(2, 1, 0).cpu().numpy()
    image_original = image_original_tensor.permute(2, 1, 0).cpu().detach().numpy()

    #print(f"Shape of the original image: {image_original.shape}, and range: [{image_original.min()}, {image_original.max()}]")
    #print(f"Shape of the reconstructed image: {reconstructed_img_np.shape}, and range: [{reconstructed_img_np.min()}, {reconstructed_img_np.max()}]")

    if image_original.min() < 0:
        # transform image data that is initially in the range [-1, 1] to the range [0, 1]
        image_original = (image_original + 1) / 2
        #print(f"Adjusted range of the original image: {image_original.min()}, {image_original.max()}")

    # convert to grayscale
    before_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(reconstructed_img_np, cv2.COLOR_BGR2GRAY)

    # SSIM
    (ssim, diff) = structural_similarity(before_gray, after_gray, full=True, data_range=before_gray.max() - before_gray.min())
    #print("Image Similarity: {:.4f}%".format(ssim * 100))
    diff = (diff * 255).astype("uint8")
    
    # MEAN SQUARE ERROR (MSE)
    mse = mean_squared_error(before_gray, after_gray)
    #print(f"MSE: {mse:.4f}")

    # MULTI-SCALE SSIM
    #print(f"Shapes and ranges before computing SSIM: reconstructed image {reconstructed_img.unsqueeze(0).size()}, [{reconstructed_img.unsqueeze(0).min()}, {reconstructed_img.unsqueeze(0).max()}], and original: {image_original_tensor.unsqueeze(0).size()}, [{image_original_tensor.unsqueeze(0).min()}, {image_original_tensor.unsqueeze(0).max()}]")

    image_original_tensor = image_original_tensor.unsqueeze(0)
    if image_original_tensor.min() < 0:
        # transform image data that is initially in the range [-1, 1] to the range [0, 1]
        image_original_tensor = (image_original_tensor.cpu() + 1) / 2
        #print(f"Adjusted original img range before computing SSIM: [{image_original_tensor.min()}, {image_original_tensor.max()}]")


    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()
    ms_ssim_res = ms_ssim(reconstructed_img.unsqueeze(0),  image_original_tensor)
    #print(f"MultiScale Structural Similarity: {ms_ssim_res:.4f}")
    # diff2 = ImageChops.difference(Image.fromarray((flipped * 255).astype(np.uint8)), Image.fromarray((manip_img * 255).astype(np.uint8)))
    #print("------------------------------------------------")
    if out_file_dir is not None:
        with open(os.path.join(out_file_dir, "ssim_info.txt"), 'a') as f:
            f.write("\nImage Similarity: {:.4f}%".format(ssim * 100))
            f.write(f"\nMSE: {mse:.4f}")
            f.write(f"\nMultiScale Structural Similarity: {ms_ssim_res:.4f}")
            f.write("\n------------------------------------------------\n")

    return diff, ssim, mse, ms_ssim_res # diff2


def convert2rgb(img):
    convert_img = img.clone().detach()
    if convert_img.min() < 0:
        # transform pixel values into the [0, 1] range
        convert_img = (convert_img + 1) / 2
    return convert_img.cpu()


def get_isMSIH(patient_ids, dataframe):
    patient_ids = [("-").join(id.split("-")[:3]) for id in patient_ids]
    filtered_df = dataframe.set_index('PATIENT').loc[patient_ids]
    return filtered_df['isMSIH'].to_dict()
