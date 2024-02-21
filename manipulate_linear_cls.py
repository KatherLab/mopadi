from exp_linear_cls import ClsModel
from torchvision import transforms
import torch
from configs.templates import *
from configs.templates_cls import *
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from skimage.metrics import structural_similarity, mean_squared_error
import cv2


class ImageManipulator:
    def __init__(self, autoenc_config, autoenc_path, cls_config, cls_path, device="cuda:0"):
        self.device = device
        self.image_dataset = None
        self.cls_config = cls_config
        self.model = self.load_model(autoenc_config, autoenc_path)
        self.classifier = self.load_cls_model(cls_config, cls_path)
        print("Both models loaded successfully.")

        if self.cls_config.manipulate_mode in [ManipulateMode.texture_all]:
            self.VALID_TARGET_CLASSES = TextureAttrDataset.id_to_cls
        elif self.cls_config.manipulate_mode in [ManipulateMode.tcga_crc_msi]:
            self.VALID_TARGET_CLASSES = TcgaCrcMsiAttrDataset.id_to_cls
        elif self.cls_config.manipulate_mode in [ManipulateMode.tcga_crc_braf]:
            self.VALID_TARGET_CLASSES = TCGACRCBRAFAttrDataset.id_to_cls
        elif self.cls_config.manipulate_mode in [ManipulateMode.brain]:
            self.VALID_TARGET_CLASSES = BrainAttrDataset.id_to_cls
        else:
            print('Target classes could not be determined.')
        print(f"Valid target classes: {self.VALID_TARGET_CLASSES}")
    
    def load_data(self, image_folder):
        print(f"Number of images found in the given image folder: {len(os.listdir(image_folder))}")
        self.image_dataset = ImageDataset(image_folder, do_augment=False, do_normalize=True)
        return self.image_dataset

    def load_model(self, model_config, checkpoint_path):
        model = LitModel(model_config)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=False)
        model.eval()
        model.to(self.device)
        return model
    
    def load_cls_model(self, cls_config, cls_path):
        cls_model = ClsModel(cls_config)
        state = torch.load(cls_path)
        print('latent step:', state['global_step'])
        cls_model.load_state_dict(state['state_dict'], strict=False)
        cls_model.to(self.device)
        return cls_model

    def manipulate_image(self, 
                        image_index, 
                        target_class, 
                        save_path=os.path.join(os.getcwd(), "results"), 
                        manipulation_amplitude=0.3, 
                        T_step=100, 
                        T_inv=200,
                        dataset=None):
        assert self.image_dataset is not None or dataset is not None, "Data is missing."
        self.image_dataset = dataset

        if target_class not in self.VALID_TARGET_CLASSES:
            raise ValueError(f"Invalid target class. Valid options are: {self.VALID_TARGET_CLASSES}")

        batch = self.image_dataset[image_index]["img"][None]

        # Image encoding and manipulation steps
        results = {}
        semantic_latent = self.model.encode(batch.to(self.device))
        stochastic_latent = self.model.encode_stochastic(
            batch.to(self.device), semantic_latent, T=T_inv
        )
        results["semantic latent"] = semantic_latent
        results["stochastic latent"] = stochastic_latent

        if self.cls_config.manipulate_mode in [ManipulateMode.texture_all]:
            cls_id = TextureAttrDataset.cls_to_id[target_class]
        elif self.cls_config.manipulate_mode in [ManipulateMode.tcga_crc_msi]:
            cls_id = TcgaCrcMsiAttrDataset.cls_to_id[target_class]
        elif self.cls_config.manipulate_mode in [ManipulateMode.tcga_crc_braf]:
            cls_id = TCGACRCBRAFAttrDataset.cls_to_id[target_class]
        elif self.cls_config.manipulate_mode in [ManipulateMode.brain]:
            cls_id = BrainAttrDataset.cls_to_id[target_class]

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

        # Render Manipulated image
        manipulated_img = self.model.render(stochastic_latent, manipulated_semantic_latent, T=T_step)[0]

        # Save the original image
        try:
            out_path = os.path.join(save_path, self.image_dataset[image_index]["filename"].split(".png")[0] + "_original.png")
        except:
            out_path = os.path.join(save_path,  f"original_{image_index}.png")
        original_img_rgb = convert2rgb(self.image_dataset[image_index]["img"])
        results["original image"] = original_img_rgb
        results["original image path"] = out_path
        save_image(original_img_rgb, str(out_path))
        print(f"Original image saved to: {out_path}")

        # Save and return manipulated image path
        try:
            save_manip_path = os.path.join(save_path, self.image_dataset[image_index]["filename"].split(".png")[0] + f"_manipulated_to_{target_class}_amplitude_{manipulation_amplitude}.png")
        except:
            save_manip_path = os.path.join(save_path, f"Image_{image_index}_manipulated_to_{target_class}_amplitude_{manipulation_amplitude}.png")

        manipulated_img_rgb = convert2rgb(manipulated_img)
        results["manipulated image"] = manipulated_img_rgb
        results["manipulated image path"] = save_manip_path
        self.save_image(manipulated_img_rgb, save_manip_path)
        print(f"Manipulated image saved to: {save_manip_path}")
        return results

    def save_image(self, image_tensor, save_path):
        image = transforms.ToPILImage()(image_tensor.cpu().squeeze(0))
        image.save(save_path)


def compute_structural_similarity(reconstructed_img, image_original_tensor):
    reconstructed_img_np = reconstructed_img.permute(2, 1, 0).cpu().numpy()
    image_original = image_original_tensor.permute(2, 1, 0).cpu().detach().numpy()

    if image_original.min() < 0:
        # transform image data that is initially in the range [-1, 1] to the range [0, 1]
        image_original = (image_original + 1) / 2

    # convert to grayscale
    before_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(reconstructed_img_np, cv2.COLOR_BGR2GRAY)

    # SSIM
    (ssim, diff) = structural_similarity(before_gray, after_gray, full=True, data_range=before_gray.max() - before_gray.min())
    print("Image Similarity: {:.4f}%".format(ssim * 100))
    diff = (diff * 255).astype("uint8")
    
    # MEAN SQUARE ERROR (MSE)
    mse = mean_squared_error(before_gray, after_gray)
    print(f"MSE: {mse:.4f}")

    # MULTI-SCALE SSIM
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()
    ms_ssim_res = ms_ssim(reconstructed_img.unsqueeze(0),  image_original_tensor.unsqueeze(0))
    print(f"MultiScale Structural Similarity: {ms_ssim_res}")
    # diff2 = ImageChops.difference(Image.fromarray((flipped * 255).astype(np.uint8)), Image.fromarray((manip_img * 255).astype(np.uint8)))
    print("------------------------------------------------")
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
