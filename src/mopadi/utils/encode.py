from torchvision import transforms
import torch
import os

from mopadi.configs.templates import *


class ImageEncoder:

    def __init__(self, autoenc_config, autoenc_path, feat_extractor = None, dataset=None, device="cuda:0"):
        self.device = device
        self.dataset = dataset
        self.model = self._load_model(autoenc_config, autoenc_path)

        if self.model.feat_extractor is None:
            if feat_extractor == 'conch':
                self.model.feat_extractor = FeatureExtractorConch(device=device)
            elif feat_extractor == 'conch1_5':
                self.model.feat_extractor = FeatureExtractorConch15(device=device)
            elif feat_extractor == 'v2':
                self.model.feat_extractor = FeatureExtractorVirchow2(device=device)
            elif feat_extractor == 'uni2':
                self.model.feat_extractor = FeatureExtractorUNI2(device=device)
    
    def load_data(self, image_folder):
        print(f"Number of images found in the given image folder: {len(os.listdir(image_folder))}")
        self.image_dataset = self.dataset(image_folder, do_augment=False, do_normalize=True)
        return self.image_dataset

    def _load_model(self, model_config, checkpoint_path):
        model = LitModel(model_config)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=False)
        model.ema_model.eval()
        model.ema_model.to(self.device)
        return model

    def encode_to_noise(self, image, features, T=250):
        return self.model.encode_stochastic(image.to(self.device), features, T)

    def decode_image(self, xT, cond, T=20):
        return self.model.render(xT, cond, T)

    def save_image(self, image_tensor, filename, save_path):
        save_path = os.path.join(save_path, filename)
        image = transforms.ToPILImage()(image_tensor.cpu().squeeze(0))
        image.save(save_path)
        return save_path


def convert2rgb(img, adjust_scale=True):
    convert_img = torch.tensor(img)
    if adjust_scale:
        convert_img = (convert_img + 1) / 2
    return convert_img.cpu()