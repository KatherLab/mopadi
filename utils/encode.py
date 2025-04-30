from torchvision import transforms
import torch
from configs.templates import *
from configs.templates_cls import *


class ImageEncoder:

    def __init__(self, autoenc_config, autoenc_path, dataset=None, device="cuda:0"):
        self.device = device
        self.dataset = dataset
        self.model = self._load_model(autoenc_config, autoenc_path)
    
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
    
    def encode_semantic(self, image):
        return self.model.encode(image.to(self.device)).squeeze().cpu().numpy()

    def encode_image(self, image, T=250):

        cond = self.model.encode(image.to(self.device))
        xT = self.model.encode_stochastic(image.to(self.device), cond, T)

        # Save and return original image path
        # out_path = os.path.join(save_path, self.image_dataset[image_index]["filename"].split(".")[0] + "_original.png")
        # save_image(convert2rgb(image), str(out_path))

        return cond, xT

    def sample_unconditional(self, nr_samples, T=20, T_latent=200):
        imgs = self.model.sample(nr_samples, device=self.device, T=T, T_latent=T_latent)
        return imgs

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


def get_isMSIH(patient_ids, dataframe):
    patient_ids = [("-").join(id.split("-")[:3]) for id in patient_ids]
    filtered_df = dataframe.set_index('PATIENT').loc[patient_ids]
    return filtered_df['isMSIH'].to_dict()
