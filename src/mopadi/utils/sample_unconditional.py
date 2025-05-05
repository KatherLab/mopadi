from mopadi.configs.templates import *
from mopadi.configs.templates_latent import *
from tqdm import tqdm
from dotenv import load_dotenv
from torch.multiprocessing import Pool, set_start_method
import torch.multiprocessing as mp

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


def main(device, model_path, output_folder, conf, num_images=10000):
    conf.T_eval = 100
    conf.latent_T_eval = 200
    print(conf.name)

    model = load_model(model_path, device, conf)
    os.makedirs(output_folder, exist_ok=True)

    with get_context("spawn").Pool(processes=8) as pool:
        tasks = [(model, device, output_folder, i) for i in range(num_images)]
        list(tqdm(pool.imap_unordered(generate_image, tasks), total=num_images))


def load_model(model_path, device, conf):
    model = LitModel(conf)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    model.to(device)
    return model


def generate_image(args):
    model, device, output_folder, index = args
    image = model.sample(1, device=device, T=100, T_latent=200)
    filename = f"image_{index:05d}.png"
    save_image(image.squeeze(0), filename=filename, save_path=output_folder)


def save_image(image_tensor, filename, save_path):
    save_path = os.path.join(save_path, filename)
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(image_tensor)
    pil_image.save(save_path)
    # print(f"Image saved at {save_path}")


if __name__== "__main__":

    # Texture100k
    # model_path = f"{ws_path}/mopadi/checkpoints/exp07-texture-latent-cls/texture100k_autoenc_latent/last.ckpt"
    # output_folder = f"{ws_path}/mopadi/generated_synthetic_images/Texture"

    # TCGA-CRC NEW
    #model_path = f"{ws_path}/mopadi/checkpoints/crc/tcga_crc_512/tcga_crc_512_latent/last.ckpt"
    #output_folder = f"{ws_path}/generated_synthetic_images/TCGA-CRC-new"
    #conf = tcga_crc_512_latent()

    # TCGA-BRCA
    #model_path = f"{ws_path}/mopadi/checkpoints/brca/brca_latent/last.ckpt"
    #output_folder = f"{ws_path}/generated_synthetic_images/TCGA-BRCA-T1k-latent1k"
    #conf = tcga_brca_latent()

    # Pancancer
    model_path = f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/pancancer_latent/last.ckpt"
    output_folder = f"{ws_path}/generated_synthetic_images/JAPAN"
    conf = pancancer_latent()
    
    mp.set_start_method('spawn')
    device = "cuda:1"

    main(device=device, model_path=model_path, output_folder=output_folder, conf=conf)
