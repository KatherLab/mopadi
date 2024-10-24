import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import TextureDataset
from configs.templates import texture100k_autoenc
from concurrent.futures import ThreadPoolExecutor, as_completed
from configs.templates_cls import texture100k_autoenc_cls
from manipulate_linear_cls import ImageManipulator, compute_structural_similarity
from cmcrameri import cm
from dotenv import load_dotenv
from tqdm import tqdm
import sys

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


def process_image(index, data, image_manipulator, lvls):
    fname = data[index]['filename']

    if not ("TUM" in fname or "NORM" in fname):
        return None

    if "TUM" in fname:
        target_class = "NORM"
        ori_class = "TUM"
    elif "NORM" in fname:
        target_class = "TUM"
        ori_class = "NORM"

    save_dir = os.path.join("results", "CRC-VAL-HE-7K_all", fname.split(".tif")[0] + f"_to_{target_class}")
    os.makedirs(save_dir, exist_ok=True)

    res = []

    for lvl in lvls:
        res.append(image_manipulator.manipulate_image(
            dataset=data,
            image_index=index,
            target_class=target_class,
            save_path=save_dir,
            manipulation_amplitude=lvl,
            T_step=100,
            T_inv=200
        ))

    sim = [(np.ones((224, 224, 3), dtype=np.uint8) * 255, "")]
    for i in range(len(res)):
        sim.append(compute_structural_similarity(res[i]["manip_img"], res[0]["ori_img"].squeeze(), out_file_dir=save_dir))

    num_rows = 2
    num_cols = 6

    img_height, img_width = (300, 300)
    dpi = 96  # dots per inch
    fig_width = (img_width * num_cols) / dpi
    fig_height = (img_height * num_rows) / dpi

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), dpi=dpi)

    ori = res[0]["ori_img"].unsqueeze(0)
    ax[0, 0].imshow(ori[0].permute(1, 2, 0).cpu())
    ax[0, 0].axis('off')
    ax[0, 0].set_title(f'Original: {ori_class}')

    for i in range(len(res)):
        ax[0, i + 1].imshow(res[i]["manip_img"].permute(1, 2, 0).cpu())
        ax[0, i + 1].set_title(f"+ {target_class} * {str(lvls[i])}")
        ax[0, i + 1].axis('off')

        ssim_image_rotated = np.rot90(sim[i + 1][0], k=-1)
        flipped_ssim = np.fliplr(ssim_image_rotated)
        ax[1, i + 1].imshow(flipped_ssim, cmap=cm.devon)
        ax[1, i + 1].set_title("SSIM {sim:.3f}".format(sim=sim[i + 1][1]))
        ax[1, i + 1].axis('off')

    fig.canvas.draw()
    plt.savefig(os.path.join(save_dir, "ssim.png"), bbox_inches='tight')
    plt.close()

    # Reverse the list to show the most manipulated image first
    res.reverse()

    sim = [(np.ones((224, 224, 3), dtype=np.uint8) * 255, "")]
    most_manipulated_image = res[0]["manip_img"]
    for i in range(len(res)-1):
        sim.append(compute_structural_similarity(res[i+1]["manip_img"], most_manipulated_image.squeeze(), out_file_dir=save_dir))
    sim.append(compute_structural_similarity(res[0]["ori_img"], most_manipulated_image.squeeze(), out_file_dir=save_dir))

    num_rows = 2
    num_cols = 6

    img_height, img_width = (300, 300)
    dpi = 96  # dots per inch
    fig_width = (img_width * num_cols) / dpi
    fig_height = (img_height * num_rows) / dpi

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), dpi=dpi)
    ax[1, 0].axis('off')

    for i in range(len(res)):
        ax[0, i].imshow(res[i]["manip_img"].permute(1, 2, 0).cpu())
        ax[0, i].set_title(f"+ {target_class} * {str(lvls[len(lvls) - 1 - i])}")
        ax[0, i].axis('off')

        ssim_image_rotated = np.rot90(sim[i+1][0], k=-1)
        flipped_ssim = np.fliplr(ssim_image_rotated)
        ax[1, i+1].imshow(flipped_ssim, cmap=cm.devon)
        ax[1, i+1].set_title("SSIM {sim:.3f}".format(sim=sim[i+1][1]))
        ax[1, i+1].axis('off')

    ori = res[0]["ori_img"].unsqueeze(0)
    ax[0,num_cols-1].imshow(ori[0].permute(1, 2, 0).cpu())
    ax[0,num_cols-1].axis('off')
    ax[0,num_cols-1].set_title(f'Original: {ori_class}')

    fig.canvas.draw()
    plt.savefig(os.path.join(save_dir, "ssim_flipped.png"), bbox_inches='tight')
    plt.close()

    return fname

def main():
    images_dir = f'{ws_path}/data/texture100k/CRC-VAL-HE-7K'
    data = TextureDataset(images_dir, do_augment=False, do_normalize=True, do_transform=True)

    image_manipulator = ImageManipulator(
        autoenc_config=texture100k_autoenc(),
        autoenc_path="checkpoints/texture100k/last.ckpt",
        cls_path="checkpoints/texture100k/texture100k_autoenc_cls/last.ckpt",
        cls_config=texture100k_autoenc_cls(),
    )

    lvls = [0.2, 0.4, 0.6, 0.8, 1.0]
    num_workers = 8

    os.makedirs(os.path.join("results", "CRC-VAL-HE-7K_all"), exist_ok=True)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_image, i, data, image_manipulator, lvls) for i in range(len(data))]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                if result is not None:
                    print(f"Processed: {result}")
            except Exception as exc:
                print(f"Generated an exception: {exc}")

if __name__ == '__main__':
    main()
