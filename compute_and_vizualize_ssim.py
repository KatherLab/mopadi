import os
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from skimage.metrics import structural_similarity, mean_squared_error
from PIL import Image
import concurrent.futures
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
from cmcrameri import cm
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv('WORKSPACE_PATH')

def compute_structural_similarity(manip_img, image_original):
    # convert to grayscale
    before_gray = rgb_to_grayscale(image_original)
    after_gray =rgb_to_grayscale(manip_img)

    # SSIM
    (ssim, diff) = structural_similarity(before_gray, after_gray, full=True, data_range=before_gray.max() - before_gray.min())
    # print("Image Similarity: {:.4f}%".format(ssim * 100))
    diff = (diff * 255).astype("uint8")
    
    # MEAN SQUARE ERROR (MSE)
    # mse = mean_squared_error(before_gray, after_gray)
    # print(f"MSE: {mse:.4f}")

    # MULTI-SCALE SSIM
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()
    # transform image data that is initially in the range [-1, 1] to the range [0, 1]
    # ms_ssim_res = ms_ssim(manip_img.unsqueeze(0).cpu(), image_original.unsqueeze(0).cpu())
    # print(f"MultiScale Structural Similarity: {ms_ssim_res}")

    # diff2 = ImageChops.difference(Image.fromarray((flipped * 255).astype(np.uint8)), Image.fromarray((manip_img * 255).astype(np.uint8)))

    return diff, ssim #, mse, ms_ssim_res # diff2


def rgb_to_grayscale(image_array):
    # Check if image_array is already in grayscale
    if image_array.ndim == 2 or (image_array.ndim == 3 and image_array.shape[2] == 1):
        # Image is already in grayscale
        return image_array
    # Apply the luminosity method
    grayscale_image = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
    return grayscale_image


def determine_class(filename, target_dict):
    current_abbr = next((abbr for name, abbr in target_dict.items() if name in filename), None)
    return current_abbr, next((abbr for abbr in target_dict.values() if abbr != current_abbr), None)


def get_manipulated_images(path, target_dict, amps):
    fnames = [f for f in os.listdir(path) if not f.startswith(".")]
    images = {}
    for amp in amps:
        amp_edited = "amp_{:.1f}".format(amp).replace('.', ',')
        for fname in fnames:
            if amp_edited in fname:
                images[amp] = np.array(Image.open(os.path.join(path, fname)).convert('RGB')).astype(np.uint8) / 255.0
            elif "original" in fname:
                ori_img = np.array(Image.open(os.path.join(path, fname)).convert('RGB')).astype(np.uint8) / 255.0
                true_class, manip_class = determine_class(fname, target_dict)
    return ori_img, images, true_class, manip_class


def compute_ssim_for_images(ori_img, imgs):
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(compute_structural_similarity, imgs[amp], ori_img): amp for amp in imgs}
        for future in concurrent.futures.as_completed(futures):
            amp = futures[future]
            diff, ssim_value = future.result()
            results[amp] = (ssim_value, diff)
    return results


if __name__ == "__main__":

    results_dir = os.path.join(ws_path, 'results-manipulated', 'Lung-subtypes')
    target_dict = {"Lung_squamous_cell_carcinoma": "LUSC", "Lung_adenocarcinoma": "LUAD"}

    man_amps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    patients = [f for f in os.listdir(results_dir) if not f.startswith(".")]

    for patient in tqdm(patients):
        paths = [os.path.join(results_dir, patient, f) for f in os.listdir(os.path.join(results_dir, patient)) if not f.startswith(".")]
        for path in paths:
            ori, imgs, true_class, manip_class = get_manipulated_images(path, target_dict, man_amps)

            if len(imgs) < len(man_amps):
                continue
            
            ssim_results = compute_ssim_for_images(ori, imgs)

            num_rows=1
            num_cols=7
            img_height, img_width = (300, 300)
            dpi = 96
            fig_width = (img_width * num_cols) / dpi
            fig_height = (img_height * num_rows) / dpi

            fig, ax = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), dpi=dpi)

            # ax[0].imshow(ori)
            ax[0].axis('off')
            # ax[0].set_title(f'Original: {true_class}')

            for i, amp in enumerate(man_amps):
                ax[i+1].imshow(ssim_results[amp][1], cmap=cm.acton)
                ax[i+1].set_title("SSIM: {sim:.3f}".format(sim = ssim_results[amp][0]))
                ax[i+1].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(os.path.join(path, "ssim_all.png")))
            # print(f"saved to {os.path.join(os.path.join(path, 'ssim_all.png'))}")
            plt.close()
