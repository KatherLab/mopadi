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


def compute_structural_similarity2(reconstructed_img, image_original_tensor):
    reconstructed_img_np = reconstructed_img.permute(2, 1, 0).cpu().numpy()
    image_original = image_original_tensor.permute(2, 1, 0).cpu().detach().numpy()

    print(f"Shape of the original image: {image_original.shape}, and range: [{image_original.min()}, {image_original.max()}]")
    print(f"Shape of the reconstructed image: {reconstructed_img_np.shape}, and range: [{reconstructed_img_np.min()}, {reconstructed_img_np.max()}]")

    if image_original.min() < 0:
        # transform image data that is initially in the range [-1, 1] to the range [0, 1]
        image_original = (image_original + 1) / 2
        print(f"Adjusted range of the original image: {image_original.min()}, {image_original.max()}")

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
    print(f"Shapes and ranges before computing SSIM: reconstructed image {reconstructed_img.unsqueeze(0).size()}, range: [{reconstructed_img.unsqueeze(0).min()}, {reconstructed_img.unsqueeze(0).max()}], and origial: {image_original_tensor.unsqueeze(0).size()}, [{image_original_tensor.unsqueeze(0).min()}, {image_original_tensor.unsqueeze(0).max()}]")

    image_original_tensor = image_original_tensor.unsqueeze(0)
    if image_original_tensor.min() < 0:
        # transform image data that is initially in the range [-1, 1] to the range [0, 1]
        image_original_tensor = (image_original_tensor.cpu() + 1) / 2
        print(f"Adjusted original img range before computing SSIM: [{image_original_tensor.min()}, {image_original_tensor.max()}]")


    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()
    ms_ssim_res = ms_ssim(reconstructed_img.unsqueeze(0),  image_original_tensor)
    print(f"MultiScale Structural Similarity: {ms_ssim_res}")
    # diff2 = ImageChops.difference(Image.fromarray((flipped * 255).astype(np.uint8)), Image.fromarray((manip_img * 255).astype(np.uint8)))
    print("------------------------------------------------")
    return diff, ssim, mse, ms_ssim_res # diff2


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

def determine_center(filename, target_dict):
    current_abbr = next((abbr for name, abbr in target_dict.items() if name in filename), None)
    return current_abbr, next((abbr for abbr in target_dict.values() if abbr != current_abbr), None)

def get_manipulated_images(path, target_dict, amps):
    fnames = [f for f in os.listdir(path) if not f.startswith(".") and not f.endswith(".txt")]
    images = {}
    ori_img = None
    true_class = None
    manip_class = None
    for amp in amps:
        amp_edited = "amp_{:.2f}".format(amp).replace('.', ',')
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

    # Lung cancer types
    #results_dir = os.path.join(ws_path, 'results-manipulated', 'Lung-subtypes-newest-clf-512')
    #target_dict = {"Lung_squamous_cell_carcinoma": "LUSC", "Lung_adenocarcinoma": "LUAD"}

    # Lung cancer types new
    #results_dir = f"{ws_path}/mopadi/checkpoints/pancancer/mil_lung/Type/manipulate_results"
    #target_dict = {"LUAD": "LUAD", "LUSC": "LUSC"}

    # BRCA batch effects
    #results_dir = f"{ws_path}/mopadi/checkpoints/brca/is_E2/manipulate_results-newfeats2"
    #target_dict = {"TCGA-E2": "yes", "original_No": "no"}

    # CRC MSI
    #results_dir = f"{ws_path}/mopadi/checkpoints/crc/isMSIH/manipulate_results2"
    #target_dict = {"MSIH": "MSIH", "nonMSIH": "nonMSIH"}

    # BRCA cancer types
    results_dir = f"{ws_path}/mopadi/checkpoints/brca/BRCA_Pathology/manipulate_results-newfeats2"
    target_dict = {"IDC": "IDC", "ILC": "ILC"}

    man_amps = [0.010, 0.020, 0.030, 0.040, 0.050]
    patients = [f for f in os.listdir(results_dir) if not f.startswith(".")]

    for patient in tqdm(patients):
        paths = [os.path.join(results_dir, patient, f) for f in os.listdir(os.path.join(results_dir, patient)) if not f.startswith(".") and not f.endswith(".txt")]
        for path in paths:
            ori, imgs, true_class, manip_class = get_manipulated_images(path, target_dict, man_amps)

            if ori is None:
                continue

            if len(imgs) < len(man_amps):
                continue
            
            ssim_results = compute_ssim_for_images(ori, imgs)

            num_rows=1
            num_cols=len(man_amps)+1
            img_height, img_width = (300, 300)
            dpi = 96
            fig_width = (img_width * num_cols) / dpi
            fig_height = (img_height * num_rows) / dpi

            fig, ax = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), dpi=dpi)

            # ax[0].imshow(ori)
            ax[0].axis('off')
            # ax[0].set_title(f'Original: {true_class}')

            for i, amp in enumerate(man_amps):
                ax[i+1].imshow(ssim_results[amp][1], cmap=cm.devon)
                ax[i+1].set_title(f"SSIM: {ssim_results[amp][0]:.3f} {amp}")
                ax[i+1].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(os.path.join(path, "ssim_all.png")))
            # print(f"saved to {os.path.join(os.path.join(path, 'ssim_all.png'))}")
            plt.close()
