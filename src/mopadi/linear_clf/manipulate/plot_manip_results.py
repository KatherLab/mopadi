import os
from PIL import Image
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv('WORKSPACE_PATH')


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
                images[amp] = (Image.open(os.path.join(path, fname)).convert('RGB'))
            elif "original" in fname:
                ori_img = Image.open(os.path.join(path, fname)).convert('RGB')
                true_class, manip_class = determine_class(fname, target_dict)
    return ori_img, images, true_class, manip_class


def process_path(path, target_dict, man_amps):
    ori, imgs, true_class, manip_class = get_manipulated_images(path, target_dict, man_amps)
    if len(imgs) < len(man_amps) or ori is None:
        print("Skipping")
        return
    num_rows = 1
    num_cols = len(man_amps) + 1
    img_height, img_width = (300, 300)
    dpi = 96
    fig_width = (img_width * num_cols) / dpi
    fig_height = (img_height * num_rows) / dpi
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), dpi=dpi)
    ax[0].imshow(ori)
    ax[0].axis('off')
    ax[0].set_title(f'Original: {true_class}')
    for i, amp in enumerate(man_amps):
        ax[i+1].imshow(imgs[amp])
        ax[i+1].axis('off')
        ax[i+1].set_title(f"+{manip_class}*{amp}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(path, "manipulated_all.png"))
    plt.close()


def main():
    # LUNG SUBTYPES ----------------------------------------------------------------------
    results_dir = f'{ws_path}/mopadi/checkpoints/lung/new-final/full_model/manip-trial-01'
    target_dict = {"Lung_squamous_cell_carcinoma": "LUSC", "Lung_adenocarcinoma": "LUAD"}
    man_amps = [0.01, 0.02, 0.03, 0.04, 0.05]

    # CRC MSI ----------------------------------------------------------------------------
    #results_dir = f'{ws_path}/results-manipulated/msi-final-ppt-512'
    #target_dict = {"nonMSIH":"nonMSIH", "MSIH":"MSIH"}
    #man_amps = [0.4, 0.8, 1.2, 1.6, 2.0]

    num_workers = 8
    patients = [f for f in os.listdir(results_dir) if not f.startswith(".")]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for patient in patients:
            paths = [os.path.join(results_dir, patient, f) for f in os.listdir(os.path.join(results_dir, patient)) if not f.startswith(".")]
            for path in paths:
                futures.append(executor.submit(process_path, path, target_dict, man_amps))
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


if __name__ == "__main__":
    main()
