import os
from configs.templates import *
from encode import *
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import encode
import cv2
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from PIL import Image, ImageChops
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


def load_test_patients(split_file):
    with open(split_file, 'r') as file:
        data = json.load(file)
        test_patients = {patient for patient in data['Test set patients']}
        #test_patients = set()
        #for cancer_type, info in data.items():
        #    test_patients.update(info.get("Test set patients", []))
    return test_patients


def compute_structural_similarity(reconstructed_image, image_original, ms_ssim):
    # transform image data that is initially in the range [-1, 1] to the range [0, 1]
    img_ori = np.array((image_original.cpu().detach().numpy() + 1) / 2)
    flipped = np.swapaxes(img_ori, 0, 2)
    
    manip_img = np.array(reconstructed_image[0].permute(2, 1, 0).cpu())
    before_gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(manip_img, cv2.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(before_gray, after_gray, full=True, data_range=before_gray.max() - before_gray.min())
    # print("Image Structural Similarity: {:.4f}%".format(score * 100))

    # MULTI-SCALE SSIM
    adjusted_image = (image_original.unsqueeze(0).cpu() + 1) / 2 
    ms_ssim_res = ms_ssim(reconstructed_image.cpu(), adjusted_image)

    mse_none = mean_squared_error(before_gray, after_gray)
    # print(f"MSE: {mse_none:.4f}")
    return score, mse_none, ms_ssim_res.numpy()


def get_mse_for_one_image(data, idx, encoder, ms_ssim, save_path):
    
    random_data = data[idx]
    img = random_data['img'][None]
    #fname = random_data['filename'].split('/')[-2] + "_" + random_data['filename'].split('/')[-1].split(".")[0]

    cond, xT = encoder.encode_image(img)
    pred = encoder.decode_image(xT, cond)
    filename = f"reconstructed_image_{idx}.png"     #{fname}.png" 
    save_image(pred.squeeze(0), filename=filename, save_path=save_path)

    img_sim, mse, ms_ssim_res = compute_structural_similarity(pred, img.squeeze(), ms_ssim)

    return img_sim, mse, ms_ssim_res


def save_image(image_tensor, filename, save_path):
    save_path = os.path.join(save_path, filename)
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(image_tensor)

    pil_image.save(save_path)


def main(encoder_path, data, conf, save_path, tiles_path, split_file):

    #cancer_types_to_process = [
    #"Lung_adenocarcinoma", 
    #"Cholangiocarcinoma", 
    #"Lung_squamous_cell_carcinoma",
    #"Liver_hepatocellular_carcinoma"
    #]

    encoder = ImageEncoder(conf, autoenc_path=encoder_path, device="cuda:1")
    min_total_images = 1000
    random_indices = []
    all_patient_tiles = []
    total_tiles = 0

    #all_patients = [fname for fname in os.listdir(tiles_path) if fname.startswith("TCGA")]
    #selected_patients = all_patients
    test_patients = load_test_patients(split_file)

    print(f"Number of patients: {len(test_patients)}")

    #for cancer_type in cancer_types_to_process:
    #    cancer_type_path = os.path.join(tiles_path, cancer_type)
        
    for patient in test_patients:
        patient_folder = next((f for f in os.listdir(tiles_path) if f.startswith(patient)), None)  # os.listdir(cancer_type_path) if f.startswith(patient)), None)
        if not patient_folder:
            #print(f"Patient {patient} folder could not be found in {cancer_type}, skipping...")
            continue
        else:
            print(f"Patient {patient} folder could be found") # in {cancer_type}")
        
        path_patient_tiles = os.path.join(tiles_path, patient_folder) #cancer_type_path, patient_folder)
        all_tiles = [fname for fname in os.listdir(path_patient_tiles) if fname.endswith(".jpg")]
        nr_of_tiles = len(all_tiles)
        
        all_patient_tiles.append((patient, all_tiles, nr_of_tiles))
        total_tiles += nr_of_tiles

    min_total_images = 1000

    if total_tiles >= min_total_images:
        print(f"Total tiles exceed {min_total_images}. Prioritizing maximum patient inclusion.")
        for patient, tiles, nr_of_tiles in all_patient_tiles:
            random_indices.append(random.randint(0, nr_of_tiles - 1))  # take 1 tile per patient

        remaining_images_needed = min_total_images - len(random_indices)
        if remaining_images_needed > 0:
            # randomly distribute the remaining images evenly among patients
            for patient, tiles, nr_of_tiles in all_patient_tiles:
                if remaining_images_needed <= 0:
                    break
                extra_tiles_to_take = min(remaining_images_needed, nr_of_tiles - 1)  # Avoid duplicating the same tile
                random_indices.extend([random.randint(0, nr_of_tiles - 1) for _ in range(extra_tiles_to_take)])
                remaining_images_needed -= extra_tiles_to_take

    else:
        print(f"Total tiles are fewer than {min_total_images}. Distributing evenly.")
        avg_per_patient = min_total_images // len(all_patient_tiles)
        extra = min_total_images % len(all_patient_tiles)

        # select at least avg_per_patient tiles from each patient
        for patient, tiles, nr_of_tiles in all_patient_tiles:
            random_indices.extend([random.randint(0, nr_of_tiles - 1) for _ in range(min(avg_per_patient, nr_of_tiles))])

        # distrib extra tiles
        for patient, tiles, nr_of_tiles in random.sample(all_patient_tiles, extra):
            random_indices.extend([random.randint(0, nr_of_tiles - 1) for _ in range(1)])

    print(f"Number of indices determined: {len(random_indices)}")
    df = pd.DataFrame(columns=['Random_Index', 'SSIM',  'MS-SSIM', 'MSE'])
    iteration_count = 0 
    save_interval = 500
    data_list = []

    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()

    for i in tqdm(random_indices):
        img_sim, mse, ms_ssim_res = get_mse_for_one_image(data, i, encoder, ms_ssim, save_path)
        data_list.append({'Random_Index': i, 'SSIM': img_sim, 'MS-SSIM': ms_ssim_res, 'MSE': mse})
        iteration_count += 1

        if iteration_count % save_interval == 0:
            df = pd.concat([pd.DataFrame([data]) for data in data_list], ignore_index=True)
            df.to_csv(os.path.join(save_path, f'autoencoder-evaluation-results_{iteration_count}.csv'), index=False)

    df = pd.concat([pd.DataFrame([data]) for data in data_list], ignore_index=True)
    df.to_csv(os.path.join(save_path, 'autoencoder-evaluation-results-final.csv'), index=False)


if __name__=="__main__":

    # TCGA-CRC
    encoder_path = f"{ws_path}/mopadi/checkpoints/crc/tcga_crc_512/last.ckpt"
    save_path = f"{ws_path}/results/reconstructed_images/TCGA-CRC-newest-sep-21"
    lmdb_path = f"/mnt/bulk-dgx/laura/mopadi/datasets/tcga_crc_512_lmdb-test"
    tiles_path = f"/mnt/bulk-ganymede/laura/deep-liver/data/TCGA-CRC/tiles_512x512_05mpp"
    split_file = f"{ws_path}/data/TCGA-CRC/new_split/data_info.json"
    conf = tcga_crc_512_autoenc()

    # TCGA-BRCA
    #encoder_path = f"{ws_path}/mopadi/checkpoints/brca/autoenc/last.ckpt"
    #save_path = f"{ws_path}/results/reconstructed_images/TCGA-BRCA"
    #lmdb_path = f"{ws_path}/mopadi/datasets/brca/tcga-brca-512-test.lmdb"
    #tiles_path = f"{ws_path}/data/TCGA-BRCA/tiles-test"
    #split_file = f"{ws_path}/mopadi/datasets/brca/data_info.json"
    #conf = tcga_brca_autoenc()

    # JAPAN 
    #encoder_path = f"{ws_path}/mopadi/checkpoints/pancancer/autoenc/last.ckpt"
    #save_path = f"{ws_path}/results/reconstructed_images/JAPAN"
    #lmdb_path = f"{ws_path}/mopadi/datasets/pancancer/japan-lmdb-test"
    #tiles_path = f"{ws_path}/data/japan/new-all"
    #split_file = f"{ws_path}/data/japan/test_train_split.txt"
    #conf = pancancer_autoenc()

    data = TcgaCRCwoMetadata(lmdb_path, do_augment=False)

    os.makedirs(save_path, exist_ok=True)

    main(encoder_path, data, conf, save_path, tiles_path, split_file)
