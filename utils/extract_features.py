import os

from configs.templates import *
from configs.templates_cls import *
# from manipulate import *
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
from encode import ImageEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


class ImageFolder(Dataset):
    """
    Custom dataset class for handling image folders.
    """
    def __init__(self, folder, exts=["jpg", "png", "tif"]):
        super().__init__()
        self.folder = folder
        self.paths = sorted(
            [
                p for ext in exts 
                for p in Path(folder).glob(f"**/*.{ext}")
                if not any(part.startswith(".") for part in p.parts)
            ]
        )
        transform = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform)     
        return

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)

        transform = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform = transforms.Compose(transform)

        return self.transform(img)


def process_patient_data(patient_name, image_ids, dataset, encoder, save_dir):
    encoded_data_list = []
    metadata_list = []

    for i in tqdm(image_ids, desc = f"Processing patient {patient_name}"):
        #print(dataset.paths[i])
        metadata_key = str(dataset.paths[i]).split("/")[-1]
        #print(metadata_key)
        image = dataset[i]
        #print(image.shape)
        encoded_data = encoder.encode_semantic(image.unsqueeze(0))
        
        encoded_data_list.append(encoded_data)
        metadata_list.append(metadata_key)
    
    with h5py.File(f'{os.path.join(save_dir, str(patient_name))}.h5', 'w') as f:

        f.create_dataset('features', data=np.stack(encoded_data_list, axis=0))
        f.create_dataset('metadata', data=metadata_list, dtype=h5py.special_dtype(vlen=str))

    print(f'Features saved to {os.path.join(save_dir, str(patient_name))}.h5')


def main():

    # BRCA
    #autoenc_path = "checkpoints/brca/last.ckpt"
    #images_dir = f"{ws_path}/data/TCGA-BRCA/tiles-test-from-ganymede"
    #save_dir = f"{ws_path}/extracted_features/TCGA-BRCA"
    #conf = tcga_brca_autoenc()
    #imgs_ids_file = 'temp/patient_image_ids_dict_BRCA2-all.txt'

    # CRC
    autoenc_path = "checkpoints/crc/tcga_crc_512/last.ckpt"
    #lmdb_path = "/mnt/bulk-dgx/laura/mopadi/datasets/tcga_crc_512_lmdb-train"
    images_dir = "/mnt/bulk-mars/laura/diffae/data/TCGA-CRC/512x512_tumor_test"
    save_dir = f"{ws_path}/extracted_features/TCGA-CRC/512x512-test"
    conf = tcga_crc_512_autoenc()
    imgs_ids_file = 'temp/crc_patient_image_ids_dict_test.txt'


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    encoder = ImageEncoder(autoenc_config=conf, autoenc_path=autoenc_path, device="cuda:1")
    dataset = ImageFolder(images_dir)
    print(f"Total images: {len(dataset)}")

    if os.path.exists(imgs_ids_file):
        with open(imgs_ids_file, 'r') as f:
            patient_image_ids = json.load(f)
        print(f"Dictionary loaded from file: {imgs_ids_file}, length: {len(patient_image_ids.keys())}")
    else:
        patient_image_ids = defaultdict(list)
        for i in tqdm(range(len(dataset)), total=len(dataset), desc="Getting patients IDs"):
            #print(dataset.paths[i])
            patient_name = str(dataset.paths[i]).split("/")[8]
            #print(patient_name)
            patient_image_ids[patient_name].append(i)
        patient_image_ids = dict(patient_image_ids)
        with open(imgs_ids_file, 'w') as f:
            json.dump(patient_image_ids, f)


    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for patient_name, image_ids in tqdm(patient_image_ids.items(), desc="Processing Patients"):
            # Skip if file already exists
            if os.path.exists(f'{os.path.join(save_dir, str(patient_name))}.h5'):
                print("Already exists, skipping...")
                continue
            futures.append(executor.submit(process_patient_data, patient_name, image_ids, dataset, encoder, save_dir))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")


if __name__ == "__main__":
    main()
