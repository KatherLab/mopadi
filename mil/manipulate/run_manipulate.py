import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torchvision import transforms
import configs.templates_cls as configs
from dataset import *
from pathlib import Path

from mil.manipulate.manipulator_mil import ImageManipulator
from tqdm import tqdm
import pandas as pd
import torch
import h5py
from dotenv import load_dotenv
import argparse
import json

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="MIL Manip")
    parser.add_argument('--feat_path', type=str, required=True, help='Path to the test feature files')
    parser.add_argument('--images_dir', type=str, required=True, help='Path to the test tiles folder')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--clini_table', type=str, required=True, help='Path to the clinical table file')
    parser.add_argument('--conf_autoenc_name', type=str, required=True, help='Autoencoder configuration function name')
    parser.add_argument('--conf_mil_name', type=str, required=True, help='Classifier configuration function name')
    parser.add_argument('--autoenc_path', type=str, required=True, help='Path to the autoenc ckpt')
    parser.add_argument('--mil_path', type=str, required=True, help='Path to the PMA MIL pth file')
    parser.add_argument('--fname', type=str, required=False, default=None, help='If to manipulate specific tile, must be in top tiles!')
    parser.add_argument('--patients', type=str, required=False, default=None, help='If to manipulate tiles of specific patients')
    parser.add_argument('--nr_top_tiles', type=int, required=False, default=5, help='How many top tiles per patient to manipulate')
    parser.add_argument('--manip_levels', type=str, required=False, default="0.1,0.5,1.0", help='At what amplitudes to manipulate the images')
    parser.add_argument('--target_label', type=str, required=False, default=None, help='Target label if differs from the one in conf')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--target_dict', type=str, required=False, default=None, help='Target dictionary for the configuration (JSON format) if different')
    parser.add_argument('--fname_index', type=int, default=3, help='How to split filename to get patient ID')

    args = parser.parse_args()

    feat_path = args.feat_path
    images_dir = args.images_dir
    save_dir = args.out_dir
    clini_table = args.clini_table
    conf_autoenc_func_name = args.conf_autoenc_name
    conf_mil_func_name = args.conf_mil_name
    num_workers = args.num_workers
    autoenc_path = args.autoenc_path
    mil_path = args.mil_path
    filename = args.fname
    target_label = args.target_label
    target_dict = args.target_dict
    fname_index = args.fname_index

    if args.patients:
        patients = args.patients.split(',')
    else:
        patients = None
    print(f"Selected patients: {patients}")

    nr_top_tiles = args.nr_top_tiles
    man_amps = [float(x) for x in args.manip_levels.split(',')]

    conf_func = getattr(configs, conf_autoenc_func_name)
    conf = conf_func()

    conf_func_mil = getattr(configs, conf_mil_func_name)
    conf_cls = conf_func_mil()

    if not target_label:
         target_label = conf_mil.target_label
    if not target_dict:
        target_dict = conf_mil.target_dict
    else:
        try:
            target_dict = json.loads(args.target_dict)
        except json.JSONDecodeError as e:
            print(f"Error parsing target_dict: {e}")
            exit(1)

    print(f"Target dict: {target_dict}")

    if not os.path.exists(save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    data = BrainCacheDataset(images_dir=images_dir, 
                       transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    manipulator = ImageManipulator(autoenc_config=conf,
                                   autoenc_path=autoenc_path, 
                                   mil_path=mil_path,
                                   dataset=data,
                                   conf_cls=conf_cls)

    if clini_table.endswith(".xlsx"):
        clini_df = pd.read_excel(clini_table)
    elif clini_table.endswith(".csv"):
        clini_df = pd.read_csv(clini_table)
    else:
        print("Clini table could not be read")

    feat_files = [f for f in os.listdir(feat_path) if not f.startswith(".")]

    for patient_fname in tqdm(feat_files):
        res = {}
        patient_name = patient_fname.split(".")[0]

        #if not "TCGA-E2" in patient_name:
        #    continue

        if patients is not None:
            if patient_name not in patients:
                continue
        #print(f"Processing patient {patient_name}")

        found_folder = next((item for item in os.listdir(images_dir) if patient_name in item), None)
        if found_folder and patient_name != found_folder:
            print(f"Patient '{patient_name}' found as '{found_folder}'")
        else:
            print(f"Patient '{patient_name}' not found in the images directory, skipping...")
            continue

        patient_id = "-".join(patient_name.split("-")[:fname_index])
        if not clini_df.loc[clini_df["PATIENT"] == patient_id].empty:
            patient_class = clini_df.loc[clini_df["PATIENT"] == patient_id, target_label].iloc[0]
        else:
            print(f"Patient {patient_id} not found in the clini table, skipping.")
            continue

        with h5py.File(os.path.join(feat_path, patient_fname), "r") as hdf_file:
            #features = torch.from_numpy(hdf_file["features"][:])
            if 'feats' in hdf_file:
                features = torch.from_numpy(hdf_file['feats'][:])
            elif 'features' in hdf_file:
                features = torch.from_numpy(hdf_file['features'][:])
            else:
                raise ValueError(f"Neither 'feats' nor 'features' found in {feat_path}")

            if 'metadata' in hdf_file:
                metadata = hdf_file["metadata"][:]
                #print(metadata)
                metadata_decoded = [str(item, "utf-8") for item in metadata]

            if 'coords' in hdf_file:
                coords = hdf_file["coords"][:]
                #print(coords)
                metadata_decoded = [f"Tile_({y},{x})" for y, x in coords]

        save_patient_path = os.path.join(save_dir, patient_name)
            
        manipulator.manipulate_patients_images(
                            patient_name=patient_name, 
                            patient_features=features.unsqueeze(dim=0),
                            metadata=metadata_decoded,
                            save_path=save_patient_path, 
                            man_amps=man_amps,
                            patient_class=patient_class,
                            target_dict=target_dict,
                            num_top_tiles=nr_top_tiles,
                            filename=filename,
                            manip_tiles_separately=True
                            )
