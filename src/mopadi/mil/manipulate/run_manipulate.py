import os
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
import h5py
import json

from mopadi.mil.manipulate.manipulator_mil import ImageManipulator
from mopadi.dataset import DefaultTilesDataset
from mopadi.configs.templates import default_autoenc
from mopadi.configs.templates_cls import default_mil_conf


def run_manipulate(config):

    conf = default_mil_conf(config)

    if conf.patients:
        print(f"Selected patients: {conf.patients}")

    save_dir = os.path.join(conf.out_dir, 'counterfactuals')
    if not os.path.exists(save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    if conf.images_dir:
        data = DefaultTilesDataset(
            root_dirs=[conf.images_dir],
            max_tiles_per_patient=None,
            cohort_size_threshold=None,
            do_normalize=conf.do_normalize,
            do_resize=conf.do_resize,
            img_size=conf.img_size,
            process_only_zips=conf.process_only_zips,
            cache_pickle_tiles_path=None,
            cache_cohort_sizes_path=None
            )
    elif conf.data_dirs is not None and conf.split == 'test':
        data = DefaultTilesDataset(
            root_dirs=conf.data_dirs,
            max_tiles_per_patient=None,
            cohort_size_threshold=None,
            test_patients_file_path=conf.test_patients_file_path,
            split=conf.split,
            do_normalize=conf.do_normalize,
            do_resize=conf.do_resize,
            img_size=conf.img_size,
            process_only_zips=conf.process_only_zips,
            cache_pickle_tiles_path=None,
            cache_cohort_sizes_path=None
            )
    else:
        print(f"Data directories in data conf: {conf.data_dirs}")
        print(f"Split in data conf: {conf.split}")
        print(f"Test patients file path in data conf: {conf.test_patients_file_path}")
        print(f"Images directory in mil_classifier conf: {conf.images_dir}")
        raise ValueError("No correct directory provided. Please provide a valid images directory.")

    print(f"Autoencoder path: {os.path.join(conf.base_dir, 'autoenc', 'last.ckpt')}")
    print(f"Classifier path: {os.path.join(conf.out_dir, 'full_model', 'PMA_mil.pth')}")
    manipulator = ImageManipulator(autoenc_config=default_autoenc(config),
                                   autoenc_path=os.path.join(conf.base_dir, 'autoenc', 'last.ckpt'), 
                                   mil_path=os.path.join(conf.out_dir, 'full_model', 'PMA_mil.pth'),
                                   dataset=data,
                                   conf_cls=conf)

    if conf.clini_table.endswith(".xlsx"):
        clini_df = pd.read_excel(conf.clini_table)
    elif conf.clini_table.endswith(".csv"):
        clini_df = pd.read_csv(conf.clini_table)
    else:
        print("Clini table could not be read")

    feat_files = [f for f in os.listdir(conf.feat_path) if not f.startswith(".")]

    for patient_fname in tqdm(feat_files):
        res = {}
        patient_name = patient_fname.split(".")[0]

        if conf.patients is not None:
            if patient_name not in conf.patients:
                continue
        #print(f"Processing patient {patient_name}")

        if patient_name not in os.listdir(conf.images_dir):
            continue

        patient_id = "-".join(patient_name.split("-")[:conf.fname_index])
        if not clini_df.loc[clini_df["PATIENT"] == patient_id].empty:
            patient_class = clini_df.loc[clini_df["PATIENT"] == patient_id, conf.target_label].iloc[0]
        else:
            print(f"Patient {patient_id} not found in the clini table, skipping.")
            continue

        with h5py.File(os.path.join(conf.feat_path, patient_fname), "r") as hdf_file:
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
                metadata_decoded = [f"Tile_({y},{x})" for y, x in coords]

        save_patient_path = os.path.join(save_dir, patient_name)
            
        manipulator.manipulate_patients_images(
                            patient_name=patient_name, 
                            patient_features=features.unsqueeze(dim=0),
                            metadata=metadata_decoded,
                            save_path=save_patient_path, 
                            man_amps=conf.man_amps,
                            patient_class=patient_class,
                            target_dict=conf.target_dict,
                            num_top_tiles=conf.nr_top_tiles,
                            filename=conf.filename,
                            manip_tiles_separately=True
                            )
