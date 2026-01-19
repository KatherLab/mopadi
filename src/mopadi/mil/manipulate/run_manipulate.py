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
        print(f"Using images directory: {conf.images_dir}")
        print(f"Using feat path: {conf.feat_path_test}")
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

    if conf.use_pretrained:
        assert conf.pretrained_autoenc_name is not None, "Pretrained name must be provided if use_pretrained is True"
        from huggingface_hub import hf_hub_download, login

        if conf.pretrained_autoenc_name == 'crc_512_model':
            autoenc_model_path = hf_hub_download(
                repo_id="KatherLab/MoPaDi",
                filename="crc_512_model/autoenc.ckpt",
            )
            clf_model_path = hf_hub_download(
                repo_id="KatherLab/MoPaDi",
                filename="crc_512_model/mil_msi_classifier.pth",
            )
        elif conf.pretrained_autoenc_name == 'brca_512_model':
            autoenc_model_path = hf_hub_download(
                repo_id="KatherLab/MoPaDi",
                filename="brca_512_model/autoenc.ckpt",
            )
            if conf.pretrained_clf_name == 'e2_center':
                clf_model_path = hf_hub_download(
                    repo_id="KatherLab/MoPaDi",
                    filename="brca_512_model/mil_e2_center_classifier.pth",
                )
            elif conf.pretrained_clf_name == 'type':
                clf_model_path = hf_hub_download(
                    repo_id="KatherLab/MoPaDi",
                    filename="brca_512_model/mil_cancer_types_classifier.pth",
                )
            else:
                raise ValueError(f"Unknown pretrained classifier name: {conf.pretrained_clf_name}. Please provide a valid type (e2_center or type) to use the pretrained model or train the classifier from scratch.")
        elif conf.pretrained_autoenc_name == 'pancancer_model':
            autoenc_model_path = hf_hub_download(
                repo_id="KatherLab/MoPaDi",
                filename="pancancer_model/autoenc.ckpt",
            )
            if conf.pretrained_clf_name == 'lung':
                clf_model_path = hf_hub_download(
                    repo_id="KatherLab/MoPaDi",
                    filename="pancancer_model/mil_lung_classifier.pth",
                )
            elif conf.pretrained_clf_name == 'liver':
                clf_model_path = hf_hub_download(
                    repo_id="KatherLab/MoPaDi",
                    filename="pancancer_model/mil_liver_classifier.pth",
                )
            else:
                raise ValueError(f"Unknown pretrained classifier name: {conf.pretrained_clf_name}. Please provide a valid type (liver or lung) to use the pretrained model or train the classifier from scratch.")
        else:
            raise ValueError(f"Unknown pretrained autoencoder name: {conf.pretrained_autoenc_name}. Please provide a valid name (crc_512_model, brca_512_model, or pancancer_model). ")
        print(f"Autoencoder's checkpoint downloaded to: {autoenc_model_path}")
        print(f"Classifier's checkpoint downloaded to: {clf_model_path}")
    else:
        autoenc_model_path = os.path.join(conf.base_dir, 'autoenc', 'last.ckpt')
        clf_model_path = os.path.join(conf.out_dir, 'full_model', 'PMA_mil.pth')
        print(f"Autoencoder path: {autoenc_model_path}")
        print(f"Classifier path: {clf_model_path}")

    if conf.pretrained_autoenc_conf is not None:
        autoenc_conf = conf.pretrained_autoenc_conf
    else:
        autoenc_conf = default_autoenc(config)

    manipulator = ImageManipulator(autoenc_config=autoenc_conf,
                                   autoenc_path=autoenc_model_path, 
                                   mil_path=clf_model_path,
                                   dataset=data,
                                   conf_cls=conf)

    if conf.clini_table.endswith(".xlsx"):
        clini_df = pd.read_excel(conf.clini_table)
    elif conf.clini_table.endswith(".csv"):
        clini_df = pd.read_csv(conf.clini_table)
    else:
        print("Clini table could not be read")

    feat_files = [f for f in os.listdir(conf.feat_path_test) if not f.startswith(".")]
    imgs_files = ["-".join(f.split('_')[0].split('-')[:conf.fname_index]) for f in os.listdir(conf.images_dir) if not f.startswith(".")]

    for patient_feats in tqdm(feat_files):
        res = {}
        patient_fname = patient_feats.split(".")[0]
        patient_id = "-".join(patient_fname.split("-")[:conf.fname_index])

        if conf.patients is not None:
            if patient_id not in conf.patients:
                print(f"Skipping patient {patient_id} as it is not in the provided patients list.")
                continue

        if patient_id not in imgs_files:
            print(f"Patient {patient_id} not found in the images directory, skipping.")
            continue

        if not clini_df.loc[clini_df["PATIENT"] == patient_id].empty:
            patient_class = clini_df.loc[clini_df["PATIENT"] == patient_id, conf.target_label].iloc[0]
        else:
            print(f"Patient {patient_id} not found in the clini table, skipping.")
            continue

        with h5py.File(os.path.join(conf.feat_path_test, patient_feats), "r") as hdf_file:
            if 'feats' in hdf_file:
                features = torch.from_numpy(hdf_file['feats'][:])
            elif 'features' in hdf_file:
                features = torch.from_numpy(hdf_file['features'][:])
            else:
                raise ValueError(f"Neither 'feats' nor 'features' found in {feat_path_test}")

            if 'metadata' in hdf_file:
                metadata = hdf_file["metadata"][:]
                #print(metadata)
                metadata_decoded = [str(item, "utf-8") for item in metadata]

            if 'coords' in hdf_file:
                coords = hdf_file["coords"][:]
                metadata_decoded = [f"Tile_({y},{x})" for y, x in coords]

        save_patient_path = os.path.join(save_dir, patient_id)
            
        manipulator.manipulate_patients_images(
                            patient_name=patient_id, 
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
