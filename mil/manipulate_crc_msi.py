import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torchvision import transforms
from configs.templates import tcga_crc_autoenc
from configs.templates_cls import msi_mil
from dataset import TCGADataset

from mil.manipulator_mil import ImageManipulator
import pandas as pd
from tqdm import tqdm
import torch
import h5py
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


if __name__=="__main__":

    images_dir = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-MSI/TCGA-CRC-only-tumor-tiles-msi-val"
    autoenc_path = "checkpoints/tcga_crc_224x224/last.ckpt"
    mil_path = "checkpoints/msi-final-ppt-512/PMA_mil.pth"
    latent_infer_path = "checkpoints/tcga_crc_224x224/latent.pkl"
    save_dir = os.path.join(f"{ws_path}", "results-manipulated", "msi-final-ppt-512")
    clini_table = f"{ws_path}/data/TCGA-CRC/clini-tables/TCGA-CRC-DX_CLINI.xlsx"
    feat_dir = f"{ws_path}/extracted_features/TCGA-CRC/TCGA-CRC-val-tumour-only-MSI-status-new-2"

    conf = tcga_crc_autoenc()
    conf_cls = msi_mil()

    #man_amps = [0.2, 0.4, 0.6, 0.8, 1.0]
    man_amps = [0.4, 0.8, 1.2, 1.6, 2.0]

    data = TCGADataset(images_dir = images_dir, 
                       transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    manipulator = ImageManipulator(autoenc_config = conf,
                                   autoenc_path = autoenc_path, 
                                   mil_path = mil_path, 
                                   latent_infer_path = latent_infer_path,
                                   dataset = data,
                                   conf_cls = conf_cls
                                   )

    clini_df = pd.read_excel(clini_table)

    feat_files = [f for f in os.listdir(feat_dir) if not f.startswith(".")]

    for fname in tqdm(feat_files):
        res = {}
        patient_name = ("-").join(fname.split(".")[0].split("-")[:3])

        patient_class = clini_df.loc[clini_df["PATIENT"] == patient_name, conf_cls.target_label].iloc[0]

        with h5py.File(os.path.join(feat_dir, fname), "r") as hdf_file:
            features = torch.from_numpy(hdf_file["features"][:])
            metadata = hdf_file["metadata"][:]
            metadata_decoded = [str(item, "utf-8") for item in metadata]
            #metadata_decoded = ["({},{})".format(item[0], item[1]) for item in metadata]
        print(features.size())
        #features = (torch.cat((features, torch.zeros(conf_cls.nr_feats - features.shape[0], features.shape[1]))))

        save_patient_path = os.path.join(save_dir, patient_name)
            
        manipulator.manipulate_patients_images(
                            patient_name=patient_name, 
                            patient_features=features.unsqueeze(dim=0),
                            metadata=metadata_decoded,
                            save_path=save_patient_path, 
                            man_amps=man_amps,
                            patient_class=patient_class,
                            target_dict=conf_cls.target_dict,
                            num_top_tiles=2)
