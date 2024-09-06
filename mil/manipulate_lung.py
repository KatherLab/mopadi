import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torchvision import transforms
from configs.templates import pancancer_autoenc
from configs.templates_cls import lung_mil
from dataset import LungDataset

from mil.manipulator_mil import ImageManipulator
from tqdm import tqdm
import pandas as pd
import torch
import h5py
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


if __name__=="__main__":

    gt_table_dir = "datasets/lung/lung_anno_val/list_attr.txt"
    images_dir = f"{ws_path}/data/lung/val"
    autoenc_path = "checkpoints/pancancer/last.ckpt"
    mil_path = "checkpoints/lung-newest-512/PMA_mil.pth"
    latent_infer_path = "checkpoints/pancancer/latent.pkl"
    save_dir = os.path.join(f"{ws_path}", "results-manipulated", "Lung-subtypes-newest-clf-512-no-zeros")
    clini_table = "datasets/lung/clini_table.csv"
    feat_dir = f"{ws_path}/extracted_features/TCGA-LUAD-LUSC/lung-val"

    conf = pancancer_autoenc()
    conf_cls = lung_mil()
    man_amps = [0.01, 0.02, 0.03]

    data = LungDataset(images_dir=images_dir, 
                       path_to_gt_table=gt_table_dir, 
                       transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    manipulator = ImageManipulator(autoenc_config=conf,
                                   autoenc_path=autoenc_path, 
                                   mil_path=mil_path, 
                                   latent_infer_path=latent_infer_path,
                                   dataset=data,
                                   conf_cls=conf_cls)

    clini_df = pd.read_csv(clini_table)

    feat_files = [f for f in os.listdir(feat_dir) if not f.startswith(".")]

    for fname in tqdm(feat_files):
        res = {}
        patient_name = fname.split(".")[0]

        patient_class = clini_df.loc[clini_df["PATIENT"] == patient_name, conf_cls.target_label].iloc[0]

        with h5py.File(os.path.join(feat_dir, fname), "r") as hdf_file:
            features = torch.from_numpy(hdf_file["features"][:])
            metadata = hdf_file["metadata"][:]
            metadata_decoded = [str(item, "utf-8") for item in metadata]

        save_patient_path = os.path.join(save_dir, patient_name)
            
        manipulator.manipulate_patients_images(
                            patient_name=patient_name, 
                            patient_features=features.unsqueeze(dim=0),
                            metadata=metadata_decoded,
                            save_path=save_patient_path, 
                            man_amps=man_amps,
                            patient_class=patient_class,
                            target_dict=conf_cls.target_dict,
                            num_top_tiles=5
                            )
