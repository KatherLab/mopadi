import os
import pandas as pd
import shutil
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


def main(table_dir, target, main_directory, copy_directory, nans_directory=None):
    df = pd.read_excel(table_dir)

    if not os.path.exists(copy_directory):
        os.mkdir(copy_directory)

    classes = df[target].unique()
    print(f"Found these classes: {classes}")

    for subdir in os.listdir(main_directory):
        subdir_path = os.path.join(main_directory, subdir)
        patient_id = ("-").join(subdir.split("-")[:3])
        print(patient_id)
        if os.path.isdir(subdir_path):
            if patient_id in df["PATIENT"].values:
                value = df.loc[df["PATIENT"] == patient_id, target].iloc[0]
                if pd.isna(value):
                    if nans_directory is not None:
                        print(f"{patient_id} has value {value}, tiles copied from {subdir_path} to: {os.path.join(nans_directory, subdir)}")
                        shutil.copytree(subdir_path, os.path.join(nans_directory, subdir))
                else:
                    print(f"{patient_id} has value {value}, tiles copied from {subdir_path} to: {os.path.join(copy_directory, subdir)}")
                    shutil.copytree(subdir_path, os.path.join(copy_directory, subdir))


if __name__ == "__main__":

    table_dir = f"{ws_path}/data/TCGA-CRC/clini-tables/TCGA-CRC-DX_CLINI.xlsx"

    # TCGA CRC MSI ----------------------------------------------------------------------------------
    # main_directory = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-only-tumor-tiles-val"
    # msi_directory = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-only-tumor-tiles-val-msi"
    # nans_directory = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-only-tumor-tiles-val-nans-for-msi"

    main_directory = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-only-tumor"
    copy_directory = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-only-tumor-tiles-msi"
    # nans_directory = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-only-tumor-tiles-nans-for-msi"
    target = "isMSIH"

    # TCGA CRC BRAF ---------------------------------------------------------------------------------
    main_directory = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-only-tumor/TCGA-CRC-only-tumor-tiles"
    copy_directory = f"{ws_path}/data/TCGA-CRC/CGA-CRC-only-tumor-BRAF/TCGA-CRC-only-tumor-BRAF-all"

    target = "BRAF"

    main(table_dir, target, main_directory, copy_directory)
