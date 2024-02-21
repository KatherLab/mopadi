import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


if __name__ == "__main__":

    base_dir = f"{ws_path}/data/lung/val"
    patients_list = os.listdir(base_dir)

    classes = ["Lung_adenocarcinoma", "Lung_squamous_cell_carcinoma"]
    print(f"Found classes: {classes}")

    # ---------------------------------------------------------------------------------
    adeno_list = []
    main_directory = f"{ws_path}/data/lung/all/Lung_adenocarcinoma"
    for dirpath, dirnames, filenames in os.walk(main_directory):
        if dirpath == main_directory:
            continue
        
        for dirname in dirnames:
            subdirectory_path = os.path.join(dirpath, dirname)
            if os.path.dirname(subdirectory_path).strip(os.sep).endswith(main_directory.strip(os.sep)):
                continue
            adeno_list.append(dirname)
    # ----------------------------------------------------------------------------------
    sq_list = []
    main_directory = f"{ws_path}/data/lung/all/Lung_squamous_cell_carcinoma"
    for dirpath, dirnames, filenames in os.walk(main_directory):
        if dirpath == main_directory:
            continue
        
        for dirname in dirnames:
            subdirectory_path = os.path.join(dirpath, dirname)
            if os.path.dirname(subdirectory_path).strip(os.sep).endswith(main_directory.strip(os.sep)):
                continue
            sq_list.append(dirname)
    # -----------------------------------------------------------------------------------

    zfill = 5
    save_dir = os.path.join(f"{ws_path}/mopadi/datasets", "lung", "lung_anno_val")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(os.path.join(save_dir, "list_attr.txt"), "w") as f:
        f.write("99999" + "\n")
        f.write(" ".join(classes) + "\n")
        i = 0

        # Iterate through the folders (classes) and their contents (images)      
        for patient_name in patients_list:
            patient_path = os.path.join(base_dir, patient_name)
            img_list = [i for i in sorted(os.listdir(patient_path)) if not i.startswith(".")]
            for i, path in enumerate(img_list):
                # Create an entry with the image name
                key = f"{patient_name}/{path}"
                i=i+1
                entry = [key]
                
                # Set the corresponding class label to 1 and others to 0
                for c in [adeno_list, sq_list]:
                    entry.append(str(1 if patient_name in c else 0))
                
                # Write the entry to the attribute file
                f.write(" ".join(entry) + "\n")
