import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


if __name__ == "__main__":

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
            
    combined_list = [(name, "Lung_adenocarcinoma") for name in adeno_list] + [(name, "Lung_squamous_cell_carcinoma") for name in sq_list]

    df = pd.DataFrame(combined_list, columns=["PATIENT", "Type"])
    df.to_csv(f"{ws_path}/mopadi/datasets/lung/clini_table.csv", index=False)
