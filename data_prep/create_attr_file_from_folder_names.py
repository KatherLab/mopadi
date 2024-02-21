import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


if __name__ == '__main__':

    # base_dir = f'{ws_path}/data/brain'
    base_dir = f'{ws_path}/data/japan/val'
    os.listdir(base_dir)

    classes = [cls for cls in os.listdir(base_dir) if not cls.startswith('.')]
    print(f"Found classes: {classes}")

    zfill = 5
    # save_dir = os.path.join(f'{ws_path}/mopadi/datasets', 'brain', 'brain_anno-GBM-IDHmut-new')
    save_dir = os.path.join(f'{ws_path}/mopadi/datasets', 'japan', 'japan_anno_val')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(os.path.join(save_dir, 'list_attr.txt'), 'w') as f:
        f.write('99999' + '\n')
        f.write(' '.join(classes) + '\n')
        i = 0

        # Iterate through the folders (classes) and their contents (images)
        for cls in classes:
            folder_path = os.path.join(base_dir, cls)
            
            for img_name in os.listdir(folder_path):
                # Create an entry with the image name
                key = f'{str(i).zfill(zfill)}.tif'
                i=i+1
                entry = [key]
                
                # Set the corresponding class label to 1 and others to -1
                for c in classes:
                    entry.append(str(1 if c == cls else 0))
                
                # Write the entry to the attribute file
                f.write(' '.join(entry) + '\n')
