import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


def check_patients_in_clini_table(base_dir, clini_table_path):

    df = pd.read_excel(clini_table_path)
    subdirs = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]

    # simplify the subdirectory names to match the format in the dataframe
    ssubdirs = ["-".join(name.split("-")[:3]) for name in subdirs]
    df_patients = df["PATIENT"].unique().tolist()
    missing_from_df = set(ssubdirs) - set(df_patients)

    if missing_from_df:
        print("Subdirectories missing from dataframe:")
        for item in missing_from_df:
            print(item)
    else:
        print("All subdirectories have a corresponding entry in the dataframe.")

def check_how_many_entries(attr_path):

    with open(attr_path, "r") as f:
        # Skip the header line
        f.readline()

        # Read the second line for class names
        classes_line = f.readline().strip()
        classes = classes_line.split()

        class_counts = {cls: 0 for cls in classes}

        for line in f:

            values = line.strip().split()
            _, class_values = values[0], values[1:]

            for cls, val in zip(classes, class_values):
                if int(val) == 1:
                    class_counts[cls] += int(val)

    print(class_counts)


def main(table_dir, target, tiles_dir, output_dir, out_fname, add_unknown_class = False, zfill = 5):

    df = pd.read_excel(table_dir)
    
    classes = df[target].unique()
    print(f"Found these classes: {classes}. Nan class will be removed.")
    print(df[target].value_counts())

    classes = [cls for cls in classes if not pd.isna(cls)]
    if add_unknown_class:
        classes.append("unknown")

    check_patients_in_clini_table(tiles_dir, table_dir)


    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(os.path.join(output_dir, out_fname), "w") as f:

        f.write("99999" + "\n")
        f.write(" ".join(classes) + "\n")
        i = 0

        for subdir in os.listdir(tiles_dir):
            subdir_path = os.path.join(tiles_dir, subdir)
            
            if os.path.isdir(subdir_path): 

                patient_name = "-".join(subdir.split("-")[:3])
                matching_rows = df[df["PATIENT"] == patient_name]
                if matching_rows.empty:
                    print(f"Warning: No matching row found for patient {patient_name}. Skipping entire folder...")
                    continue

                cls = matching_rows.iloc[0][target]
                
                if add_unknown_class:
                    if pd.isna(cls):
                        cls = "unknown"

                for img_name in os.listdir(subdir_path):
                    key = f"{str(i).zfill(zfill)}.tif"
                    i += 1
                    entry=[key]

                    entry.extend(["1" if c == cls else "-1" for c in classes])

                    if all([val == "-1" for val in entry[1:]]):
                        print(f"Error with image: {img_name} in folder {subdir}. Extracted class: {cls}. Entry: {' '.join(entry)}")

                    f.write(" ".join(entry) + "\n")

        print(f"Finished creating attr file at {os.path.join(output_dir, out_fname)}")


if __name__ == "__main__":

    # TCGA CRC MSI---------------------------------------------------------------------------

    tiles_dir = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-MSI/TCGA-CRC-only-tumor-tiles-msi-train"
    table_dir = f"{ws_path}/data/TCGA-CRC/clini-tables/TCGA-CRC-DX_CLINI.xlsx"
    output_dir = f"{ws_path}/mopadi/datasets/tcga/tcga_crc_anno_only_msi"
    out_fname = "list_attr_tcga_crc_224x224-msi.txt"
    target = "isMSIH"
    add_unknown_class = False

    # main(table_dir, target, tiles_dir, output_dir, out_fname, add_unknown_class)
    # check_how_many_entries(os.path.join(output_dir, out_fname))
    check_how_many_entries(f"{ws_path}/mopadi/datasets/old/list_attr_tcga_crc_224x224-msi.txt")


    # TCGA CRC BRAF -------------------------------------------------------------------------
    """
    tiles_dir = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-only-tumor-BRAF/TCGA-CRC-only-tumor-BRAF-train"
    table_dir = f"{ws_path}/data/TCGA-CRC/clini-tables/TCGA-CRC-DX_CLINI.xlsx"
    output_dir = f"{ws_path}/mopadi/datasets/tcga/tcga_crc_anno_only_braf"
    out_fname = "list_attr_tcga_crc_224x224-braf.txt"
    target = "BRAF"
    add_unknown_class = False

    main(table_dir, target, tiles_dir, output_dir, out_fname)
    """
