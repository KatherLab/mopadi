import os
from mil.utils import *
from tqdm import tqdm
from pathlib import Path
from configs.templates_cls import *
from dotenv import load_dotenv
from dataset import TCGADataset

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")

def get_top_tiles(model, feats, k=5, cls_id=1, device='cuda:0'):
    unsq = feats.squeeze(0).unsqueeze(1).to(device)
    scores = F.softmax(model(unsq), dim=1)
    try:
        top_scores, top_indices = scores[:, cls_id].topk(k)
    except Exception as e:
        print(e)
        return None, None

    return top_indices, top_scores

def save_image(image_tensor, save_path):
    image = transforms.ToPILImage()(image_tensor.cpu().squeeze(0))
    image.save(save_path)

def convert2rgb(img):
    convert_img = img.clone().detach()
    if convert_img.min() < 0:
        # transform pixel values into the [0, 1] range
        convert_img = (convert_img + 1) / 2
    return convert_img.cpu()


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

target_label = "isMSIH"
#target_label = "is_E2"
#target_label = "BRCA_Pathology"

#feats_test = f"{ws_path}/extracted_features/TCGA-BRCA/test-only-tumor-new"
feats_test = f"{ws_path}/extracted_features/TCGA-CRC/test-only-tumor"
mil_dir = f"{ws_path}/mopadi/checkpoints/crc/{target_label}/full_model-newfeats/PMA_mil.pth"
#mil_dir = f"{ws_path}/mopadi/checkpoints/brca/{target_label}/full_model-newfeats/PMA_mil.pth"
tiles_save_path = f"{ws_path}/mopadi/checkpoints/crc/{target_label}/top_tiles"
#tiles_save_path = f"{ws_path}/mopadi/checkpoints/brca/{target_label}/top_tiles"
images_dir = f"{ws_path}/data/TCGA-CRC/512x512_tumor_test"
#images_dir = f"{ws_path}/data/TCGA-BRCA/tiles_512x512_only_tumor-test"
#clini_table=f"{ws_path}/data/TCGA-BRCA/tables/all_df_E2_center.csv"
clini_table = f"{ws_path}/data/TCGA-CRC/clini-tables/TCGA-CRC-DX_CLINI.xlsx"
target_dict = {"nonMSIH": 0, "MSIH": 1}
#target_dict={"No": 0, "Yes": 1}
#target_dict={"IDC": 0, "ILC": 1}
conf = crc_mil()
#conf = brca_mil()
n_top_tiles=6

if not os.path.exists(tiles_save_path):
    Path(tiles_save_path).mkdir(parents=True, exist_ok=True)

test_files = [os.path.join(feats_test, f) for f in os.listdir(feats_test)]
print(f"{len(test_files)=}")
    
cls_model = Classifier(dim=conf.dim, num_heads=conf.num_heads, num_seeds=conf.num_seeds, num_classes=conf.num_classes)
weights = torch.load(mil_dir)
cls_model.load_state_dict(weights)
cls_model = cls_model.to("cuda:0")
cls_model.eval()

if clini_table.endswith(".csv"):
    clini_df = pd.read_csv(clini_table)
elif clini_table.endswith(".xlsx"):
    clini_df = pd.read_excel(clini_table)

data = TCGADataset(images_dir = images_dir, 
                    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

for path in tqdm(test_files):

    with h5py.File(path, "r") as hdf_file:
        patient_features = torch.from_numpy(hdf_file["feats"][:])

        if 'metadata' in hdf_file:
            metadata = hdf_file["metadata"][:]
            metadata_decoded = [str(item, "utf-8") for item in metadata]

        if 'coords' in hdf_file:
            coords = hdf_file["coords"][:]
            metadata_decoded = [f"Tile_({y},{x})" for y, x in coords]
    
    patient_name = os.path.basename(path).split(".h5")[0]
    patient_class = clini_df.loc[clini_df["PATIENT"] == ("-").join(patient_name.split(".")[0].split("-")[:3]), target_label].iloc[0]

    if not patient_class in target_dict.keys():
        continue

    if not os.path.exists(os.path.join(tiles_save_path, patient_name + "_" + str(patient_class))):
        Path(os.path.join(tiles_save_path, patient_name + "_" + str(patient_class))).mkdir(parents=True, exist_ok=True)

    top_indices, top_scores = get_top_tiles(model=cls_model, feats=patient_features, k=n_top_tiles, cls_id=target_dict[patient_class], device="cuda:0")
    
    if top_indices is None:
        print(f"Skipping {patient_name}")
        continue

    for top_idx, score in tqdm(zip(top_indices, top_scores), total=len(top_indices.cpu().tolist()), desc='Saving top tiles'):
        
        score_str = f"{score.item():.4f}"
        fname = metadata_decoded[top_idx.item()]

        patient_details = data.get_images_by_patient_and_fname(patient_name, fname)
        #print(patient_details)

        if patient_details:
            img_rgb = convert2rgb(patient_details["image"][None])
            save_image(img_rgb, os.path.join(tiles_save_path, patient_name + "_" + patient_class, f"{fname}_{score_str}_{patient_class}.png"))
