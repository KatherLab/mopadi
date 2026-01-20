import os
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

from mopadi.dataset import DefaultTilesDataset
from mopadi.mil.utils import *
from mopadi.configs.templates_cls import *

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

#feats_test = f"{ws_path}/features/TCGA-BRCA/test-only-tumor-new"
feats_test = f"{ws_path}/features/TCGA-CRC/test-only-tumor"

mil_dir = f"{ws_path}/mopadi/checkpoints/crc-paper/isMSIH/full_model-newfeats/PMA_mil.pth"
#mil_dir = f"{ws_path}/mopadi/checkpoints/brca/{target_label}/full_model-newfeats/PMA_mil.pth"

tiles_save_path = f"{ws_path}/mopadi/checkpoints/crc-paper/mil_classifier_isMSIH/MSIL_top_tiles"
#tiles_save_path = f"{ws_path}/mopadi/checkpoints/crc/{target_label}/top_tiles"
#tiles_save_path = f"{ws_path}/mopadi/checkpoints/brca/{target_label}/top_tiles"

images_dir = f"{ws_path}/data/TCGA-CRC/512x512_tumor_test"
#images_dir = f"{ws_path}/data/TCGA-BRCA/tiles_512x512_only_tumor-test"

#clini_table=f"{ws_path}/data/TCGA-BRCA/tables/all_df_E2_center.csv"
clini_table = f"{ws_path}/data/TCGA-CRC/clini-tables/TCGA-CRC-DX_CLINI.xlsx"

target_dict = {"nonMSIH": 0, "MSIH": 1}
#target_dict={"No": 0, "Yes": 1}
#target_dict={"IDC": 0, "ILC": 1}
conf = crc_pretrained_mil()
#conf = brca_mil()
n_top_tiles=5

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

data = DefaultTilesDataset(root_dirs = [images_dir], 
                           img_size = 512)

for path in tqdm(test_files):

    with h5py.File(path, "r") as hdf_file:
        patient_features = torch.from_numpy(hdf_file["feats"][:])

        if 'metadata' in hdf_file:
            metadata = hdf_file["metadata"][:]
            metadata_decoded = [str(item, "utf-8") for item in metadata]

        if 'coords' in hdf_file:
            coords = hdf_file["coords"][:]
            metadata_decoded = [f"Tile_({y},{x})" for y, x in coords]
    
    patient_name = "-".join(os.path.basename(path).split(".h5")[0].split("-")[:3])

    # do it only for MSIL patients
    if patient_name not in ['TCGA-D5-6533', 'TCGA-AY-5543', 'TCGA-D5-5541', 'TCGA-G4-6295', 'TCGA-AY-A54L', 'TCGA-EI-6509', 'TCGA-D5-6931', 'TCGA-CM-6169', 'TCGA-NH-A50T', 'TCGA-F4-6459', 'TCGA-D5-5538', 'TCGA-CA-6716', 'TCGA-G4-6317', 'TCGA-AA-A02E', 'TCGA-AA-3837', 'TCGA-F5-6571', 'TCGA-D5-5540', 'TCGA-DC-4749', 'TCGA-AA-3679', 'TCGA-D5-5539', 'TCGA-AA-3696', 'TCGA-A6-5662', 'TCGA-AF-2687', 'TCGA-AZ-4315', 'TCGA-AY-4071', 'TCGA-D5-6928', 'TCGA-CA-5796', 'TCGA-A6-6142', 'TCGA-AF-A56K', 'TCGA-A6-5660', 'TCGA-AY-A8YK', 'TCGA-AA-A024', 'TCGA-D5-6930', 'TCGA-AY-A71X', 'TCGA-AD-6965', 'TCGA-QL-A97D', 'TCGA-DC-6158', 'TCGA-F4-6854', 'TCGA-D5-6541', 'TCGA-AD-A5EK', 'TCGA-NH-A6GC', 'TCGA-AY-A69D', 'TCGA-D5-6535', 'TCGA-EI-6881', 'TCGA-CM-5862', 'TCGA-D5-6531', 'TCGA-D5-6532', 'TCGA-D5-6536', 'TCGA-D5-6898', 'TCGA-AA-3693', 'TCGA-CA-5797', 'TCGA-AY-4070', 'TCGA-D5-6538']:
        continue
    print("Getting top tiles for patient:", patient_name)

    patient_class = clini_df.loc[clini_df["PATIENT"] == ("-").join(patient_name.split(".")[0].split("-")[:3]), target_label].iloc[0]
    not_gt_cls_id = None
    if len(target_dict) == 2:
        # Binary case (e.g., {"MSI-H": 1, "nonMSIH": 0} or {"MSI-H":1, "MSS":0})
        not_gt_cls_id = [lbl for lbl, cid in target_dict.items() if lbl != patient_class][0]

    if not patient_class in target_dict.keys():
        continue

    top_indices, top_scores = get_top_tiles(model=cls_model, feats=patient_features, k=n_top_tiles, cls_id=target_dict[patient_class], device="cuda:0")
    if not_gt_cls_id is not None:
        top_indices_notgtclass, top_scores_notgtclass = get_top_tiles(model=cls_model, feats=patient_features, k=n_top_tiles, cls_id=target_dict[not_gt_cls_id], device="cuda:0")
    
    if top_indices is None:
        print(f"Skipping {patient_name}")
        continue

    for top_idx, score in tqdm(zip(top_indices, top_scores), total=len(top_indices.cpu().tolist()), desc='Saving top tiles'):
        
        score_str = f"{score.item():.4f}"
        fname = metadata_decoded[top_idx.item()]

        patient_details = data.get_images_by_patient_and_fname(patient_name, fname)
        #print(patient_details)

        save_dir = os.path.join(tiles_save_path, patient_name, "GT_" + patient_class)
        if not os.path.exists(save_dir):
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        if patient_details:
            img_rgb = convert2rgb(patient_details["image"][None])
            save_image(img_rgb, os.path.join(save_dir, f"{fname}_{score_str}_{patient_class}.png"))

    if not_gt_cls_id is not None:
            for top_idx, score in tqdm(zip(top_indices_notgtclass, top_scores_notgtclass), total=len(top_indices_notgtclass.cpu().tolist()), desc='Saving top tiles for not GT class'):
        
                score_str = f"{score.item():.4f}"
                fname = metadata_decoded[top_idx.item()]

                patient_details = data.get_images_by_patient_and_fname(patient_name, fname)
                #print(patient_details)

                save_dir = os.path.join(tiles_save_path, patient_name, not_gt_cls_id)
                if not os.path.exists(save_dir):
                    Path(save_dir).mkdir(parents=True, exist_ok=True)

                if patient_details:
                    img_rgb = convert2rgb(patient_details["image"][None])
                    save_image(img_rgb, os.path.join(save_dir, f"{fname}_{score_str}_{not_gt_cls_id}.png"))
