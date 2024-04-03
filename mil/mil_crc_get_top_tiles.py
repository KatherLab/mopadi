import os
from mil.utils import *
from tqdm import tqdm
from pathlib import Path
from configs.templates_cls import msi_mil
from dotenv import load_dotenv
from dataset import TCGADataset

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")

def get_top_tiles(model, feats, k=15, cls_id=1, device='cuda:0'):
    unsq = feats.squeeze(0).unsqueeze(1).to(device)
    scores = F.softmax(model(unsq), dim=1)
    top_scores, top_indices = scores[:, cls_id].topk(k)
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
feat_path_val = f"{ws_path}/extracted_features/TCGA-CRC/TCGA-CRC-val-tumour-only-MSI-status-new-2"
mil_dir = "checkpoints/msi-final-ppt-512/PMA_mil.pth"
tiles_path = f"{ws_path}/top_tiles_MSI-w-cls"
images_dir = f"{ws_path}/data/TCGA-CRC/TCGA-CRC-MSI/TCGA-CRC-only-tumor-tiles-msi-val"
clini_table = f"{ws_path}/data/TCGA-CRC/clini-tables/TCGA-CRC-DX_CLINI.xlsx"

target_dict = {"nonMSIH": 0, "MSIH": 1}

conf = msi_mil()

if not os.path.exists(tiles_path):
    Path(tiles_path).mkdir(parents=True, exist_ok=True)

test_files = [os.path.join(feat_path_val, f) for f in os.listdir(feat_path_val)]
print(f"{len(test_files)=}")
    
cls_model = Classifier(dim=conf.dim, num_heads=conf.num_heads, num_seeds=conf.num_seeds, num_classes=conf.num_classes)
weights = torch.load(mil_dir)
cls_model.load_state_dict(weights)
cls_model = cls_model.to("cuda:0")
cls_model.eval()

clini_df = pd.read_excel(clini_table)

data = TCGADataset(images_dir = images_dir, 
                    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

for path in tqdm(test_files):

    with h5py.File(path, "r") as hdf_file:
        patient_features = torch.from_numpy(hdf_file["features"][:])
        metadata = hdf_file["metadata"][:]
        metadata_decoded = [str(item, "utf-8") for item in metadata]
    
    patient_name = os.path.basename(path).split(".h5")[0]
    patient_class = clini_df.loc[clini_df["PATIENT"] == ("-").join(patient_name.split(".")[0].split("-")[:3]), "isMSIH"].iloc[0]

    if not os.path.exists(os.path.join(tiles_path, patient_name + "_" + patient_class)):
        Path(os.path.join(tiles_path, patient_name + "_" + patient_class)).mkdir(parents=True, exist_ok=True)

    top_indices, top_scores = get_top_tiles(model=cls_model, feats=patient_features, k=30, cls_id=target_dict[patient_class], device="cuda:0")

    for top_idx, score in tqdm(zip(top_indices, top_scores), total=len(top_indices.cpu().tolist()), desc='Saving top tiles'):
        
        score_str = f"{score.item():.4f}"
        fname = metadata_decoded[top_idx.item()]

        patient_details = data.get_images_by_patient_and_fname(patient_name, fname)

        img_rgb = convert2rgb(patient_details["image"][None])
        save_image(img_rgb, os.path.join(tiles_path, patient_name + "_" + patient_class, f"{fname}_{score_str}_{patient_class}.tiff"))
