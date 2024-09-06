# %%
import os
from mil.utils import *
from pathlib import Path
from configs.templates_cls import msi_mil
from dotenv import load_dotenv
from sklearn.utils.class_weight import compute_class_weight

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")
# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
feat_path = f"{ws_path}/extracted_features/TCGA-CRC/TCGA-CRC-train-tumour-only-MSI-status-3"
feat_path_val = f"{ws_path}/extracted_features/TCGA-CRC/TCGA-CRC-val-tumour-only-MSI-status-new-2"
out_dir = f"{ws_path}/mopadi/checkpoints/msi-final-ppt-512"
annot_file = f"{ws_path}/data/TCGA-CRC/clini-tables/TCGA-CRC-DX_CLINI.xlsx"

conf = msi_mil()

if not os.path.exists(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
# %%
feat_files = np.random.RandomState(seed=420).permutation([os.path.join(feat_path, f) for f in os.listdir(feat_path)]).tolist()
test_files = np.random.RandomState(seed=420).permutation([os.path.join(feat_path_val, f) for f in os.listdir(feat_path_val)]).tolist()

nr_val = int(len(feat_files) * 0.2)
val_files = feat_files[:nr_val]
tr_files = feat_files[nr_val:]

print(f"{len(tr_files)=}")
print(f"{len(val_files)=}")
print(f"{len(test_files)=}")

dataset = FeatDataset(feat_files+test_files, annot_file, conf.target_label, conf.target_dict, conf.nr_feats)
trainset = FeatDataset(tr_files, annot_file, conf.target_label, conf.target_dict, conf.nr_feats)
valset = FeatDataset(val_files, annot_file, conf.target_label, conf.target_dict, conf.nr_feats)
testset = FeatDataset(test_files, annot_file, conf.target_label, conf.target_dict, conf.nr_feats)
    
# %%
model = Classifier(dim=conf.dim, num_heads=conf.num_heads, num_seeds=conf.num_seeds, num_classes=conf.num_classes)
positive_weights = compute_class_weight('balanced', classes=[0,1], y=trainset.get_targets())
# %%
model, loader_dict = train(model=model, 
                           train_set=trainset,
                           val_set=valset,
                           test_set=testset,
                           full_dataset=dataset,
                           out_dir=out_dir,
                           positive_weights=positive_weights,
                           conf=conf
                           )

test(model, loader_dict, conf.target_label, out_dir, positive_weights)
# %%
