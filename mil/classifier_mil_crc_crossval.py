import os
from pathlib import Path
from mil.utils import *
from sklearn.model_selection import KFold
from configs.templates_cls import msi_mil
from torch.utils.data.dataset import Subset
from sklearn.utils.class_weight import compute_class_weight
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

feat_path = f"{ws_path}/extracted_features/TCGA-CRC/TCGA-CRC-train-tumour-only-MSI-status-3"
feat_path_val = f"{ws_path}/extracted_features/TCGA-CRC/TCGA-CRC-val-tumour-only-MSI-status-new-2"
out_dir = f"{ws_path}/mopadi/checkpoints/crc-msi-final-ppt"
annot_file = f"{ws_path}/data/TCGA-CRC/clini-tables/TCGA-CRC-DX_CLINI.xlsx"

conf = msi_mil()

nr_folds = 5
num_workers = 8

if not os.path.exists(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

train_files = np.random.RandomState(seed=420).permutation([os.path.join(feat_path, f) for f in os.listdir(feat_path)]).tolist()
test_files = np.random.RandomState(seed=420).permutation([os.path.join(feat_path_val, f) for f in os.listdir(feat_path_val)]).tolist()

train_dataset = FeatDataset(train_files, annot_file, conf.target_label, conf.target_dict, conf.nr_feats)
test_dataset = FeatDataset(test_files, annot_file, conf.target_label, conf.target_dict, conf.nr_feats)
full_dataset = FeatDataset(train_files+test_files, annot_file, conf.target_label, conf.target_dict, conf.nr_feats)

kf = KFold(n_splits=nr_folds, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kf.split(range(len(train_dataset)))):

    out_fold_dir = os.path.join(out_dir, f"fold_{fold}")
    if not os.path.exists(out_fold_dir):
        Path(out_fold_dir).mkdir(parents=True, exist_ok=True)

    train_subset = Subset(train_dataset, train_index)
    val_subset = Subset(train_dataset, val_index)

    print(f"Train set positives: {train_dataset.get_nr_pos(indices=train_subset.indices)}; negatives: {train_dataset.get_nr_neg(indices=train_subset.indices)}")
    print(f"Val set positives: {train_dataset.get_nr_pos(indices=val_subset.indices)}; negatives: {train_dataset.get_nr_neg(indices=val_subset.indices)}")
    print(f"Test set positives: {test_dataset.get_nr_pos()}; negatives: {test_dataset.get_nr_neg()}")

    positive_weights = compute_class_weight('balanced', classes=[0,1], y=train_dataset.get_targets(indices=train_subset.indices))
    print(f"{positive_weights=}")

    model = Classifier(dim=conf.dim, num_heads=conf.num_heads, num_seeds=conf.num_seeds, num_classes=conf.num_classes)

    model, loader_dict = train(model=model, 
                                train_set=train_subset, 
                                val_set=val_subset,
                                test_set=test_dataset,
                                full_dataset=full_dataset,
                                out_dir=out_fold_dir,
                                positive_weights=positive_weights,
                                conf=conf
                                )

    test(model, loader_dict, conf.target_label, out_fold_dir, positive_weights)
