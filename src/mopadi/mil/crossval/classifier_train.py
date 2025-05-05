import os
from pathlib import Path
from sklearn.model_selection import KFold, GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data.dataset import Subset
import matplotlib.pyplot as plt
from wanshi.visualizations.roc import plot_multiple_decorated_roc_curves
from collections import defaultdict
import argparse
import json

from mopadi.mil.utils import *
import mopadi.configs.templates_cls as configs


def run_crossval(conf):

    required_keys = ['feat_path', 'feat_path_test', 'out_dir', 'clini_table', 'target_label', 'target_dict']
    conf_dict = vars(conf)
    missing = [key for key in required_keys if key not in conf_dict]
    assert not missing, f"Missing required keys in conf: {missing}"

    out_dir = os.path.join(conf.out_dir, 'crossval')
    if not os.path.exists(out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(out_dir, "parameters.json"), 'w') as config_file:
        json.dump(conf_dict, config_file, indent=4)

    if conf.clini_table.endswith(".tsv"):
        clini_df = pd.read_csv(conf.clini_table,sep="\t")
    elif conf.clini_table.endswith(".xlsx"):
        clini_df = pd.read_excel(conf.clini_table)
    else:    
        clini_df = pd.read_csv(conf.clini_table)

    valid_patient_df = clini_df.dropna(subset=[conf.target_label])
    valid_patient_df = valid_patient_df[valid_patient_df[conf.target_label].isin(conf.target_dict.keys())]
    valid_patient_ids = valid_patient_df['PATIENT'].unique()
    assert len(valid_patient_ids) != 0, "No patients with valid values found..."
    #print(valid_patient_ids)

    train_files = np.random.RandomState(seed=420).permutation([os.path.join(conf.feat_path, f) for f in os.listdir(conf.feat_path)]).tolist()
    test_files = np.random.RandomState(seed=420).permutation([os.path.join(conf.feat_path_test, f) for f in os.listdir(conf.feat_path_test)]).tolist()

    # filter feature files based on valid patient IDs
    train_files = [f for f in train_files if extract_patient_id(f.split('/')[-1], index=conf.fname_index) in valid_patient_ids]
    test_files = [f for f in test_files if extract_patient_id(f.split('/')[-1], index=conf.fname_index) in valid_patient_ids]

    # group files by patient id to ensure that one patient's imgs are only in train or test
    patient_files = defaultdict(list)
    for f in train_files:
        patient_id = extract_patient_id(f.split('/')[-1], index=conf.fname_index)
        patient_files[patient_id].append(f)

    all_train_files = []
    groups = []
    for patient_id, files in patient_files.items():
        all_train_files.extend(files)
        groups.extend([patient_id] * len(files))

    print(f"Number of training files: {len(all_train_files)}")
    print(f"Number of unique patient IDs: {len(set(groups))}")
    assert len(all_train_files) is not None, "No training files found..."
    assert len(all_train_files) == len(groups), "Mismatch between number of training files and group labels"

    if conf.target_dict is not None:
        classes = conf.target_dict.values()
    else:
        classes = clini_df[conf.target_label].unique()

    print(f"Running {conf.nr_folds}-fold cross-validation...")
    print(f"Classes: {np.array(list(classes))}")
    group_kf = GroupKFold(n_splits=conf.nr_folds)

    train_val_dataset = FeatDataset(feat_list=all_train_files, annot_file=conf.clini_table, target_label=conf.target_label, target_dict=conf.target_dict, fname_index=conf.fname_index)
    test_dataset = FeatDataset(feat_list=test_files, annot_file=conf.clini_table, target_label=conf.target_label, target_dict=conf.target_dict, fname_index=conf.fname_index)
    full_dataset = FeatDataset(feat_list=all_train_files+test_files, annot_file=conf.clini_table, target_label=conf.target_label, target_dict=conf.target_dict, fname_index=conf.fname_index)

    for fold, (train_index, val_index) in enumerate(group_kf.split(X=all_train_files, groups=groups)):
        out_fold_dir = os.path.join(out_dir, f"fold_{fold}")

        model_path = os.path.join(out_fold_dir, "PMA_mil.pth")

        if os.path.exists(model_path):
            print(f"Skipping fold {fold} as model already exists at {model_path}")

        if not os.path.exists(out_fold_dir):
            Path(out_fold_dir).mkdir(parents=True, exist_ok=True)

        train_patients = [all_train_files[i] for i in train_index]
        val_patients = [all_train_files[i] for i in val_index]

        train_dataset = FeatDataset(train_patients, conf.clini_table, conf.target_label, conf.target_dict, conf.nr_feats, shuffle=True, fname_index=conf.fname_index)
        val_dataset = FeatDataset(val_patients, conf.clini_table, conf.target_label, conf.target_dict, shuffle=False, fname_index=conf.fname_index)

        test_patients = list(set([extract_patient_id(f.split('/')[-1]) for f in test_files]))

        patient_split_info = {
            "train_patients": train_patients,
            "val_patients": val_patients,
            "test_patients": test_patients
        }

        with open(os.path.join(out_fold_dir, "patients_split.json"), 'w') as pat_split_file:
            json.dump(patient_split_info, pat_split_file, indent=4)

        print(f"Train set positives: {train_dataset.get_nr_pos(indices=train_dataset.indices)}; negatives: {train_dataset.get_nr_neg(indices=train_dataset.indices)}")
        print(f"Val set positives: {train_dataset.get_nr_pos(indices=val_dataset.indices)}; negatives: {train_dataset.get_nr_neg(indices=val_dataset.indices)}")
        print(f"Test set positives: {test_dataset.get_nr_pos()}; negatives: {test_dataset.get_nr_neg()}")

        positive_weights = compute_class_weight('balanced', classes=np.array(list(classes)), y=train_dataset.get_targets(indices=train_dataset.indices))
        print(f"{positive_weights=}")

        model = Classifier(dim=conf.dim, num_heads=conf.num_heads, num_seeds=conf.num_seeds, num_classes=len(classes))

        #layer_info = {name: str(module) for name, module in model.named_modules()}

        model, loader_dict = train_mil(model=model, 
                                train_set=train_dataset, 
                                val_set=val_dataset,
                                test_set=test_dataset,
                                full_dataset=full_dataset,
                                out_dir=out_fold_dir,
                                positive_weights=positive_weights,
                                conf=conf
                                )

        test(model, loader_dict, conf.target_label, out_fold_dir, positive_weights)

    csv_files = [f"{out_dir}/fold_{fold}/PMA_mil_preds_test.csv" for fold in range(conf.nr_folds)]

    y_trues = []
    y_scores = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        y_trues.append(df[conf.target_label].values)
        y_scores.append(df['preds'].values)

    fig, ax = plt.subplots(figsize=(10, 8))

    plot_multiple_decorated_roc_curves(
        ax=ax,
        y_trues=y_trues,
        y_scores=y_scores,
        #title=f"{conf.target_label}",
        n_bootstrap_samples=None
    )

    ax.set_xlabel(ax.get_xlabel(), fontsize=22)
    ax.set_ylabel(ax.get_ylabel(), fontsize=22)

    ax.set_title(ax.get_title(), fontsize=20)

    ax.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), prop={'size': 22})
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "roc.svg"))
    plt.show()
