import os
from sklearn.model_selection import KFold, GroupKFold
from pathlib import Path
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict
import argparse
import json
import pickle

from mopadi.mil.utils import *


def run_train(conf):
    test_only = False

    required_keys = ['feat_path', 'feat_path_test', 'out_dir', 'clini_table', 'target_label']
    missing = [key for key in required_keys if key not in vars(conf)]
    assert not missing, f"Missing required keys in conf: {missing}"

    if not test_only:
        out_dir = os.path.join(conf.out_dir, 'full_model')
        if not os.path.exists(out_dir):
            Path(out_dir).mkdir(parents=True, exist_ok=True)

        if conf.clini_table.endswith(".tsv"):
            clini_df = pd.read_csv(conf.clini_table,sep="\t")
        elif conf.clini_table.endswith(".xlsx"):
            clini_df = pd.read_excel(conf.clini_table)
        else:    
            clini_df = pd.read_csv(conf.clini_table)

        if conf.target_dict is not None:
            classes = conf.target_dict.values()
        else:
            classes = sorted(clini_df[conf.target_label].dropna().unique())
            conf.target_dict = {label: idx for idx, label in enumerate(classes)}
            print(f"Auto-generated target_dict: {conf.target_dict}")
        print(f"Classes (N = {len(classes)}): {np.array(list(classes))}")

        with open(os.path.join(out_dir, "parameters.json"), 'w') as config_file:
            json.dump(vars(conf), config_file, indent=4)

        valid_patient_df = clini_df.dropna(subset=[conf.target_label])
        valid_patient_df = valid_patient_df[valid_patient_df[conf.target_label].isin(conf.target_dict.keys())]
        valid_patient_ids = valid_patient_df['PATIENT'].unique()
        assert len(valid_patient_ids) != 0, "No patients with valid values found..."

        train_files = np.random.RandomState(seed=42).permutation([os.path.join(conf.feat_path, f) for f in os.listdir(conf.feat_path)]).tolist()
        test_files = np.random.RandomState(seed=42).permutation([os.path.join(conf.feat_path_test, f) for f in os.listdir(conf.feat_path_test)]).tolist()

        # filter feature files based on valid patient IDs
        train_files = [f for f in train_files if extract_patient_id(f.split('/')[-1], index=conf.fname_index) in valid_patient_ids]
        test_files = [f for f in test_files if extract_patient_id(f.split('/')[-1], index=conf.fname_index) in valid_patient_ids]

        dataset = FeatDataset(train_files+test_files, conf.clini_table, conf.target_label, conf.target_dict, conf.nr_feats, fname_index=conf.fname_index)
        trainset = FeatDataset(train_files, conf.clini_table, conf.target_label, conf.target_dict, conf.nr_feats, fname_index=conf.fname_index)
        testset = FeatDataset(test_files, conf.clini_table, conf.target_label, conf.target_dict, conf.nr_feats, fname_index=conf.fname_index)

        print(f"Train set positives: {trainset.get_nr_pos()}; negatives: {trainset.get_nr_neg()}")
        print(f"Test set positives: {testset.get_nr_pos()}; negatives: {testset.get_nr_neg()}")
            
        model = Classifier(dim=conf.dim, num_heads=conf.num_heads, num_seeds=conf.num_seeds, num_classes=len(classes))
        positive_weights = compute_class_weight('balanced', classes=np.array(list(classes)), y=trainset.get_targets())

        with open(f"{out_dir}/positive_weights.pkl", 'wb') as f:
            pickle.dump(positive_weights, f)

        split_info = {
            'Train Files': train_files,
            'Test Files': test_files
        }
        df_split_info = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in split_info.items()]))
        df_split_info.to_csv(os.path.join(out_dir, f'data_split.csv'), index=False)

        model, loader_dict = train_full(model=model, 
                                    train_set=trainset,
                                    test_set=testset,
                                    full_dataset=dataset,
                                    out_dir=out_dir,
                                    positive_weights=positive_weights,
                                    conf=conf
                                    )

        test(model, loader_dict, conf.target_label, out_dir, positive_weights)

    else:
        print("Testing only...")
        with open(os.path.join(out_dir, "loader.pkl"), 'rb') as file:
            loader_dict = pickle.load(file)

        with open(os.path.join(out_dir, "positive_weights.pkl"), 'rb') as file:
            positive_weights = pickle.load(file)

        model = load_cls_model(conf, os.path.join(out_dir, "PMA_mil.pth"), device='cuda:0')
        test(model, loader_dict, conf.target_label, os.path.join(out_dir, "new"), positive_weights)
