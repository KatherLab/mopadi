import os
from mil.utils import *
from sklearn.model_selection import KFold, GroupKFold
from pathlib import Path
import configs.templates_cls as configs
from dotenv import load_dotenv
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict
import argparse
import json

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")

def extract_patient_id(filename):
    return "-".join(filename.split('-')[:3])

parser = argparse.ArgumentParser(description="MIL Train")
parser.add_argument('--feat_path', type=str, required=True, help='Path to the training/val feature files')
parser.add_argument('--feat_path_test', type=str, required=True, help='Path to the testing feature files')
parser.add_argument('--out_dir', type=str, required=True, help='Output directory for checkpoints and results')
parser.add_argument('--clini_table', type=str, required=True, help='Path to the clinical table file')
parser.add_argument('--conf_name', type=str, required=True, help='Configuration function name')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
parser.add_argument('--target_label', type=str, required=False, default=None, help='Target label if different from config')
parser.add_argument('--target_dict', type=str, required=False, default=None, help='Target dictionary for the configuration (JSON format) if different')

args = parser.parse_args()

feat_path = args.feat_path
feat_path_test = args.feat_path_test
out_dir = args.out_dir
clini_table = args.clini_table
conf_func_name = args.conf_name
num_workers = args.num_workers
target_label = args.target_label
target_dict = args.target_dict

try:
    target_dict = json.loads(args.target_dict)
    print(f"Parsed target_dict: {target_dict}")
except json.JSONDecodeError as e:
    print(f"Error parsing target_dict: {e}")
    exit(1)

conf_func = getattr(configs, conf_func_name)
conf = conf_func()

if target_label:
    conf.target_label = target_label
if target_dict:
    conf.target_dict = target_dict

config_params = {
    'feat_path': feat_path,
    'feat_path_test': feat_path_test,
    'clini_table': clini_table,
    'conf_name': conf_func_name,
    'config': vars(conf),
}

with open(os.path.join(out_dir, "parameters.json"), 'w') as config_file:
    json.dump(config_params, config_file, indent=4)

print(f"Configuration and parameters saved to {os.path.join(out_dir, 'parameters.json')}")

if not os.path.exists(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

if clini_table.endswith(".tsv"):
    clini_df = pd.read_csv(clini_table,sep="\t")
elif clini_table.endswith(".xlsx"):
    clini_df = pd.read_excel(clini_table)
else:    
    clini_df = pd.read_csv(clini_table)

valid_patient_df = clini_df.dropna(subset=[conf.target_label])
valid_patient_df = valid_patient_df[valid_patient_df[conf.target_label].isin(conf.target_dict.keys())]
valid_patient_ids = valid_patient_df['PATIENT'].unique()
assert len(valid_patient_ids) != 0, "No patients with valid values found..."

train_files = np.random.RandomState(seed=42).permutation([os.path.join(feat_path, f) for f in os.listdir(feat_path)]).tolist()
test_files = np.random.RandomState(seed=42).permutation([os.path.join(feat_path_test, f) for f in os.listdir(feat_path_test)]).tolist()

# filter feature files based on valid patient IDs
train_files = [f for f in train_files if extract_patient_id(f.split('/')[-1]) in valid_patient_ids]
test_files = [f for f in test_files if extract_patient_id(f.split('/')[-1]) in valid_patient_ids]

print(f"{len(train_files)=}")
print(f"{len(test_files)=}")

dataset = FeatDataset(train_files+test_files, clini_table, conf.target_label, conf.target_dict, conf.nr_feats)
trainset = FeatDataset(train_files, clini_table, conf.target_label, conf.target_dict, conf.nr_feats)
testset = FeatDataset(test_files, clini_table, conf.target_label, conf.target_dict, conf.nr_feats)
    
model = Classifier(dim=conf.dim, num_heads=conf.num_heads, num_seeds=conf.num_seeds, num_classes=conf.num_classes)
positive_weights = compute_class_weight('balanced', classes=[0,1], y=trainset.get_targets())

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