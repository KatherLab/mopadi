import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from wanshi.visualizations.roc import plot_multiple_decorated_roc_curves, plot_multiple_decorated_pr_curves
import numpy as np
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


#target_label = "Type"
#target_label = "is_E2"
target_label = "isMSIH"

#out_dir = f"{ws_path}/mopadi/checkpoints/pancancer/mil_liver/{target_label}/full_model"
#out_dir = f"{ws_path}/mopadi/checkpoints/brca/{target_label}/full_model"
out_dir = f"{ws_path}/mopadi/checkpoints/crc/{target_label}/full_model-newfeats"
csv_file = os.path.join(out_dir, 'PMA_mil_preds_test.csv')

y_trues = []
y_scores = []

df = pd.read_csv(csv_file)

class_data = []

unique_classes = df[target_label].unique()
print(unique_classes)

for i in unique_classes:
    #pred_column = f"{target_label}_{i}"
    pred_column = "preds"
    if pred_column not in df.columns:
        print(f"Warning: Prediction column '{pred_column}' not found for class {i}.")
        continue

    y_true = df[target_label] == i
    if i == 0:
        y_pred = 1 - pd.to_numeric(df[pred_column], errors='coerce')
    else:
        y_pred = pd.to_numeric(df[pred_column], errors='coerce')
    
    if len(y_true) != len(y_pred):
        print(f"Error: Mismatch in lengths for class {i}. True labels: {len(y_true)}, Predictions: {len(y_pred)}")
        continue
    class_data.append((str(i), y_true.values, y_pred.values))

class_labels, y_trues, y_preds = zip(*class_data) if class_data else ([], [], [])

for i, (label, true, pred) in enumerate(zip(class_labels, y_trues, y_preds)):
    print(f"Class {label}: {sum(true)} positive samples, {len(pred)} total predictions")

y_trues = [np.array(y_true) for y_true in y_trues]
y_preds = [np.array(y_pred) for y_pred in y_preds]

roc_curve_figure_aspect_ratio = 1.08
fig, ax = plt.subplots(
    figsize=(3.8, 3.8 * roc_curve_figure_aspect_ratio),
    dpi=300,
)

# plot PR curve
plot_multiple_decorated_pr_curves(
    ax=ax,
    y_trues=y_trues,
    y_scores=y_preds,
    class_labels=class_labels,
    #title=f"{target_label}",
    n_bootstrap_samples=1000
)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "pr.png"), dpi=300)
fig.savefig(os.path.join(out_dir, "pr.svg"))
plt.show()
plt.close()

# plot ROC curve
fig, ax = plt.subplots(
    figsize=(3.8, 3.8 * roc_curve_figure_aspect_ratio),
    dpi=300,
)

plot_multiple_decorated_roc_curves(
    ax=ax,
    y_trues=y_trues,
    y_scores=y_preds,
    class_labels=class_labels,
    #title=f"{target_label}",
    n_bootstrap_samples=1000
)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "roc.png"), dpi=300)
fig.savefig(os.path.join(out_dir, "roc.svg"))
plt.show()
plt.close()