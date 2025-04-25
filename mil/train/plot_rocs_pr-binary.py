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


target_label = "2021WHO"
out_dir = f"{ws_path}/mopadi/checkpoints/brain/brain-full/{target_label}/full_model"
csv_file = os.path.join(out_dir, 'PMA_mil_preds_test.csv')
positive_lavel = "A4_IDH"

y_trues = []
y_scores = []

df = pd.read_csv(csv_file)

class_data = []

unique_classes = df[target_label].unique()
print(unique_classes)

positive_class = 1 
pred_column = "preds"

if pred_column not in df.columns:
    raise ValueError(f"Prediction column '{pred_column}' not found in the DataFrame.")

y_trues = [df[target_label] == positive_class]
y_preds = [pd.to_numeric(df[pred_column], errors='coerce').values]

class_labels = [positive_lavel]

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
    title=f"{target_label}",
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
    title=f"{target_label}",
    n_bootstrap_samples=1000
)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "roc.png"), dpi=300)
fig.savefig(os.path.join(out_dir, "roc.svg"))
plt.show()
plt.close()