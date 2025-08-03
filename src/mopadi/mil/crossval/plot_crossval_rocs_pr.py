import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from wanshi.visualizations.roc import plot_multiple_decorated_roc_curves, plot_multiple_decorated_pr_curves
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")

targets = [
  "BRCA1_binary",
  "BRCA2_binary",
  "TP53_binary",
  "PIK3CA_binary",
  "ERBB2_binary",
  "GATA3_binary",
  "AKT1_binary",
  "PTEN_binary",
  "CDH1_binary",
  "ESR1_binary",
  "RB1_binary",
  "NOTCH1_binary",
  "BCL2_binary",
  "AR_binary",
  "MYC_binary",
  "KRAS_binary",
  "SMARCA4_binary",
  "SMARCB1_binary",
]

#targets_new = ["RB1_binary"]
#target_label = "BRCA_Pathology"
#target_label = "isMSIH"
#target_label = "is_E2"
target_label = "Type"

for exp in range(1):

    #out_dir = f"{ws_path}/mopadi/checkpoints/pancancer/Type/crossval/{exp}"
    out_dir = f"{ws_path}/mopadi/checkpoints/pancancer/mil_liver/{target_label}/crossval-silu-final"
    #out_dir = f"{ws_path}/mopadi/checkpoints/brca/{target_label}/crossval-final-newfeats"
    #out_dir = f"{ws_path}/mopadi/checkpoints/crc/{target_label}/crossval-final-newfeats"
    #out_dir = f"{ws_path}/mopadi/checkpoints/pancancer/mil_lung/{target_label}/crossval"
    nr_folds = 5

    csv_files = [f"{out_dir}/fold_{fold}/PMA_mil_preds_test.csv" for fold in range(nr_folds)]

    y_trues = []
    y_scores = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        y_trues.append(df[target_label].values)
        y_scores.append(df['preds'].values)

    roc_curve_figure_aspect_ratio = 1.08
    fig, ax = plt.subplots(
        figsize=(3.8, 3.8 * roc_curve_figure_aspect_ratio),
        dpi=300,
    )
    plot_multiple_decorated_pr_curves(
        ax=ax,
        y_trues=y_trues,
        y_scores=y_scores,
        #title=f"{target_label}",
        n_bootstrap_samples=None
    )

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "pr.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "pr.svg"), bbox_inches='tight')
    plt.show()
    plt.close()

    fig, ax = plt.subplots(
        figsize=(3.8, 3.8 * roc_curve_figure_aspect_ratio),
        dpi=300,
    )

    plot_multiple_decorated_roc_curves(
        ax=ax,
        y_trues=y_trues,
        y_scores=y_scores,
        #title=f"{target_label}",
        n_bootstrap_samples=None
    )

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "roc.svg"), bbox_inches='tight')
    plt.show()

