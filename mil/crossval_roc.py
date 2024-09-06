# %%
import os
import pandas as pd
from configs.templates_cls import msi_mil, lung_mil
import matplotlib.pyplot as plt
from wanshi.visualizations.roc import plot_multiple_decorated_roc_curves
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")
# %%
nr_feats_list = [512]
conf_cls = msi_mil()

for nr_feats in nr_feats_list:

    results_dir = f"{ws_path}/mopadi/checkpoints/crc-msi-final-ppt/crc-msi-mil-{nr_feats}-crossval-test"
    nr_fold = 5

    csv_files = [f"{results_dir}/fold_{fold}/PMA_mil_preds_test.csv" for fold in range(nr_fold)]

    y_trues = []
    y_scores = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        y_trues.append(df[conf_cls.target_label].values)
        y_scores.append(df['preds'].values)

    fig, ax = plt.subplots(figsize=(10, 8))

    plot_multiple_decorated_roc_curves(
        ax=ax,
        y_trues=y_trues,
        y_scores=y_scores,
        title=f"{conf_cls.target_label} (nr_feats={nr_feats})",
        n_bootstrap_samples=None
    )

    ax.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), prop={'size': 14})
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "roc.png"), dpi=300)
    fig.savefig(os.path.join(results_dir, "roc.svg"))
    plt.show()
# %%
