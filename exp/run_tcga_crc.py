from configs.templates import *
from configs.templates_cls import *
from train_diff_autoenc import *
from linear_clf.train_linear_cls import *

if __name__ == '__main__':    
    # train the autoenc model
    # NOTE: this requires 8 (40Gb) or 4 (80Gb) x A100
    # Run first 'sbatch run_tcga_crc-hpc.py', and only after this script
    gpus = [0, 1, 2, 3]
    conf = tcga_crc_512_autoenc()
    train(conf, gpus=gpus)
