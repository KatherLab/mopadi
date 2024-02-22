from configs.templates import *
from configs.templates_latent import *
from configs.templates_cls import *
from train_diff_autoenc import *
from train_linear_cls import *

if __name__ == '__main__':
    # train the autoenc moodel
    # NOTE: this requires 8 (40Gb) or 4 (80Gb) x A100
    gpus = [0, 1, 2, 3]
    conf = tcga_crc_autoenc()
    train(conf, gpus=gpus)

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    print("Infering the latents...")
    gpus = [0, 1, 2, 3]
    conf = tcga_crc_autoenc()
    conf.eval_programs = ['infer']
    train(conf, gpus=gpus, mode='eval')

    # the rest can be trained locally (run_tcga_crc.py)
    # train the latent DPM
    # gpus = [0]
    # conf = tcga_crc_autoenc_latent()
    # train(conf, gpus=gpus)

    # train linear latent classifier
    # gpus = [0]
    # conf = tcga_crc_autoenc_cls()
    # train_cls(conf, gpus=gpus)
