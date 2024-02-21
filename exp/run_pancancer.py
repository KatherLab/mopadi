from configs.templates import *
from configs.templates_latent import *
from configs.templates_cls import *
from experiment import *
from exp_linear_cls import *

if __name__ == '__main__':
    # NOTE: this requires 8 (40Gb) or 4 (80Gb) x A100
    # Run first 'sbatch run_tcga_crc-hpc.py', and only after this script
    # train the autoenc model
    # gpus = [0, 1, 2, 3]
    # conf = pancancer_autoenc()
    # train(conf, gpus=gpus)

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    # print("Infering the latents...")
    # gpus = [0]
    # conf = pancancer_autoenc()
    # conf.eval_programs = ['infer']
    # train(conf, gpus=gpus, mode='eval')

    # the rest can be trained locally
    # train the latent DPM
    # gpus = [0]
    # conf = pancancer_autoenc_latent()
    # train(conf, gpus=gpus)

    # train linear latent classifier
    gpus = [0]
    conf = lung_subtypes_autoenc_cls()
    train_cls(conf, gpus=gpus)
