from configs.templates import *
from configs.templates_latent import *
from configs.templates_cls import *
from experiment_classifier import *

if __name__ == '__main__':
    # NOTE: this requires 8 x V100s / 8 or 4 x A100
    # Run first 'sbatch run_tcga_brca_512-hpc.py', and only after this script
    # train the autoenc model
    gpus = [0, 1, 2, 3]
    conf = tcga_brca_512_autoenc()
    train(conf, gpus=gpus)

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    print("Infering the latents...")
    gpus = [0, 1, 2, 3]
    conf = tcga_brca_512_autoenc()
    conf.eval_programs = ['infer']
    train(conf, gpus=gpus, mode='eval')

    # the rest can be trained locally
    # train the latent DPM
    gpus = [0]
    conf = tcga_brca_512_autoenc_latent()
    train(conf, gpus=gpus)
