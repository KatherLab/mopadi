from configs.templates import *
from configs.templates_latent import *
from configs.templates_cls import *
from experiment import *
from exp_linear_cls import *

if __name__ == '__main__':
    # train the autoenc model
    # NOTE: this requires 8 (40Gb) or 4 (80Gb) x A100
    # Run first sbatch run_texture100k-hpc.py, and only after this script
    # gpus = [0, 1, 2, 3]
    # nodes = 2   # if training on the cluster
    # conf = texture100k_autoenc()
    # train(conf, gpus=gpus, nodes=nodes)

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    # gpus = [0, 1, 2, 3]
    # conf = texture100k_autoenc()
    # conf.eval_programs = ['infer']
    # train(conf, gpus=gpus, mode='eval')

    # the rest can be trained locally
    # train the latent DPM
    gpus = [0]
    conf = texture100k_autoenc_latent()
    train(conf, gpus=gpus)

    # train the linear classifier
    gpus = [0]
    conf = texture100k_autoenc_cls()
    train_cls(conf, gpus=gpus)
