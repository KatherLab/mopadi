import argparse
import yaml
import torch
import os

from train_diff_autoenc import train
from linear_clf.train_linear_cls import train_cls
from configs.templates import default_autoenc
from configs.templates_latent import default_latent
from configs.templates_cls import default_linear_clf

def validate_gpus(gpus):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but GPUs were specified in the config.")
    n_gpus = torch.cuda.device_count()
    print('-----------------------------------')
    print(f"{n_gpus} CUDA device(s) available:")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"User requested GPUs: {gpus}")
    for gpu in gpus:
        if not isinstance(gpu, int) or gpu < 0 or gpu >= n_gpus:
            raise ValueError(
                f"Requested GPU {gpu} is not available! "
                f"Available device indices: 0 to {n_gpus-1}"
            )
    print('-----------------------------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run mopadi with YAML config')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    gpus = config.get('gpus', None)
    # if they are not given in the config, check cuda and use the first one, otherwise exit
    if gpus is None:
        if torch.cuda.is_available():
            gpus = [0]
            print(f"No GPUs specified, using CUDA device 0: {torch.cuda.get_device_name(0)}")
        else:
            raise RuntimeError("No CUDA-capable GPU is available! Exiting.")
    else:
        # make sure it's a list (could be int if user does 'gpus: 0' in YAML)
        if isinstance(gpus, int):
            gpus = [gpus]
        elif not isinstance(gpus, list):
            raise ValueError("gpus in config must be an int or list of ints.")
        validate_gpus(gpus)

    train_config = config.get('train', {})

    if train_config.get('train_type') == 'autoenc':
        # train the autoenc model
        # NOTE: this requires 8 (40Gb) or 4 (80Gb) x A100
        autoenc_conf = default_autoenc(config)
        train(autoenc_conf, gpus=gpus)

    elif train_config.get('train_type') == 'latent_dpm':
        # infer the features for training the latent DPM
        # NOTE: not gpu heavy, but more gpus can be of use!
        autoenc_conf = default_autoenc(config)
        latent_infer_path = os.path.join(autoenc_conf.base_dir, 'latent.pkl')
        if not os.path.exists(latent_infer_path):
            print(f"Infering the features with the autoencoder from {os.path.join(autoenc_conf.base_dir, 'last.ckpt')}...")
            autoenc_conf.eval_programs = ['infer']
            train(autoenc_conf, gpus=gpus, mode='eval')
        
        # train the latent DPM (not computationally heavy)
        print(f"Starting the training of the latent DPM with the features from {latent_infer_path}...")
        latent_dpm_conf = default_latent(config)
        train(latent_dpm_conf, gpus=gpus)

    elif train_config.get('train_type') == 'linear_cls':
        autoenc_conf = default_autoenc(config)
        latent_infer_path = os.path.join(autoenc_conf.base_dir, 'latent.pkl')
        if not os.path.exists(latent_infer_path):
            print(f"Infering the features with the autoencoder from {os.path.join(autoenc_conf.base_dir, 'last.ckpt')}...")
            autoenc_conf.eval_programs = ['infer']
            train(autoenc_conf, gpus=gpus, mode='eval')

        print(f"Starting the training of the linear classifier with the features from {latent_infer_path}...")
        # train the linear classifier (not computationally heavy)
        linear_clf_conf = default_linear_clf(config)
        train_cls(linear_clf_conf, gpus=gpus)

    else:
        raise ValueError("Invalid train_type specified in config")