from configs.templates import *


def latent_diffusion_config(conf: TrainConfig):
    conf.batch_size = 128
    conf.train_mode = TrainMode.latent_diffusion
    conf.latent_gen_type = GenerativeType.ddim
    conf.latent_loss_type = LossType.mse
    conf.latent_model_mean_type = ModelMeanType.eps
    conf.latent_model_var_type = ModelVarType.fixed_large
    conf.latent_rescale_timesteps = False
    conf.latent_clip_sample = False
    conf.latent_T_eval = 20
    conf.latent_znormalize = True
    conf.total_samples = 96_000_000
    conf.sample_every_samples = 400_000
    conf.eval_every_samples = 20_000_000
    conf.eval_ema_every_samples = 20_000_000
    conf.save_every_samples = 2_000_000
    return conf


def latent_diffusion128_config(conf: TrainConfig):
    conf = latent_diffusion_config(conf)
    conf.batch_size_eval = 32
    return conf


def latent_mlp_2048_norm_10layers(conf: TrainConfig):
    conf.net_latent_net_type = LatentNetType.skip
    conf.net_latent_layers = 10
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    conf.net_latent_activation = Activation.silu
    conf.net_latent_num_hid_channels = 2048
    conf.net_latent_use_norm = True
    conf.net_latent_condition_bias = 1
    return conf


def latent_mlp_2048_norm_20layers(conf: TrainConfig):
    conf = latent_mlp_2048_norm_10layers(conf)
    conf.net_latent_layers = 20
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    return conf


def latent_256_batch_size(conf: TrainConfig):
    conf.batch_size = 124
    conf.eval_ema_every_samples = 100_000_000
    conf.eval_every_samples = 100_000_000
    conf.sample_every_samples = 1_000_000
    conf.save_every_samples = 2_000_000
    conf.total_samples = 301_000_000
    return conf


def latent_512_batch_size(conf: TrainConfig):
    conf.batch_size = 512
    conf.eval_ema_every_samples = 100_000_000
    conf.eval_every_samples = 100_000_000
    conf.sample_every_samples = 1_000_000
    conf.save_every_samples = 5_000_000
    conf.total_samples = 501_000_000
    return conf


def latent_2048_batch_size(conf: TrainConfig):
    conf.batch_size = 2048
    conf.eval_ema_every_samples = 200_000_000
    conf.eval_every_samples = 200_000_000
    conf.sample_every_samples = 4_000_000
    conf.save_every_samples = 20_000_000
    conf.total_samples = 1_501_000_000
    return conf


def adamw_weight_decay(conf: TrainConfig):
    conf.optimizer = OptimizerType.adamw
    conf.weight_decay = 0.01
    return conf

def texture100k_autoenc_latent():
    conf = pretrain_texture100k_autoenc()
    conf = latent_diffusion128_config(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = latent_256_batch_size(conf)
    conf = adamw_weight_decay(conf)
    conf.total_samples = 130_000_000
    conf.latent_loss_type = LossType.l1
    conf.latent_beta_scheduler = 'const0.008'
    conf.eval_ema_every_samples = 200_000_000
    conf.eval_every_samples = 200_000_000
    conf.sample_every_samples = 4_000_000
    conf.name = 'texture100k_autoenc_latent'
    return conf


def brain_autoenc_latent():
    conf = pretrain_brain_autoenc()
    conf = latent_diffusion128_config(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = latent_256_batch_size(conf)
    conf = adamw_weight_decay(conf)
    conf.total_samples = 130_000_000
    conf.latent_loss_type = LossType.l1
    conf.latent_beta_scheduler = 'const0.008'
    conf.eval_ema_every_samples = 200_000_000
    conf.eval_every_samples = 200_000_000
    conf.sample_every_samples = 4_000_000
    conf.name = 'brain_autoenc_latent'
    return conf

def tcga_crc_autoenc_latent():
    conf = pretrain_tcga_crc_autoenc()
    conf = latent_diffusion128_config(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = latent_256_batch_size(conf)
    conf = adamw_weight_decay(conf)
    conf.total_samples = 130_000_000
    conf.latent_loss_type = LossType.l1
    conf.latent_beta_scheduler = 'const0.008'
    conf.eval_ema_every_samples = 200_000_000
    conf.eval_every_samples = 200_000_000
    conf.sample_every_samples = 4_000_000
    conf.name = 'tcga_crc_autoenc_latent'
    return conf


def tcga_crc_512_autoenc_latent():
    conf = pretrain_tcga_crc_512_autoenc()
    conf = latent_diffusion128_config(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = latent_256_batch_size(conf)
    conf = adamw_weight_decay(conf)
    conf.total_samples = 130_000_000
    conf.latent_loss_type = LossType.l1
    conf.latent_beta_scheduler = 'const0.008'
    conf.eval_ema_every_samples = 200_000_000
    conf.eval_every_samples = 200_000_000
    conf.sample_every_samples = 4_000_000
    conf.name = 'tcga_crc_autoenc_512_latent'
    return conf

def tcga_brca_512_autoenc_latent():
    conf = pretrain_tcga_brca_512_autoenc()
    conf = latent_diffusion128_config(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = latent_256_batch_size(conf)
    conf = adamw_weight_decay(conf)
    conf.total_samples = 150_000_000
    conf.latent_loss_type = LossType.l1
    conf.latent_beta_scheduler = 'const0.008'
    conf.eval_ema_every_samples = 200_000_000
    conf.eval_every_samples = 200_000_000
    conf.sample_every_samples = 4_000_000
    conf.name = 'tcga_brca_autoenc_512_latent'
    return conf

def pancancer_autoenc_latent():
    conf = pretrain_pancancer_autoenc()
    conf = latent_diffusion128_config(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = latent_256_batch_size(conf)
    conf = adamw_weight_decay(conf)
    conf.total_samples = 130_000_000
    conf.latent_loss_type = LossType.l1
    conf.latent_beta_scheduler = 'const0.008'
    conf.eval_ema_every_samples = 200_000_000
    conf.eval_every_samples = 200_000_000
    conf.sample_every_samples = 4_000_000
    conf.name = 'pancancer_latent'
    return conf
