from mopadi.train_diff_autoenc import *


def ddpm():
    """
    base configuration for all DDIM-based models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_ddpm
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512   # conch v1.5 = 768
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf


def autoenc_base():
    """
    base configuration for all Diff-AE models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_autoenc
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512   # conch v1.5 = 768
    conf.net_beatgans_resnet_two_cond = True
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_pool = 'adaptivenonzero'
    conf.sample_size = 16    # change here to 16 for 512x512 model
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf


def default_autoenc(config):
    conf = autoenc_base()
    conf.base_dir = config.get('base_dir', './checkpoints/no_name')

    data_config = config.get('data', {})
    autoenc_config = config.get('autoenc_model', {})

    conf.data_dirs = list(data_config.get('data_dirs', []))
    conf.feature_dirs = list(data_config.get('feature_dirs', []))
    conf.data_val_dirs = list(data_config.get('data_val_dirs', []))
    conf.feature_val_dirs = list(data_config.get('feature_val_dirs', []))
    conf.feat_extractor = data_config.get('feature_extractor', None)
    #conf.test_patients_file_path = data_config.get('test_patients_file_path', None)
    #conf.process_only_zips = data_config.get('process_only_zips', False)
    #conf.cache_pickle_tiles_path = data_config.get('cache_pickle_tiles_path', None)
    #conf.cache_cohort_sizes_path = data_config.get('cache_cohort_sizes_path', None)
    #conf.split = data_config.get('split', 'none')
    conf.max_tiles_per_patient = data_config.get('max_tiles_per_patient', None)
    conf.cohort_size_threshold = data_config.get('cohort_size_threshold', None)
    conf.as_tensor = data_config.get('as_tensor', True)
    conf.do_normalize = data_config.get('do_normalize', True)
    conf.do_resize = data_config.get('do_resize', False)

    conf.load_pretrained_autoenc = False

    _size_by_extractor = {
        "conch": 448,
        "conch1_5": 448,
        "v2": 224,
        "uni2": 224,
    }
    fe_key = (conf.feat_extractor or "").lower()
    default_img_size = _size_by_extractor.get(fe_key, 224)

    conf.img_size = autoenc_config.get("img_size", default_img_size)
    conf.sample_size = autoenc_config.get('sample_size', 32)
    conf.batch_size = autoenc_config.get('batch_size', 64)
    conf.batch_size_eval = autoenc_config.get('batch_size_eval', 64)
    conf.total_samples = autoenc_config.get('total_samples', 200_000_000)
    conf.steps_per_epoch = autoenc_config.get('steps_per_epoch', 5_000)
    conf.warmup = autoenc_config.get('warmup', 0)
    conf.net_ch = autoenc_config.get('net_ch', 128)
    conf.net_ch_mult = tuple(autoenc_config.get('net_ch_mult', (1, 1, 2, 2, 4, 4)))
    conf.net_enc_channel_mult = tuple(autoenc_config.get('net_enc_channel_mult', (1, 1, 2, 2, 4, 4, 4)))
    conf.eval_every_samples = autoenc_config.get('eval_every_samples', 1_000_000)
    conf.eval_ema_every_samples = autoenc_config.get('eval_ema_every_samples', 1_000_000)

    conf.optimizer = autoenc_config.get('optimizer', 'adam')
    if conf.optimizer == 'adam':
        conf.optimizer = OptimizerType.adam
    elif conf.optimizer == 'lion':
        conf.optimizer = OptimizerType.lion
    elif conf.optimizer == 'adamw':
        conf.optimizer = OptimizerType.adamw

    _feat_dim_by_extractor = {
    "conch": 512,
    "conch1_5": 512,
    "v2": 1280,
    "uni2": 1536,
    }
    feat_dim = _feat_dim_by_extractor.get(fe_key, 512)
    conf.feat_dim = autoenc_config.get('feat_dim', feat_dim)
    conf.style_ch = conf.feat_dim
    conf.net_beatgans_embed_channels = conf.feat_dim

    conf.feat_loss = autoenc_config.get('feat_loss', False)
    conf.lambda_feat = autoenc_config.get('lambda_feat', 0.3)
    conf.lambda_lp = autoenc_config.get('lambda_lp', 0.1)

    # Additional training parameters
    conf.lr = autoenc_config.get('lr', 1e-4)
    conf.weight_decay = autoenc_config.get('weight_decay', 0.0)
    conf.dropout = autoenc_config.get('dropout', 0.1)
    conf.num_workers = autoenc_config.get('num_workers', 6)
    conf.grad_clip = autoenc_config.get('grad_clip', 1.0)

    # Checkpointing
    conf.save_every_samples = autoenc_config.get('save_every_samples', 100_000)
    conf.reconstruct_every_samples = autoenc_config.get('reconstruct_every_samples', 20_000)
    conf.eval_num_images = autoenc_config.get('eval_num_images', 5_000)

    conf.name = 'autoenc'
    conf.make_model_conf()
    return conf


def texture100k_autoenc():
    conf = autoenc_base()
    conf.data_name = 'texture'
    conf.base_dir = 'checkpoints/texture100k'
    conf.warmup = 0
    conf.total_samples = 200_000_000
    conf.sample_size = 32
    conf.img_size = 224
    conf.batch_size = 64
    conf.batch_size_eval = 64
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    # conf.scale_up_gpus(4)
    conf.make_model_conf()
    return conf


def tcga_crc_autoenc():
    conf = autoenc_base()
    conf.data_name = 'tcga_crc_512'
    conf.base_dir = 'checkpoints/crc/tcga_crc_512'
    conf.warmup = 0
    conf.total_samples = 70_000_000
    conf.sample_size = 16
    conf.img_size = 512
    conf.batch_size = 32
    conf.batch_size_eval = 32
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    # conf.scale_up_gpus(2)
    conf.make_model_conf()
    return conf


def tcga_brca_autoenc():
    conf = autoenc_base()
    conf.data_name = 'tcga_brca_512'
    conf.base_dir = 'checkpoints/brca/autoenc'
    conf.warmup = 0
    conf.total_samples = 70_000_000
    conf.sample_size = 16
    conf.img_size = 512
    conf.batch_size = 16
    conf.batch_size_eval = 16
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    # conf.scale_up_gpus(2)
    conf.make_model_conf()
    return conf


def pancancer_autoenc():
    conf = autoenc_base()
    conf.data_name = 'pancancer'
    conf.base_dir = '/checkpoints/pancancer/autoenc'
    conf.warmup = 0
    conf.total_samples = 200_000_000
    conf.sample_size = 32
    conf.img_size = 256
    conf.batch_size = 48  # had to reduce due to one broken gpu on dgx
    conf.batch_size_eval = 48  # had to reduce due to one broken gpu on dgx
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.make_model_conf()
    return conf
