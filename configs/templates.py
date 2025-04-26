from train_diff_autoenc import *
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv('WORKSPACE_PATH')


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
    conf.net_beatgans_embed_channels = 512
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
    conf.net_beatgans_embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_pool = 'adaptivenonzero'
    conf.sample_size = 32    # change here to 16 for 512x512 model
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf

def default_autoenc(config):
    conf = autoenc_base()
    conf.base_dir = config.get('exp', {}).get('base_dir', './checkpoints/no_name')

    train_config = config.get('train', {})
    conf.data_dirs = list(train_config.get('data', {}).get('data_dirs', None))
    conf.test_patients_file_path = train_config.get('data', {}).get('test_patients_file_path', None)
    conf.process_only_zips = train_config.get('data', {}).get('process_only_zips', False)
    conf.cache_pickle_tiles_path = train_config.get('data', {}).get('cache_pickle_tiles_path', None)
    conf.cache_cohort_sizes_path = train_config.get('data', {}).get('cache_cohort_sizes_path', None)
    conf.split = train_config.get('data', {}).get('split', 'none')
    conf.max_tiles_per_patient = train_config.get('data', {}).get('max_tiles_per_patient', None)
    conf.cohort_size_threshold = train_config.get('data', {}).get('cohort_size_threshold', None)
    conf.as_tensor = train_config.get('data', {}).get('as_tensor', True)
    conf.do_normalize = train_config.get('data', {}).get('do_normalize', True)
    conf.do_resize = train_config.get('data', {}).get('do_resize', False)

    model_config = train_config.get('model', {})
    conf.img_size = model_config.get('img_size', 224)
    conf.batch_size = model_config.get('batch_size', 64)
    conf.batch_size_eval = model_config.get('batch_size_eval', 64)
    conf.total_samples = model_config.get('total_samples', 200_000_000)
    conf.warmup = model_config.get('warmup', 0)
    conf.net_ch = model_config.get('net_ch', 128)
    conf.net_ch_mult = tuple(model_config.get('net_ch_mult', (1, 1, 2, 2, 4, 4)))
    conf.net_enc_channel_mult = tuple(model_config.get('net_enc_channel_mult', (1, 1, 2, 2, 4, 4, 4)))
    conf.eval_every_samples = model_config.get('eval_every_samples', 1_000_000)
    conf.eval_ema_every_samples = model_config.get('eval_ema_every_samples', 1_000_000)
    
    conf.name = 'autoenc'
    conf.make_model_conf()
    return conf

def texture100k_autoenc():
    conf = autoenc_base()
    conf.data_name = 'texture'
    conf.base_dir = f'{ws_path}/mopadi/checkpoints/texture100k'
    conf.warmup = 0
    conf.total_samples = 200_000_000
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
    conf.data_name = 'tcga_crc'
    conf.base_dir = f'{ws_path}/mopadi/checkpoints/tcga_crc_224x224/autoenc'
    conf.warmup = 0
    conf.total_samples = 200_000_000
    conf.img_size = 224
    conf.batch_size = 64
    conf.batch_size_eval = 64
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.scale_up_gpus(2)
    conf.make_model_conf()
    return conf


def tcga_crc_512_autoenc():
    conf = autoenc_base()
    conf.data_name = 'tcga_crc_512'
    conf.base_dir = f'{ws_path}/mopadi/checkpoints/crc/tcga_crc_512'
    conf.warmup = 0
    conf.total_samples = 70_000_000
    conf.sample_size = 16
    conf.img_size = 512
    conf.batch_size = 24
    conf.batch_size_eval = 24
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
    conf.base_dir = f'{ws_path}/mopadi/checkpoints/brca/autoenc'
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


def brain_autoenc():
    conf = autoenc_base()
    conf.data_name = 'brain'
    conf.base_dir = f'{ws_path}/mopadi/checkpoints/brain'
    conf.warmup = 0
    conf.total_samples = 200_000_000
    conf.img_size = 224
    conf.batch_size = 64
    conf.batch_size_eval = 64
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.make_model_conf()
    return conf


def pancancer_autoenc():
    conf = autoenc_base()
    conf.data_name = 'pancancer'
    conf.base_dir = f'{ws_path}/mopadi/checkpoints/pancancer/autoenc'
    conf.warmup = 0
    conf.total_samples = 200_000_000
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
