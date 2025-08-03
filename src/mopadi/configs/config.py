# Code snippets sourced from the Official implementation of Diffusion Autoencoders by Konpat Preechakul
# with modifications by Laura Zigutyte and Tim Lenz
# Original Source: https://github.com/phizaz/diffae
# License: MIT

import os
import shutil
from typing import Tuple
from multiprocessing import get_context
from dataclasses import dataclass
from dotenv import load_dotenv

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from mopadi.configs.config_base import BaseConfig
from mopadi.dataset import *
from mopadi.diffusion import *
from mopadi.diffusion.base import GenerativeType, LossType, ModelMeanType, ModelVarType, get_named_beta_schedule
from mopadi.model import *
from mopadi.configs.choices import *
from mopadi.model.unet import ScaleAt
from mopadi.model.latentnet import *
from mopadi.diffusion.resample import UniformSampler
from mopadi.diffusion.diffusion import space_timesteps

load_dotenv()
ws_path = os.getenv('WORKSPACE_PATH')


tcga_all = [
    f'{ws_path}/cache/TCGA-CRC',
    f'{ws_path}/cache/TCGA-BRCA',
    f'{ws_path}/cache/TCGA-BLCA',
    f'{ws_path}/cache/TCGA-CESC',
    f'{ws_path}/cache/TCGA-CHOL',
    f'{ws_path}/cache/TCGA-DLBC',
    f'{ws_path}/cache/TCGA-ESCA',
    f'{ws_path}/cache/TCGA-GBM',
    f'{ws_path}/cache/TCGA-HNSC',
    f'{ws_path}/cache/TCGA-KICH',
    f'{ws_path}/cache/TCGA-KIRC',
    f'{ws_path}/cache/TCGA-KIRP',
    f'{ws_path}/cache/TCGA-LGG',
    f'{ws_path}/cache/TCGA-LIHC',
    f'{ws_path}/cache/TCGA-LUAD',
    f'{ws_path}/cache/TCGA-LUSC',
    f'{ws_path}/cache/TCGA-MESO',
    f'{ws_path}/cache/TCGA-OV',
    f'{ws_path}/cache/TCGA-PAAD',
    f'{ws_path}/cache/TCGA-PCPG',
    f'{ws_path}/cache/TCGA-PRAD',
    f'{ws_path}/cache/TCGA-SARC',
    f'{ws_path}/cache/TCGA-SKCM',
    f'{ws_path}/cache/TCGA-STAD',
    f'{ws_path}/cache/TCGA-TGCT',
    f'{ws_path}/cache/TCGA-THCA',
    f'{ws_path}/cache/TCGA-THYM',
    f'{ws_path}/cache/TCGA-UCEC',
    f'{ws_path}/cache/TCGA-UCS',
]


tcga_all_feats = [
    f'{ws_path}/features/mahmood-conch/TCGA-LUSC-from-cache/mahmood-conch-02627079',
    f'{ws_path}/features/mahmood-conch/TCGA-LIHC-from-cache/mahmood-conch-34c73b45', 
    f'{ws_path}/features/mahmood-conch/TCGA-LUAD-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-CRC-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-BRCA-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-BLCA-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-CESC-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-CHOL-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-DLBC-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-ESCA-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-GBM-from-cache/mahmood-conch-34c73b45', 
    f'{ws_path}/features/mahmood-conch/TCGA-HNSC-from-cache/mahmood-conch-34c73b45', 
    f'{ws_path}/features/mahmood-conch/TCGA-KICH-from-cache/mahmood-conch-34c73b45', 
    f'{ws_path}/features/mahmood-conch/TCGA-KIRC-from-cache/mahmood-conch-34c73b45', 
    f'{ws_path}/features/mahmood-conch/TCGA-KIRP-from-cache/mahmood-conch-34c73b45', 
    f'{ws_path}/features/mahmood-conch/TCGA-LGG-from-cache/mahmood-conch-34c73b45', 
    f'{ws_path}/features/mahmood-conch/TCGA-MESO-from-cache/mahmood-conch-34c73b45', 
    f'{ws_path}/features/mahmood-conch/TCGA-OV-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-PAAD-from-cache/mahmood-conch-34c73b45', 
    f'{ws_path}/features/mahmood-conch/TCGA-PCPG-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-PRAD-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-SARC-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-SKCM-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-STAD-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-TGCT-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-THCA-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-THYM-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-UCEC-from-cache/mahmood-conch-34c73b45',
    f'{ws_path}/features/mahmood-conch/TCGA-UCS-from-cache/mahmood-conch-34c73b45',
]

data_paths = {
    'tcga_all_conch': tcga_all,
    'tcga_all_conch_sample_1024': tcga_all,

    'tcga_crc_512_conch_nolmdb':  f'{ws_path}/cache/TCGA-CRC',
    'tcga_brca_512_conch_nolmdb': f'{ws_path}/cache/TCGA-BRCA',
    'tcga_blca_512_conch_nolmdb': f'{ws_path}/cache/TCGA-BLCA',
    'tcga_cesc_512_conch_nolmdb': f'{ws_path}/cache/TCGA-CESC',
    'tcga_chol_512_conch_nolmdb': f'{ws_path}/cache/TCGA-CHOL',
    'tcga_dlbc_512_conch_nolmdb': f'{ws_path}/cache/TCGA-DLBC',
    'tcga_esca_512_conch_nolmdb': f'{ws_path}/cache/TCGA-ESCA',
    'tcga_gbm_512_conch_nolmdb':  f'{ws_path}/cache/TCGA-GBM',
    'tcga_hnsc_512_conch_nolmdb': f'{ws_path}/cache/TCGA-HNSC',
    'tcga_kich_512_conch_nolmdb': f'{ws_path}/cache/TCGA-KICH',
    'tcga_kirc_512_conch_nolmdb': f'{ws_path}/cache/TCGA-KIRC',
    'tcga_kirp_512_conch_nolmdb': f'{ws_path}/cache/TCGA-KIRP',
    'tcga_lgg_512_conch_nolmdb':  f'{ws_path}/cache/TCGA-LGG',
    'tcga_lihc_512_conch_nolmdb': f'{ws_path}/cache/TCGA-LIHC',
    'tcga_luad_512_conch_nolmdb': f'{ws_path}/cache/TCGA-LUAD',
    'tcga_lusc_512_conch_nolmdb': f'{ws_path}/cache/TCGA-LUSC',
    'tcga_meso_512_conch_nolmdb': f'{ws_path}/cache/TCGA-MESO',
    'tcga_ov_512_conch_nolmdb':   f'{ws_path}/cache/TCGA-OV',
    'tcga_paad_512_conch_nolmdb': f'{ws_path}/cache/TCGA-PAAD',
    'tcga_pcpg_512_conch_nolmdb': f'{ws_path}/cache/TCGA-PCPG',
    'tcga_prad_512_conch_nolmdb': f'{ws_path}/cache/TCGA-PRAD',
    'tcga_sarc_512_conch_nolmdb': f'{ws_path}/cache/TCGA-SARC',
    'tcga_skcm_512_conch_nolmdb': f'{ws_path}/cache/TCGA-SKCM',
    'tcga_stad_512_conch_nolmdb': f'{ws_path}/cache/TCGA-STAD',
    'tcga_tgct_512_conch_nolmdb': f'{ws_path}/cache/TCGA-TGCT',
    'tcga_thca_512_conch_nolmdb': f'{ws_path}/cache/TCGA-THCA',
    'tcga_thym_512_conch_nolmdb': f'{ws_path}/cache/TCGA-THYM',
    'tcga_ucec_512_conch_nolmdb': f'{ws_path}/cache/TCGA-UCEC',
    'tcga_ucs_512_conch_nolmdb':  f'{ws_path}/cache/TCGA-UCS',

    'tcga_crc_224_v2': f'{ws_path}/cache/TCGA-CRC',
    'tcga_crc_448_conch1_5': f'{ws_path}/cache/TCGA-CRC',
    'tcga_crc_448_conch': f'{ws_path}/cache/TCGA-CRC',

    'tcga_crc_224_uni2': f'{ws_path}/cache/TCGA-CRC',
}

feat_paths = {
    # CONCH
    'tcga_all_conch': tcga_all_feats,
    'tcga_all_conch_sample_1024': tcga_all_feats,

    'tcga_crc_512_conch_nolmdb_fl32':  f'{ws_path}/features/mahmood-conch/TCGA-CRC-from-cache-fl32/mahmood-conch-68e004f9',
    'tcga_crc_512_conch_nolmdb_fl32':  f'{ws_path}/features/mahmood-conch/TCGA-CRC-from-cache-fl32/mahmood-conch-68e004f9',     # float32
    'tcga_brca_512_conch_nolmdb_fl32': f'{ws_path}/features/mahmood-conch/TCGA-BRCA-from-cache-fl32/mahmood-conch-68e004f9',   # float32

    'tcga_lusc_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-LUSC-from-cache/mahmood-conch-02627079',

    'tcga_lihc_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-LIHC-from-cache/mahmood-conch-34c73b45', 
    'tcga_luad_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-LUAD-from-cache/mahmood-conch-34c73b45',
    'tcga_crc_512_conch_nolmdb':  f'{ws_path}/features/mahmood-conch/TCGA-CRC-from-cache/mahmood-conch-34c73b45',
    'tcga_brca_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-BRCA-from-cache/mahmood-conch-34c73b45',
    'tcga_blca_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-BLCA-from-cache/mahmood-conch-34c73b45',
    'tcga_cesc_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-CESC-from-cache/mahmood-conch-34c73b45',
    'tcga_chol_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-CHOL-from-cache/mahmood-conch-34c73b45',
    'tcga_dlbc_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-DLBC-from-cache/mahmood-conch-34c73b45',
    'tcga_esca_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-ESCA-from-cache/mahmood-conch-34c73b45',
    'tcga_gbm_512_conch_nolmdb':  f'{ws_path}/features/mahmood-conch/TCGA-GBM-from-cache/mahmood-conch-34c73b45', 
    'tcga_hnsc_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-HNSC-from-cache/mahmood-conch-34c73b45', 
    'tcga_kich_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-KICH-from-cache/mahmood-conch-34c73b45', 
    'tcga_kirc_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-KIRC-from-cache/mahmood-conch-34c73b45', 
    'tcga_kirp_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-KIRP-from-cache/mahmood-conch-34c73b45', 
    'tcga_lgg_512_conch_nolmdb':  f'{ws_path}/features/mahmood-conch/TCGA-LGG-from-cache/mahmood-conch-34c73b45', 
    'tcga_meso_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-MESO-from-cache/mahmood-conch-34c73b45', 
    'tcga_ov_512_conch_nolmdb':   f'{ws_path}/features/mahmood-conch/TCGA-OV-from-cache/mahmood-conch-34c73b45',
    'tcga_paad_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-PAAD-from-cache/mahmood-conch-34c73b45', 
    'tcga_pcpg_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-PCPG-from-cache/mahmood-conch-34c73b45',
    'tcga_prad_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-PRAD-from-cache/mahmood-conch-34c73b45',
    'tcga_sarc_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-SARC-from-cache/mahmood-conch-34c73b45',
    'tcga_skcm_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-SKCM-from-cache/mahmood-conch-34c73b45',
    'tcga_stad_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-STAD-from-cache/mahmood-conch-34c73b45',
    'tcga_tgct_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-TGCT-from-cache/mahmood-conch-34c73b45',
    'tcga_thca_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-THCA-from-cache/mahmood-conch-34c73b45',
    'tcga_thym_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-THYM-from-cache/mahmood-conch-34c73b45',
    'tcga_ucec_512_conch_nolmdb': f'{ws_path}/features/mahmood-conch/TCGA-UCEC-from-cache/mahmood-conch-34c73b45',
    'tcga_ucs_512_conch_nolmdb':  f'{ws_path}/features/mahmood-conch/TCGA-UCS-from-cache/mahmood-conch-34c73b45',

    # VIRCHOW2
    'tcga_crc_224_v2': f'{ws_path}/features/virchow2/TCGA-CRC-from-cache/virchow2-34c73b45',
    'tcga_brca_224_v2': f'{ws_path}/features/virchow2/TCGA-BRCA-from-cache/virchow2-34c73b45',

    # CONCHv1.5
    'tcga_crc_448_conch1_5': f'{ws_path}/features/mahmood-conch1_5/TCGA-CRC-from-cache/mahmood-conch1_5-34c73b45',
    'tcga_brca_448_conch1_5': f'{ws_path}/features/mahmood-conch1_5/TCGA-BRCA-from-cache/mahmood-conch1_5-34c73b45',

    #UNI2
    'tcga_crc_224_uni2': f'{ws_path}/features/mahmood-uni2/TCGA-CRC-from-cache/uni2-34c73b45',

    #CONCH
    'tcga_crc_448_conch': f'{ws_path}/features/mahmood-conch/TCGA-CRC-from-cache/mahmood-conch-34c73b45',

}

test_patient_files = {
    'tcga_all_conch': None,
    'tcga_crc_512_conch_nolmdb': f'{ws_path}/dev_fm/mopadi/datasets/patient_splits/TCGA_CRC_test_split.json'
}

@dataclass
class PretrainConfig(BaseConfig):
    name: str
    path: str
    
@dataclass
class MILconfig(BaseConfig):
    target_label: str = ""
    target_dict: dict = None
    dim: int = 512
    num_heads: int = 8
    num_seeds: int = 4
    num_classes: int = 2
    nr_feats: int = 512
    nr_top_tiles: int = 15
    num_epochs: int = 300
    lr: float = 1e-4
    batch_size: int = 16
    num_workers: int = 8
    es: int = 20

@dataclass
class TrainConfig(BaseConfig):
    # random seed
    seed: int = 0
    linear: bool = True
    train_mode: TrainMode = TrainMode.diffusion
    train_cond0_prob: float = 0
    train_pred_xstart_detach: bool = True
    train_interpolate_prob: float = 0
    train_interpolate_img: bool = False
    manipulate_mode: str = ''
    manipulate_cls: str = None
    manipulate_shots: int = None
    manipulate_loss: ManipulateLossType = ManipulateLossType.bce
    manipulate_znormalize: bool = False
    manipulate_seed: int = 0
    accum_batches: int = 1
    autoenc_mid_attn: bool = True
    batch_size: int = 16
    batch_size_eval: int = None
    beatgans_gen_type: GenerativeType = GenerativeType.ddim
    beatgans_loss_type: LossType = LossType.mse
    feat_loss: bool = False
    lambda_feat: float = 0.5   # for computing final loss if feat_loss True
    beatgans_model_mean_type: ModelMeanType = ModelMeanType.eps
    beatgans_model_var_type: ModelVarType = ModelVarType.fixed_large
    beatgans_rescale_timesteps: bool = False
    latent_znormalize: bool = False
    beta_scheduler: str = 'linear'
    data_name: str = ''
    data_val_name: str = None
    diffusion_type: str = None
    dropout: float = 0.1
    ema_decay: float = 0.9999
    eval_num_images: int = 5_000
    eval_every_samples: int = 200_000
    eval_ema_every_samples: int = 200_000
    fid_use_torch: bool = True
    fp16: bool = False
    grad_clip: float = 1
    img_size: int = 64
    lr: float = 0.0001  # 0.0001 default (adam); 3-10x smaller for lion optimizer
    optimizer: OptimizerType = OptimizerType.adam  #adam or adamw or lion
    weight_decay: float = 0  # default - 0; lion - 3-10x larger than that for AdamW
    model_conf: ModelConfig = None
    model_name: ModelName = None
    model_type: ModelType = None
    net_attn: Tuple[int] = None
    net_beatgans_attn_head: int = 1
    # not necessarily the same as the the number of style channels
    net_beatgans_embed_channels: int = 512 # conch v1.5 = 768
    feat_extractor: str = 'conch'
    feat_dim: int = 512
    enc_transform_dim: int = 1024  # feat projection layer
    enc_transform_nheads: int = 8  # feat projection layer
    enc_transform_num_layers: int = 2  # feat projection layer
    net_resblock_updown: bool = True
    net_enc_use_time: bool = False
    net_enc_pool: str = 'adaptivenonzero'
    net_beatgans_gradient_checkpoint: bool = False
    net_beatgans_resnet_two_cond: bool = False
    net_beatgans_resnet_use_zero_module: bool = True
    net_beatgans_resnet_scale_at: ScaleAt = ScaleAt.after_norm
    net_beatgans_resnet_cond_channels: int = None
    net_ch_mult: Tuple[int] = None
    net_ch: int = 64
    net_enc_attn: Tuple[int] = None
    net_enc_k: int = None
    # number of resblocks for the encoder (half-unet)
    net_enc_num_res_blocks: int = 2
    net_enc_channel_mult: Tuple[int] = None
    net_enc_grad_checkpoint: bool = False
    net_autoenc_stochastic: bool = False
    #feature_extractor: FeatureExtractor = FeatureExtractor.conch
    net_num_res_blocks: int = 2
    # number of resblocks for the UNET
    net_num_input_res_blocks: int = None
    net_enc_num_cls: int = None
    num_workers: int = 4 # 20 for dgx
    parallel: bool = False
    postfix: str = ''
    sample_size: int = 64
    sample_every_samples: int = 20_000
    save_every_samples: int = 100_000
    style_ch: int = 512   # conch v1.5 = 768
    T_eval: int = 1_000
    T_sampler: str = 'uniform'
    T: int = 1_000
    total_samples: int = 10_000_000
    warmup: int = 0
    pretrain: PretrainConfig = None
    continue_from: PretrainConfig = None
    eval_programs: Tuple[str] = None
    # if present load the checkpoint from this path instead
    eval_path: str = None
    base_dir: str = 'checkpoints'
    use_cache_dataset: bool = False
    data_cache_dir: str = os.path.join(ws_path, 'cache')
    work_cache_dir: str = os.path.join(ws_path, 'mopadi_cache')
    # to be overridden
    name: str = ''

    def __post_init__(self):
        self.batch_size_eval = self.batch_size_eval or self.batch_size
        self.data_val_name = self.data_val_name or self.data_name

    def scale_up_gpus(self, num_gpus, num_nodes=1):
        self.eval_ema_every_samples *= num_gpus * num_nodes
        self.eval_every_samples *= num_gpus * num_nodes
        self.sample_every_samples *= num_gpus * num_nodes
        self.batch_size *= num_gpus * num_nodes
        self.batch_size_eval *= num_gpus * num_nodes
        return self

    @property
    def batch_size_effective(self):
        return self.batch_size * self.accum_batches

    @property
    def fid_cache(self):
        return f'{self.work_cache_dir}/eval_images/{self.data_name}_size{self.img_size}_{self.eval_num_images}'

    @property
    def data_path(self):
        # may use the cache dir
        path = data_paths[self.data_name]
        if self.use_cache_dataset and path is not None:
            path = use_cached_dataset_path(
                path, f'{self.data_cache_dir}/{self.data_name}')
        return path

    @property
    def feat_path(self):
        feat_path = feat_paths[self.data_name]
        return feat_path

    @property
    def test_patient_file(self):
        test_patient_file = test_patient_files[self.data_name]
        return test_patient_file

    @property
    def logdir(self):
        return f'{self.base_dir}/{self.name}'

    @property
    def generate_dir(self):
        return f'{self.work_cache_dir}/gen_images/{self.name}'

    @property
    def normalization_params(self):
        params_dict = {
            "conch1_5": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            "conch": ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
            "v2": ([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250]),
            "uni2": ([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250]),
            "default": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        }
        
        if self.feat_extractor in params_dict:
            return params_dict[self.feat_extractor]
        else:
            raise ValueError(f"Unknown feature extractor '{self.feat_extractor}'. "
                             "Please add it to 'normalization_params'.")


    def _make_diffusion_conf(self, T=None):
        if self.diffusion_type == 'beatgans':
            # can use T < self.T for evaluation
            # follows the guided-diffusion repo conventions
            # t's are evenly spaced
            if self.beatgans_gen_type == GenerativeType.ddpm:
                section_counts = [T]
            elif self.beatgans_gen_type == GenerativeType.ddim:
                section_counts = f'ddim{T}'
            else:
                raise NotImplementedError()

            diffusion_conf = SpacedDiffusionBeatGansConfig(
                gen_type=self.beatgans_gen_type,
                model_type=self.model_type,
                betas=get_named_beta_schedule(self.beta_scheduler, self.T),
                model_mean_type=self.beatgans_model_mean_type,
                model_var_type=self.beatgans_model_var_type,
                loss_type=self.beatgans_loss_type,
                rescale_timesteps=self.beatgans_rescale_timesteps,
                use_timesteps=space_timesteps(num_timesteps=self.T,
                                              section_counts=section_counts),
                fp16=self.fp16,
                feat_loss=self.feat_loss,
                lambda_feat=self.lambda_feat
            )
            diffusion_conf.normalization_params = self.normalization_params
            #TODO: this part is weird
            if self.feat_extractor == 'conch':
                diffusion_conf.feature_extractor = FeatureExtractorConch()
            elif self.feat_extractor == 'conch1_5':
                diffusion_conf.feature_extractor = FeatureExtractorConch15()
            elif self.feat_extractor == 'v2':
                diffusion_conf.feature_extractor = FeatureExtractorVirchow2()
            elif self.feat_extractor == 'uni2':
                diffusion_conf.feature_extractor = FeatureExtractorUNI2()
            else:
                raise ValueError(f"Unknown feature extractor '{self.feat_extractor}'. Please check configuration.")

            return diffusion_conf
        else:
            raise NotImplementedError()

    @property
    def model_out_channels(self):
        return 3

    def make_T_sampler(self):
        if self.T_sampler == 'uniform':
            return UniformSampler(self.T)
        else:
            raise NotImplementedError()

    def make_diffusion_conf(self):
        return self._make_diffusion_conf(self.T)

    def make_eval_diffusion_conf(self):
        return self._make_diffusion_conf(T=self.T_eval)

    def make_dataset(self, path=None, **kwargs):
        print(f"Used dataset: {self.data_name}, {path} or {self.data_path}")
        if self.data_name == 'tcga_crc_512_conch_nolmdb' or self.data_name == 'tcga_brca_512_conch_nolmdb':
            return ImageTileDatasetWithFeatures(root_dirs=[path] or [self.data_path], features_dirs=self.feat_path, test_patients_file=self.test_patient_file, feat_extractor='conch', **kwargs)
        elif self.data_name == 'tcga_all_conch':
            return ImageTileDatasetWithFeatures(root_dirs=self.data_path, features_dirs=self.feat_path, test_patients_file=None, feat_extractor='conch', cache_pickle_tiles_path='temp/tcga_all_tile_paths_all.pkl', **kwargs)
        elif self.data_name == 'tcga_all_conch_sample_1024':
            return ImageTileDatasetWithFeatures(root_dirs=self.data_path, features_dirs=self.feat_path, test_patients_file=None, max_tiles_per_patient=1024, feat_extractor='conch', cache_pickle_tiles_path='temp/tcga_all_tile_paths_sampled_1024.pkl', **kwargs)
        elif self.data_name == 'tcga_crc_224_v2':
            return ImageTileDatasetWithFeatures(root_dirs=[self.data_path], features_dirs=[self.feat_path], test_patients_file=None, max_tiles_per_patient=None, feat_extractor='v2', cache_pickle_tiles_path='temp/tcga_crc.pkl', **kwargs)
        elif self.data_name == 'tcga_crc_448_conch1_5':
            return ImageTileDatasetWithFeatures(root_dirs=[self.data_path], features_dirs=[self.feat_path], test_patients_file=None, max_tiles_per_patient=None, feat_extractor='conch1_5', cache_pickle_tiles_path='temp/tcga_crc.pkl', **kwargs)
        elif self.data_name == 'tcga_crc_448_conch':
            return ImageTileDatasetWithFeatures(root_dirs=[self.data_path], features_dirs=[self.feat_path], test_patients_file=None, max_tiles_per_patient=None, feat_extractor='conch', cache_pickle_tiles_path='temp/tcga_crc.pkl', **kwargs)
        elif self.data_name == 'tcga_crc_224_uni2':
            return ImageTileDatasetWithFeatures(root_dirs=[self.data_path], features_dirs=[self.feat_path], test_patients_file=None, max_tiles_per_patient=None, feat_extractor='uni2', cache_pickle_tiles_path='temp/tcga_crc.pkl', **kwargs)
        else:
            raise NotImplementedError()

    def make_loader(self,
                    dataset,
                    shuffle: bool = False,
                    num_worker: bool = None,
                    drop_last: bool = True,
                    batch_size: int = None,
                    parallel: bool = False,
                    sampler = None
                    ):
        if parallel and distributed.is_initialized():
            # drop last to make sure that there is no added special indexes
            print("Parallel and distributed")
            sampler = DistributedSampler(dataset,
                                         shuffle=shuffle,
                                         drop_last=True)
        #else:
        #    sampler = None
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            sampler=sampler,
            # with sampler, use the sample instead of this option
            shuffle=False if sampler else shuffle,
            num_workers=num_worker or self.num_workers,
            pin_memory=True,
            drop_last=drop_last,
            multiprocessing_context=get_context('fork'),
        )

    def make_model_conf(self):
        if self.model_name == ModelName.beatgans_ddpm:
            self.model_type = ModelType.ddpm
            self.model_conf = BeatGANsUNetConfig(
                attention_resolutions=self.net_attn,
                channel_mult=self.net_ch_mult,
                conv_resample=True,
                dims=2,
                dropout=self.dropout,
                embed_channels=self.net_beatgans_embed_channels,
                image_size=self.img_size,
                in_channels=3,
                model_channels=self.net_ch,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.net_beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.net_beatgans_gradient_checkpoint,
                use_new_attention_order=False,
                resnet_two_cond=self.net_beatgans_resnet_two_cond,
                resnet_use_zero_module=self.
                net_beatgans_resnet_use_zero_module,
            )
        elif self.model_name in [
                ModelName.beatgans_autoenc,
        ]:
            cls = BeatGANsAutoencConfig
            # supports both autoenc and vaeddpm
            if self.model_name == ModelName.beatgans_autoenc:
                self.model_type = ModelType.autoencoder
            else:
                raise NotImplementedError()

            self.model_conf = cls(
                attention_resolutions=self.net_attn,
                channel_mult=self.net_ch_mult,
                conv_resample=True,
                dims=2,
                dropout=self.dropout,
                embed_channels=self.net_beatgans_embed_channels,
                enc_out_channels=self.style_ch,
                enc_pool=self.net_enc_pool,
                enc_num_res_block=self.net_enc_num_res_blocks,
                enc_channel_mult=self.net_enc_channel_mult,
                enc_grad_checkpoint=self.net_enc_grad_checkpoint,
                enc_attn_resolutions=self.net_enc_attn,
                image_size=self.img_size,
                in_channels=3,
                model_channels=self.net_ch,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.net_beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.net_beatgans_gradient_checkpoint,
                use_new_attention_order=False,
                resnet_two_cond=self.net_beatgans_resnet_two_cond,
                resnet_use_zero_module=self.
                net_beatgans_resnet_use_zero_module,
                resnet_cond_channels=self.net_beatgans_resnet_cond_channels,
            )
        else:
            raise NotImplementedError(self.model_name)

        return self.model_conf
    
def use_cached_dataset_path(source_path, cache_path):
    if get_rank() == 0:
        if not os.path.exists(cache_path):
            # shutil.rmtree(cache_path)
            print(f'copying the data: {source_path} to {cache_path}')
            shutil.copytree(source_path, cache_path)
    barrier()
    return cache_path

def denormalize_img(images, conf: TrainConfig):
    """
    Undo the normalization of an image tensor.
    """
    mean, std = conf.normalization_params
    assert images.ndim == 4, f"Expected 4D input, got {images.ndim}D"
    mean = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    images = images * std + mean
    return torch.clamp(images, 0, 1)

def normalize_img(images, conf: TrainConfig):
    """
    Apply the normalization to an image tensor.
    """
    mean, std = conf.normalization_params
    assert images.ndim == 4, f"Expected 4D input, got {images.ndim}D"
    mean = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    images = (images - mean) / std
    return images