# Code snippets sourced from the Official implementation of Diffusion Autoencoders by Konpat Preechakul
# with modifications by Laura Zigutyte and Tim Lenz
# Original Source: https://github.com/phizaz/diffae
# License: MIT

import os
from typing import Tuple
from multiprocessing import get_context
from dataclasses import dataclass
import glob

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from mopadi.configs.config_base import BaseConfig
from mopadi.dataset import *
from mopadi.diffusion import *
from mopadi.diffusion.base import GenerativeType, LossType, ModelMeanType, ModelVarType, get_named_beta_schedule
from mopadi.model import *
from mopadi.configs.choices import *
from mopadi.model.unet import ScaleAt
from mopadi.model.extractor import *
from mopadi.diffusion.resample import UniformSampler
from mopadi.diffusion.diffusion import space_timesteps


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
    lambda_lp: float = 0.1
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
    net_beatgans_gradient_checkpoint: bool = False
    net_beatgans_resnet_two_cond: bool = False
    net_beatgans_resnet_use_zero_module: bool = True
    net_beatgans_resnet_scale_at: ScaleAt = ScaleAt.after_norm
    net_beatgans_resnet_cond_channels: int = None
    net_ch_mult: Tuple[int] = None
    net_ch: int = 64
    net_autoenc_stochastic: bool = False
    net_num_res_blocks: int = 2
    # number of resblocks for the UNET
    net_num_input_res_blocks: int = None
    num_workers: int = 6 # 20 for dgx
    parallel: bool = False
    postfix: str = ''
    sample_size: int = 64
    reconstruct_every_samples: int = 20_000
    save_every_samples: int = 100_000
    style_ch: int = 512   # conch v1.5 = 768
    T_eval: int = 1_000
    T_sampler: str = 'uniform'
    T: int = 1_000
    total_samples: int = 10_000_000
    steps_per_epoch: int = 5_000
    warmup: int = 0
    pretrain: PretrainConfig = None
    continue_from: PretrainConfig = None
    eval_programs: Tuple[str] = None
    # if present load the checkpoint from this path instead
    eval_path: str = None
    base_dir: str = 'checkpoints'
    use_cache_dataset: bool = False
    data_cache_dir: str = 'cache'
    work_cache_dir: str = 'mopadi_cache'
    load_pretrained_autoenc: bool = False
    # to be overridden
    name: str = ''


    def __post_init__(self):
        self.batch_size_eval = self.batch_size_eval or self.batch_size
        self.data_val_name = self.data_val_name or self.data_name

    def scale_up_gpus(self, num_gpus, num_nodes=1):
        self.eval_ema_every_samples *= num_gpus * num_nodes
        self.eval_every_samples *= num_gpus * num_nodes
        self.reconstruct_every_samples *= num_gpus * num_nodes
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
    def logdir(self):
        return f'{self.base_dir}/{self.name}'

    @property
    def generate_dir(self):
        return f'{self.work_cache_dir}/gen_images/{self.name}'


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

    def make_dataset(self, use_web_dataset=True, **kwargs):
        urls = expand_shards(self.data_dirs)
        if use_web_dataset:
            return WDSTilesWithFeatures(
                shards=urls,
                feature_dirs=self.feature_dirs,
                do_normalize=self.do_normalize,
                do_resize=self.do_resize,
                feat_extractor=self.feat_extractor,
                img_size=self.img_size,
            )
        else:
            if self.data_name == 'tcga_crc_512_conch_nolmdb' or self.data_name == 'tcga_brca_512_conch_nolmdb':
                return ImageTileDatasetWithFeatures(root_dirs=[self.data_path], features_dirs=self.feat_path, test_patients_file=self.test_patient_file, feat_extractor='conch', **kwargs)
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


    def make_loader(
        self,
        dataset,
        shuffle: bool = False,   # ignored for WebDataset; shuffling happens inside the pipeline
        num_worker: int = None,  # <-- int, not bool
        drop_last: bool = True,
        batch_size: int = None,
        parallel: bool = False,
        sampler=None,            # ignored for WebDataset
    ):
        if isinstance(dataset, (WDSTiles, WDSTilesWithFeatures)):
            return dataset.to_loader(
                batch_size=batch_size or self.batch_size,
                num_workers=num_worker or self.num_workers,
                steps_per_epoch=self.steps_per_epoch,
            )

        # Fallback: classic Dataset path (for non-WebDataset datasets)
        if parallel and distributed.is_initialized():
            print("Parallel and distributed")
            sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=True)
            shuffle = False  # sampler controls ordering

        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            sampler=sampler,
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
    
def expand_shards(paths_or_patterns):
    """Return a list of .tar URLs; accept dirs, globs, or explicit files."""
    urls = []
    for p in paths_or_patterns:
        if os.path.isdir(p):
            urls += sorted(glob.glob(os.path.join(p, "*.tar")))
        else:
            # allow patterns like "/data/shards/BRCA-*.tar" or brace lists
            matches = sorted(glob.glob(p))
            urls += matches if matches else [p]
    urls = [u for u in urls if u.endswith(".tar")]
    if not urls:
        raise ValueError("No .tar shards found from: " + ", ".join(paths_or_patterns))
    return urls
