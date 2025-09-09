# Code snippets sourced from the Official implementation of Diffusion Autoencoders by Konpat Preechakul
# with modifications by Laura Zigutyte
# Original Source: https://github.com/phizaz/diffae
# License: MIT

import copy
import json
import os
import re
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *

import torch
from torch.amp import autocast
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torchvision.utils import make_grid, save_image

from mopadi.configs.config import *
from mopadi.dataset import *
from mopadi.utils.dist_utils import *
from mopadi.utils.metrics import *
from mopadi.utils.renderer import *
from mopadi.utils.misc import *
from mopadi.model.extractor import (
    FeatureExtractorConch, FeatureExtractorConch15,
    FeatureExtractorVirchow2, FeatureExtractorUNI2
)

torch.set_float32_matmul_precision('medium')


class LitModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode != TrainMode.manipulate
        if conf.seed is not None:
            pl.seed_everything(conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())

        self.conf = conf

        self.model = conf.make_model_conf().make_model()
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        # load and initialize pretrained feature extractor
        self.feature_extractor = None

        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        #print('Model params: %.2f M' % (model_size / 1024 / 1024))

        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()
        self.T_sampler = conf.make_T_sampler()

        # initial variables for consistent sampling
        self.register_buffer(
            'x_T',
            torch.randn(conf.sample_size, 3, conf.img_size, conf.img_size))
        
        pretrained_path = os.path.join(conf.base_dir, 'autoenc', 'last.ckpt')
        if conf.load_pretrained_autoenc:
            if os.path.exists(pretrained_path):
                print(f'Loading pretrained model from {pretrained_path}')
                state = torch.load(pretrained_path, map_location='cpu')
                print('step:', state['global_step'])
                self.load_state_dict(state['state_dict'], strict=False)
            else:
                raise FileNotFoundError(f"Pretrained autoencoder checkpoint not found at {pretrained_path}")

        latent_infer_path = os.path.join(conf.base_dir, 'features.pkl')
        if os.path.exists(latent_infer_path):
            print('Loading pre-extracted features of the encoder...')
            state = torch.load(latent_infer_path)
            self.conds = state['conds']
            self.register_buffer('conds_mean', state['conds_mean'][None, :])
            self.register_buffer('conds_std', state['conds_std'][None, :])
        else:
            self.conds_mean = None
            self.conds_std = None

    def render(self, noise, cond, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()

        pred_img = render_condition(self.conf,
                                    self.ema_model,
                                    noise,
                                    sampler=sampler,
                                    cond=cond)
        
        pred_img = (pred_img + 1) / 2
        return pred_img

    def encode_stochastic(self, x, cond, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
        out = sampler.ddim_reverse_sample_loop(self.ema_model,
                                               x,
                                               model_kwargs={'cond': cond})
        return out['sample']

    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        ##############################################

        self.train_data = self.conf.make_dataset()
        self.val_data = self.train_data

        # initialize Feature Extractor after the device is set
        if self.feature_extractor is None: 
            if self.conf.feat_extractor == 'conch':
                self.feature_extractor = FeatureExtractorConch(device=self.device)
            elif self.conf.feat_extractor == 'conch1_5':
                self.feature_extractor = FeatureExtractorConch15(device=self.device)
            elif self.conf.feat_extractor == 'v2':
                self.feature_extractor = FeatureExtractorVirchow2(device=self.device)
            elif self.conf.feat_extractor == 'uni2':
                self.feature_extractor = FeatureExtractorUNI2(device=self.device)

        self.model.feature_extractor = self.feature_extractor
        self.ema_model.feature_extractor = self.feature_extractor

    def on_fit_start(self):
        # Make sure the extractor is on the same device as the model
        if self.feature_extractor is not None:
            if hasattr(self.feature_extractor, "model"):
                self.feature_extractor.model = self.feature_extractor.model.to(self.device)
            if hasattr(self.feature_extractor, "device"):
                self.feature_extractor.device = self.device
        # reattach in case of rewraps
        self.model.feature_extractor = self.feature_extractor
        self.ema_model.feature_extractor = self.feature_extractor

        if not self.conf.load_pretrained_autoenc:
            print("\n=== SANITY CHECK: Checking first batch from DataLoader ===")
            dl = self.train_dataloader()
            batch = next(iter(dl))
            if isinstance(batch, dict):
                for k, v in batch.items():
                    print(f"  {k}: shape={getattr(v, 'shape', None)}, dtype={getattr(v, 'dtype', None)}")
            elif isinstance(batch, (tuple, list)):
                print("  tuple/list:", [getattr(b, 'shape', None) for b in batch])
            else:
                print("  type:", type(batch))

            if 'coords' in batch:
                coords = batch['coords'] 
                if (coords == -1).all(dim=1).any():
                    print("[WARNING] Some or all filenames do not contain coordinates (or they were not extracted correctly)! Using dummy value [-1, -1] for tiles without coords.")
            print("=== END SANITY CHECK ===\n")

            self.sanity_check_precomputed_feats(n_batches=1)

    def train_dataloader(self):
        """
        return the dataloader, if diffusion mode => return image dataset
        """
        # make sure to use the fraction of batch size
        # the batch size is global!
        conf = self.conf.clone()
        conf.batch_size = self.batch_size

        dataloader = conf.make_loader(self.train_data,
                                      shuffle=True,
                                      drop_last=True)
        return dataloader

    @property
    def batch_size(self):
        """
        local batch size for each worker
        """
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws

    @property
    def num_samples(self):
        """
        (global) batch size * iterations
        """
        # batch size here is global!
        # global_step already takes into account the accum batches
        return self.global_step * self.conf.batch_size_effective

    def is_last_accum(self, batch_idx):
        """
        is it the last gradient accumulation loop? 
        used with gradient_accum > 1 and to see if the optimizer will perform "step" in this iteration or not
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0

    def training_step(self, batch, batch_idx):
        """
        given an input, calculate the loss function
        no optimization at this stage.
        """
        with autocast(device_type='cuda', enabled=False):

            imgs = batch['img']
            feats = batch['feat']
            model_kwargs = {'cond': feats}

            if self.conf.train_mode == TrainMode.diffusion:
                """
                Main training mode for diffusion models (using precomputed features).
                """
                # with numpy seed we have the problem that the sample t's are related!
                t, _ = self.T_sampler.sample(len(imgs), imgs.device)
                losses = self.sampler.training_losses(
                    model=self.model, 
                    x_start=imgs,
                    cond=feats,
                    t=t,
                    model_kwargs=model_kwargs
                )
            else:
                raise NotImplementedError()

            loss = losses['loss'].mean()
            # divide by accum batches to make the accumulated gradient exact!
            for key in ['loss', 'vae', 'mmd', 'chamfer', 'arg_cnt']:
                if key in losses:
                    losses[key] = self.all_gather(losses[key]).mean()

            if self.global_rank == 0:
                self.logger.experiment.add_scalar('loss', losses['loss'], self.num_samples)
                for key in ['vae', 'mmd', 'chamfer', 'arg_cnt']:
                    if key in losses:
                        self.logger.experiment.add_scalar(f'loss/{key}', losses[key], self.num_samples)

        return {'loss': loss}

    def on_train_batch_end(self, outputs, batch, batch_idx: int,) -> None: # dataloader_idx: int
        """
        after each training step ...
        """
        if self.is_last_accum(batch_idx):
            # only apply ema on the last gradient accumulation step,
            # if it is the iteration that has optimizer.step()
            ema(self.model, self.ema_model, self.conf.ema_decay)

            # logging
            imgs = batch['img']
            conds = batch['feat']
            self.log_sample(x_start=imgs, cond=conds)
            self.evaluate_scores()

    def on_before_optimizer_step(self, optimizer: Optimizer,) -> None:
        # optimizer_idx: int
        # fix the fp16 + clip grad norm problem with pytorch lightinng
        # this is the currently correct way to do it
        if self.conf.grad_clip > 0:
            # from trainer.params_grads import grads_norm, iter_opt_params
            params = [
                p for group in optimizer.param_groups for p in group['params']
            ]
            # print('before:', grads_norm(iter_opt_params(optimizer)))
            torch.nn.utils.clip_grad_norm_(params,
                                           max_norm=self.conf.grad_clip)
            # print('after:', grads_norm(iter_opt_params(optimizer)))

    def log_sample(self, x_start, cond):
        """
        put images to the tensorboard
        """
        def do(model, postfix, use_xstart, save_real=False, cond=None):

            model.eval()

            with torch.no_grad():
                all_x_T = self.split_tensor(self.x_T)

                if use_xstart:
                    all_x_T = all_x_T[:len(x_start)]

                batch_size = min(len(all_x_T), self.conf.batch_size_eval)
                # allow for superlarge models
                loader = DataLoader(all_x_T, batch_size=batch_size)

                Gen, Reals = [], []
                offset = 0

                for x_T in loader:
                    n = x_T.size(0)
                    if use_xstart:
                        _xstart = x_start[offset: offset + n]
                        _cond = cond[offset: offset + n] if cond is not None else None
                        offset += n
                    else:
                        _xstart = None
                        _cond = cond

                    if _xstart is not None:
                        assert cond is not None, "Features missing for a given xstart img"

                    gen = self.eval_sampler.sample(model=model,
                                                    noise=x_T,
                                                    cond=_cond,
                                                    x_start=_xstart)
                    Gen.append(gen)

                    if save_real and use_xstart:
                        Reals.append(_xstart)

                gen = torch.cat(Gen)
                gen = self.all_gather(gen)
                if gen.dim() == 5:
                    # (n, c, h, w)
                    gen = gen.flatten(0, 1)

                if save_real and use_xstart:
                    # save the original images to the tensorboard
                    real = torch.cat(Reals, dim=0)
                    real = self.all_gather(real)
                    if real.dim() == 5:
                        real = real.flatten(0, 1)

                    if self.global_rank == 0:
                        grid_real = (make_grid(real) + 1) / 2
                        sample_dir = os.path.join(self.conf.logdir, f'sample_real{postfix}')
                        if not os.path.exists(sample_dir):
                            os.makedirs(sample_dir)
                        path = os.path.join(sample_dir, f'{self.num_samples}.png')
                        save_image(grid_real, path)
                        self.logger.experiment.add_image(f'sample{postfix}/real', grid_real, self.num_samples)

                if self.global_rank == 0:
                    # save samples to the tensorboard
                    grid = (make_grid(gen) + 1) / 2
                    sample_dir = os.path.join(self.conf.logdir, f'sample{postfix}')
                    if not os.path.exists(sample_dir):
                        os.makedirs(sample_dir)
                    path = os.path.join(sample_dir, '%d.png' % self.num_samples)
                    save_image(grid, path)
                    self.logger.experiment.add_image(f'sample{postfix}', grid, self.num_samples)
            model.train()

        if self.conf.sample_every_samples > 0 and is_time(
                self.num_samples, self.conf.sample_every_samples,
                self.conf.batch_size_effective):

            do(self.model, '', use_xstart=True, save_real=True, cond=cond)
            do(self.ema_model, '_ema', use_xstart=True, save_real=True, cond=cond)

    def evaluate_scores(self):
        """
        evaluate FID and other scores during training (put to the tensorboard)
        For, FID. It is a fast version with 5k images (gold standard is 50k).
        Don't use its results in the paper!
        """
        def fid(model, postfix):
            score = evaluate_fid(self.eval_sampler,
                                 model,
                                 self.conf,
                                 device=self.device,
                                 train_data=self.train_data,
                                 val_data=self.val_data,
                                 )

            if self.global_rank == 0:
                self.logger.experiment.add_scalar(f'FID{postfix}', score, self.num_samples)
                if not os.path.exists(self.conf.logdir):
                    os.makedirs(self.conf.logdir)
                with open(os.path.join(self.conf.logdir, 'eval.txt'),
                          'a') as f:
                    metrics = {
                        f'FID{postfix}': score,
                        'num_samples': self.num_samples,
                    }
                    f.write(json.dumps(metrics) + "\n")

        def lpips(model, postfix):
            if self.conf.model_type.has_autoenc(
            ) and self.conf.train_mode.is_autoenc():
                # {'lpips', 'ssim', 'mse'}
                score = evaluate_lpips(self.eval_sampler,
                                       model,
                                       self.conf,
                                       device=self.device,
                                       val_data=self.val_data,
                                       )

                if self.global_rank == 0:
                    for key, val in score.items():
                        self.logger.experiment.add_scalar(
                            f'{key}{postfix}', val, self.num_samples)
                if not os.path.exists(self.conf.logdir):
                    os.makedirs(self.conf.logdir)
                with open(os.path.join(self.conf.logdir, 'eval.txt'),
                        'a') as f:
                    metrics = {
                        f'LPIPS{postfix}': score,
                        'num_samples': self.num_samples,
                    }
                    f.write(json.dumps(metrics) + "\n")

        if self.conf.eval_every_samples > 0 and self.num_samples > 0 and is_time(
                self.num_samples, self.conf.eval_every_samples,
                self.conf.batch_size_effective):
            print(f'eval fid @ {self.num_samples}')
            lpips(self.model, '')
            fid(self.model, '')

        if self.conf.eval_ema_every_samples > 0 and self.num_samples > 0 and is_time(
                self.num_samples, self.conf.eval_ema_every_samples,
                self.conf.batch_size_effective):
            print(f'eval fid ema @ {self.num_samples}')
            fid(self.ema_model, '_ema')
            # it's too slow
            # lpips(self.ema_model, '_ema')

    def configure_optimizers(self):
        out = {}
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(self.model.parameters(),
                                    lr=self.conf.lr,
                                    weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(self.model.parameters(),
                                    lr=self.conf.lr,
                                    betas=(0.9, 0.99),
                                    eps=1e-06,
                                    weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.lion:
            from lion_pytorch import Lion
            optim = Lion(self.model.parameters(), 
                                    lr=self.conf.lr, 
                                    betas=(0.95, 0.98),
                                    weight_decay=self.conf.weight_decay)
        else:
            raise NotImplementedError()
            
        out['optimizer'] = optim

        sched = None
        if self.conf.optimizer == OptimizerType.lion:
            cosine = CosineAnnealingWarmRestarts(
                optim,
                T_0=20_000,     # in steps since we set interval='step'
                T_mult=2,
                eta_min=1e-6
            )
            if self.conf.warmup > 0:
                # warmup for Lion, then cosine
                warmup = LambdaLR(optim, lr_lambda=lambda s: min(s + 1, self.conf.warmup) / self.conf.warmup)
                sched = SequentialLR(optim, schedulers=[warmup, cosine], milestones=[self.conf.warmup])
            else:
                sched = cosine
        else:
            # Adam / AdamW: warmup -> Cosine (no restarts)
            total_steps = max(1, self.conf.total_samples // self.conf.batch_size_effective)
            if self.conf.warmup > 0:
                warmup = LambdaLR(optim, lr_lambda=lambda s: min(s + 1, self.conf.warmup) / self.conf.warmup)
                cosine = CosineAnnealingLR(optim, T_max=max(1, total_steps - self.conf.warmup), eta_min=1e-6)
                sched = SequentialLR(optim, schedulers=[warmup, cosine], milestones=[self.conf.warmup])
            else:
                cosine = CosineAnnealingLR(optim, T_max=total_steps, eta_min=1e-6)
                sched = cosine

        if sched is not None:
            out["lr_scheduler"] = {"scheduler": sched, "interval": "step"}

        return out
    
    def sanity_check_precomputed_feats(
        self,
        n_batches: int = 1,
        atol: float = 5e-4,          # absolute tolerance for MAE
        cos_thresh: float = 0.999,   # min cosine similarity per-batch
    ):
        if get_rank() != 0:
            return
        
        self.model.eval()
        dl = self.train_dataloader()
        it = iter(dl)

        print("\n=== FEATURE SANITY CHECK ===")
        bad = 0
        with torch.no_grad():
            for bi in range(n_batches):
                try:
                    batch = next(it)
                except StopIteration:
                    break

                imgs = batch["img"].to(self.device)
                feats_pre = batch["feat"].to(self.device).float()

                # Bring images to [0,1] for the extractor
                imgs01 = imgs
                if imgs01.min() < -0.1:            # typically [-1,1]
                    imgs01 = ((imgs01 + 1) / 2).clamp(0, 1)
                elif imgs01.max() > 1.5:           # uint8 0..255
                    imgs01 = (imgs01 / 255.0).clamp(0, 1)

                feats_new = self.feature_extractor.extract_feats(imgs01, need_grad=False).float()

                if feats_new.shape != feats_pre.shape:
                    print(f"[FEAT SANITY] Shape mismatch: on-the-fly {tuple(feats_new.shape)} "
                        f"vs precomputed {tuple(feats_pre.shape)}")
                    bad += 1
                    continue

                mae = (feats_new - feats_pre).abs().mean().item()
                maxe = (feats_new - feats_pre).abs().max().item()
                cos = torch.nn.functional.cosine_similarity(feats_new, feats_pre, dim=1)
                cos_mean = cos.mean().item()
                cos_min = cos.min().item()

                print(f"[FEAT SANITY] batch {bi}: MAE={mae:.3e}  MAX={maxe:.3e}  "
                    f"COS(mean)={cos_mean:.6f}  COS(min)={cos_min:.6f}")

                # Flag batch as bad if either numeric error is too large or cosine too low
                if mae > atol or cos_min < cos_thresh:
                    bad += 1

        if bad == 0:
            print("[FEAT SANITY] OK: precomputed and on-the-fly features match within tolerance.\n")
        else:
            print(f"[FEAT SANITY] WARNING: {bad} batch(es) exceeded thresholds. "
                "Check extractor variant, resize, and normalization.\n")

        self.model.train()

    def split_tensor(self, x):
        """
        extract the tensor for a corresponding "worker" in the batch dimension

        Args:
            x: (n, c)

        Returns: x: (n_local, c)
        """
        n = len(x)
        rank = self.global_rank
        world_size = get_world_size()
        # print(f'rank: {rank}/{world_size}')
        per_rank = n // world_size
        return x[rank * per_rank:(rank + 1) * per_rank]

    def test_step(self, batch, *args, **kwargs):
        """
        for the "eval" mode. 
        We first select what to do according to the "conf.eval_programs". 
        test_step will only run for "one iteration" (it's a hack!).
        
        We just want the multi-gpu support. 
        """
        # make sure you seed each worker differently!
        # it will run only one step!
        print('global step:', self.global_step)

        # evals those "fidXX"
        """
        "fid<T>" = unconditional generation (conf.train_mode = diffusion).
            Note:   Diff. autoenc will still receive real images in this mode.
        """
        for each in self.conf.eval_programs:
            if each.startswith('fid'):

                # evalT
                _, T = each.split('fid')
                T = int(T)
                print(f'evaluating FID T = {T}...')

                self.train_dataloader()
                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                conf.eval_num_images = 50_000
                score_gen, score_rec = evaluate_fid(
                    sampler,
                    self.ema_model,
                    conf,
                    device=self.device,
                    train_data=self.train_data,
                    val_data=self.val_data,
                    conds_mean=self.conds_mean,
                    conds_std=self.conds_std,
                    remove_cache=False,
                    #clip_latent_noise=clip_latent_noise,
                )

                self.log(f'fid_ema_T{T}_gen', score_gen)
                self.log(f'fid_ema_T{T}_rec', score_rec)

        """
        "recon<T>" = reconstruction & autoencoding (without noise inversion)
        """
        for each in self.conf.eval_programs:
            if each.startswith('recon'):
                self.model: BeatGANsAutoencModel
                _, T = each.split('recon')
                T = int(T)
                print(f'evaluating reconstruction T = {T}...')

                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                # eval whole val dataset
                conf.eval_num_images = len(self.val_data)
                # {'lpips', 'mse', 'ssim'}
                score = evaluate_lpips(sampler,
                                       self.ema_model,
                                       conf,
                                       device=self.device,
                                       val_data=self.val_data,
                                       )
                for k, v in score.items():
                    self.log(f'{k}_ema_T{T}', v)
        """
        "inv<T>" = reconstruction with noise inversion
        """
        for each in self.conf.eval_programs:
            if each.startswith('inv'):
                self.model: BeatGANsAutoencModel
                _, T = each.split('inv')
                T = int(T)
                print(
                    f'evaluating reconstruction with noise inversion T = {T}...'
                )

                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                # eval whole val dataset
                conf.eval_num_images = len(self.val_data)
                # {'lpips', 'mse', 'ssim'}
                score = evaluate_lpips(sampler,
                                       self.ema_model,
                                       conf,
                                       device=self.device,
                                       val_data=self.val_data,
                                       use_inverted_noise=True
                                       )
                for k, v in score.items():
                    self.log(f'{k}_inv_ema_T{T}', v)


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup


def is_time(num_samples, every, step_size):
    closest = (num_samples // every) * every
    return num_samples - closest < step_size


def train(conf: TrainConfig, gpus, nodes=1, mode: str = 'train'):
    model = LitModel(conf)

    if get_rank() == 0:  # Ensure only the main worker creates the directory
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)

    checkpoint = ModelCheckpoint(dirpath=f'{conf.logdir}',
                                 save_last=True,
                                 save_top_k=1,
                                 every_n_train_steps=conf.save_every_samples //
                                 conf.batch_size_effective)

    checkpoint_model_path = os.path.join(conf.logdir, 'last.ckpt')
    print('Checkpoint model path:', checkpoint_model_path)
    if os.path.exists(checkpoint_model_path):
        resume = True
    else:
        if conf.continue_from is not None and os.path.exists(conf.continue_from):
            # continue from a checkpoint
            checkpoint_model_path = conf.continue_from
            resume = True
        else:
            resume = False

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir,
                                             name=None,
                                             version='')

    strategy=None
    plugins = []
    if len(gpus) == 1 and nodes == 1:
        accelerator = 'gpu'
        strategy='auto'
    elif len(gpus) > 1:
        """ 
        # older pytorch-lightning version (e.g. 2.0.6)
        accelerator = 'ddp'
        from pytorch_lightning.plugins import DDPPlugin
        # important for working with gradient checkpoint
        plugins.append(DDPPlugin(find_unused_parameters=True))
        """
        from pytorch_lightning.strategies import DDPStrategy     # pytorch-lightning version 2.1.1
        strategy = DDPStrategy(find_unused_parameters=True)
        # strategy = 'ddp'
        accelerator = 'gpu'
    else:
        accelerator = 'cpu'

    print(f'Accelerator: {accelerator}, strategy: {strategy}, devices: {gpus}, num nodes: {nodes}')

    trainer = pl.Trainer(
        max_steps=conf.total_samples // conf.batch_size_effective,
        # resume_from_checkpoint=checkpoint_model_path,  # older pytorch-lightning version (e.g. 2.0.6)
        # gpus=gpus,                               # older pytorch-lightning version (e.g. 2.0.6)
        devices=gpus,                              # only for newer pytorch-lightning versions (>2.1.1)
        strategy=strategy,
        num_nodes=nodes,
        accelerator=accelerator,
        precision="16-mixed" if conf.fp16 else 32,
        callbacks=[
            checkpoint,
            LearningRateMonitor(),
        ],
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        plugins=plugins,
    )

    if mode == 'train':
        if resume:
            trainer.fit(model, ckpt_path=checkpoint_model_path)
        else:
            trainer.fit(model)
    elif mode == 'eval':
        # load the latest checkpoint
        # perform lpips
        # dummy loader to allow calling "test_step"
        dummy = DataLoader(TensorDataset(torch.tensor([0.] * conf.batch_size)),
                           batch_size=conf.batch_size)
        print('Loading from:', checkpoint_model_path)
        state = torch.load(checkpoint_model_path, map_location='cpu')
        print('Step:', state['global_step'])
        model.load_state_dict(state['state_dict'])
        # trainer.fit(model)
        out = trainer.test(model, dataloaders=dummy)
        # first (and only) loader
        out = out[0]
        print(out)

        if get_rank() == 0:
            # save to tensorboard
            for k, v in out.items():
                tb_logger.experiment.add_scalar(
                    k, v, state['global_step'] * conf.batch_size_effective)

            # # save to file
            # # make it a dict of list
            # for k, v in out.items():
            #     out[k] = [v]
            tgt = f'evals/{conf.name}.txt'
            dirname = os.path.dirname(tgt)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            with open(tgt, 'a') as f:
                f.write(json.dumps(out) + "\n")
            # pd.DataFrame(out).to_csv(tgt)
    else:
        raise NotImplementedError()
