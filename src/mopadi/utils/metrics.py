# Code sourced from the Official implementation of Diffusion Autoencoders by Konpat Preechakul
# Original Source: https://github.com/phizaz/diffae
# License: MIT

import os
import shutil
from multiprocessing import get_context
from tqdm import tqdm, trange
import lpips
import math

import torchvision
from pytorch_fid import fid_score

import torch
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from mopadi.utils.misc import render_condition
from mopadi.configs.config import *
from mopadi.diffusion import Sampler
from mopadi.utils.dist_utils import *
from mopadi.utils.ssim import ssim


def make_subset_loader(conf: TrainConfig,
                       dataset: Dataset,
                       batch_size: int,
                       shuffle: bool,
                       parallel: bool,
                       drop_last=True):
    """
    Build a loader that yields up to conf.eval_num_images examples.

    - For WebDataset-based iterables (WDSTiles / WDSTilesWithFeatures), we set an
      epoch length in *batches* so it stops after ~eval_num_images.
    - For map-style datasets, we wrap with SubsetDataset and use a sampler if needed.
    """

    # for WebDataset / IterableDataset
    if isinstance(dataset, IterableDataset):
        # Prefer using the dataset's own WebLoader to keep the pipeline intact.
        steps = max(1, math.ceil(conf.eval_num_images / batch_size))
        # Note: shuffle is handled inside the WebDataset pipeline; DataLoader shuffle must stay False.
        return dataset.to_loader(
            batch_size=batch_size,
            num_workers=conf.num_workers,
            steps_per_epoch=steps,   # this limits to ~eval_num_images
        )

    # for Map-style dataset
    # Cap the requested size by the dataset length
    size = min(conf.eval_num_images, len(dataset))
    subset = SubsetDataset(dataset, size=size)

    if parallel and distributed.is_initialized():
        sampler = DistributedSampler(subset, shuffle=shuffle, drop_last=drop_last)
        do_shuffle = False  # sampler controls order
    else:
        sampler = None
        do_shuffle = shuffle

    return DataLoader(
        subset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=do_shuffle,
        num_workers=conf.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        multiprocessing_context=get_context('fork'),
    )


def evaluate_lpips(
    sampler: Sampler,
    model: Model,
    conf: TrainConfig,
    device,
    val_data: Dataset,
    use_inverted_noise: bool = False,
):
    """
    compare the generated images from autoencoder on validation dataset

    Args:
        use_inversed_noise: the noise is also inverted from DDIM
    """
    lpips_fn_alex = lpips.LPIPS(net='alex').to(device)
    val_loader = make_subset_loader(conf,
                                    dataset=val_data,
                                    batch_size=conf.batch_size_eval,
                                    shuffle=False,
                                    parallel=True)

    model.eval()
    with torch.no_grad():
        scores = {
            'lpips_alex': [],
            'fm_mse': [],
            'mse': [],
            'ssim': [],
            'psnr': [],
        }
        for batch in tqdm(val_loader, desc='lpips'):
            imgs = batch['img'].to(device)
            cond = batch['feat'].to(device)

            if use_inverted_noise:
                # inverse the noise
                # with condition from the encoder
                x_T = sampler.ddim_reverse_sample_loop(
                    model=model,
                    x=imgs,
                    clip_denoised=True,
                    model_kwargs={'cond': cond})
                x_T = x_T['sample']
            else:
                x_T = torch.randn((len(imgs), 3, conf.img_size, conf.img_size), device=device)

            pred_imgs = render_condition(conf=conf,
                                         model=model,
                                         x_T=x_T,
                                         cond=cond,
                                         sampler=sampler)

            scores['lpips_alex'].append(lpips_fn_alex.forward(imgs, pred_imgs).view(-1))

            norm_pred_imgs = (pred_imgs + 1) / 2   # converts from [-1,1] → [0,1]
            recon_feats = model.feat_extractor.extract_feats(norm_pred_imgs)

            lpips_custom = torch.nn.functional.mse_loss(cond, recon_feats, reduction='none').mean(dim=1)
            scores['fm_mse'].append(lpips_custom)

            norm_imgs = (imgs + 1) / 2
            
            # (n, )
            scores['ssim'].append(
                ssim(norm_imgs, norm_pred_imgs, size_average=False))
            # (n, )
            scores['mse'].append(
                (norm_imgs - norm_pred_imgs).pow(2).mean(dim=[1, 2, 3]))
            # (n, )
            scores['psnr'].append(psnr(norm_imgs, norm_pred_imgs))

        # (N, )
        for key in scores.keys():
            scores[key] = torch.cat(scores[key]).float()
    model.train()

    barrier()

    # support multi-gpu
    outs = {
        key: [
            torch.zeros(len(scores[key]), device=device)
            for i in range(get_world_size())
        ]
        for key in scores.keys()
    }
    for key in scores.keys():
        all_gather(outs[key], scores[key])

    # final scores
    for key in scores.keys():
        scores[key] = torch.cat(outs[key]).mean().item()

    return scores


def psnr(img1, img2):
    """
    Args:
        img1: (n, c, h, w)
    """
    v_max = 1.
    # (n,)
    mse = torch.mean((img1 - img2)**2, dim=[1, 2, 3])
    return 20 * torch.log10(v_max / torch.sqrt(mse))


def evaluate_fid(
    sampler: Sampler,
    model: Model,
    conf: TrainConfig,
    device,
    train_data: Dataset,
    val_data: Dataset,
    conds_mean=None,
    conds_std=None,
    remove_cache: bool = False,
):
    assert conf.fid_cache is not None
    gen_dir = os.path.join(conf.work_cache_dir, conf.base_dir.split('/')[-1], 'gen_from_noise_and_cond')

    if get_rank() == 0:
        # no parallel
        # validation data for a comparing FID
        val_loader = make_subset_loader(conf,
                                        dataset=val_data,
                                        batch_size=conf.batch_size_eval,
                                        shuffle=False,
                                        parallel=False)

        # put the val images to a directory
        cache_dir = f'{conf.fid_cache}_{conf.eval_num_images}'
        if (os.path.exists(cache_dir)
                and len(os.listdir(cache_dir)) < conf.eval_num_images):
            shutil.rmtree(cache_dir)

        if not os.path.exists(cache_dir):
            # write files to the cache
            # the images are normalized, hence need to denormalize first
            loader_to_path(val_loader, cache_dir, denormalize=True)

        # create the generate dir
        if os.path.exists(gen_dir):
            shutil.rmtree(gen_dir)
        os.makedirs(gen_dir)

    barrier()

    world_size = get_world_size()
    rank = get_rank()
    batch_size = chunk_size(conf.batch_size_eval, rank, world_size)

    def filename(idx):
        return world_size * idx + rank

    model.eval()
    with torch.no_grad():
        if conf.model_type == ModelType.autoencoder:
            # evaluate autoencoder (given the cond & random noise)
            # to make the FID fair, autoencoder must not see the validation dataset
            # also shuffle to make it closer to unconditional generation
            train_loader = make_subset_loader(conf,
                                                dataset=train_data,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                parallel=True)

            i = 0
            for batch in tqdm(train_loader, desc='reconstructing images from noise & cond'):
                imgs = batch['img'].to(device)
                cond = batch['feat'].to(device)

                x_T = torch.randn(
                    (len(imgs), 3, conf.img_size, conf.img_size),
                    device=device)
                batch_images = render_condition(
                    conf=conf,
                    model=model,
                    x_T=x_T,
                    cond=cond,
                    sampler=sampler).cpu()

                # denormalize the images
                batch_images = (batch_images + 1) / 2
                # keep the generated images
                for j in range(len(batch_images)):
                    img_name = filename(i + j)
                    torchvision.utils.save_image(
                        batch_images[j],
                        os.path.join(gen_dir, f'{img_name}.png'))
                i += len(imgs)

    model.train()

    barrier()

    if get_rank() == 0:
        fid = fid_score.calculate_fid_given_paths(
            [cache_dir, gen_dir],
            batch_size,
            device=device,
            dims=2048)

        # remove the cache
        if remove_cache:
            shutil.rmtree(gen_dir)

    barrier()

    if get_rank() == 0:
        # need to float it! unless the broadcasted value is wrong
        fid = torch.tensor(float(fid), device=device)
        broadcast(fid, 0)
    else:
        fid = torch.tensor(0., device=device)
        broadcast(fid, 0)
    fid = fid.item()
    print(f'FID reconstructed images from noise & cond ({get_rank()}): {fid}')  

    return fid


def loader_to_path(loader: DataLoader, path: str, denormalize: bool):
    # not process safe!

    if not os.path.exists(path):
        os.makedirs(path)

    # write the loader to files
    i = 0
    for batch in tqdm(loader, desc='copy images'):
        imgs = batch['img']
        if denormalize:
            imgs = (imgs + 1) / 2
        for j in range(len(imgs)):
            torchvision.utils.save_image(imgs[j],
                                         os.path.join(path, f'{i+j}.png'))
        i += len(imgs)
