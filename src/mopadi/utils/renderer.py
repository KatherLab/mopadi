# Code sourced from the Official implementation of Diffusion Autoencoders by Konpat Preechakul
# Original Source: https://github.com/phizaz/diffae
# License: MIT

from mopadi.configs.config import *
from torch import amp


def render_uncondition(conf: TrainConfig,
                       model: BeatGANsAutoencModel,
                       x_T,
                       sampler: Sampler):
    """
    Unconditional rendering without latent diffusion or conditioning on pre-extracted features.
    """
    # Ensure that the model is in the proper sampling mode
    device = x_T.device

    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.can_sample(), "Model type must support sampling."
        # Directly sample from the model using noise
        return sampler.sample(model=model, noise=x_T)
    else:
        raise NotImplementedError("Only diffusion train mode is supported in this setup.")

    

def render_condition(conf: TrainConfig,
                     model: BeatGANsAutoencModel,
                     x_T,
                     sampler: Sampler,
                     cond):
    """
    Generate samples conditioned on extracted features.
    """
    device = x_T.device

    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.has_autoenc()
        return sampler.sample(model=model, noise=x_T, cond=cond)
    else:
        raise NotImplementedError("Only diffusion mode is supported with precomputed features.")

