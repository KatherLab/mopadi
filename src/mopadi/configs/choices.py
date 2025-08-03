# Code snippets sourced from the Official implementation of Diffusion Autoencoders by Konpat Preechakul
# Original Source: https://github.com/phizaz/diffae
# License: MIT

from enum import Enum
from torch import nn


class TrainMode(Enum):
    # manipulate mode = training the classifier
    manipulate = 'manipulate'
    # default training mode!
    diffusion = 'diffusion'

    def is_manipulate(self):
        return self in [
            TrainMode.manipulate,
        ]

    def is_diffusion(self):
        return self in [
            TrainMode.diffusion,
        ]

    def is_autoenc(self):
        # the network possibly does autoencoding
        return self in [
            TrainMode.diffusion,
        ]


class ModelType(Enum):
    """
    Kinds of the backbone models
    """

    # unconditional ddpm
    ddpm = 'ddpm'
    # autoencoding ddpm cannot do unconditional generation
    autoencoder = 'autoencoder'

    def has_autoenc(self):
        return self in [
            ModelType.autoencoder,
        ]

    def can_sample(self):
        return self in [ModelType.ddpm]


class ModelName(Enum):
    """
    List of all supported model classes
    """

    beatgans_ddpm = 'beatgans_ddpm'
    beatgans_autoenc = 'beatgans_autoenc'


class ModelMeanType(Enum):
    """
    Which type of output the model predicts.
    """

    eps = 'eps'  # the model predicts epsilon


class ModelVarType(Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    # posterior beta_t
    fixed_small = 'fixed_small'
    # beta_t
    fixed_large = 'fixed_large'


class LossType(Enum):
    mse = 'mse'  # use raw MSE loss (and KL when learning variances)
    l1 = 'l1'


class GenerativeType(Enum):
    """
    How's a sample generated
    """

    ddpm = 'ddpm'
    ddim = 'ddim'


class OptimizerType(Enum):
    adam = 'adam'
    adamw = 'adamw'
    lion = 'lion'


class Activation(Enum):
    none = 'none'
    relu = 'relu'
    lrelu = 'lrelu'
    silu = 'silu'
    tanh = 'tanh'

    def get_act(self):
        if self == Activation.none:
            return nn.Identity()
        elif self == Activation.relu:
            return nn.ReLU()
        elif self == Activation.lrelu:
            return nn.LeakyReLU(negative_slope=0.2)
        elif self == Activation.silu:
            return nn.SiLU()
        elif self == Activation.tanh:
            return nn.Tanh()
        else:
            raise NotImplementedError()


class ManipulateLossType(Enum):
    bce = 'bce'
    mse = 'mse'
