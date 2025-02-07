import math
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Tuple

import torch
from configs.choices import *
from configs.config_base import BaseConfig
from torch import nn
from torch.nn import init
from functools import cache

from huggingface_hub import login, whoami
from transformers import AutoModel
import torch

from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


class FeatureExtractor(Enum):
    conch = 'conch'
    virchow = 'virchow2'


class FeatureExtractorConch:
    def __init__(self, device='cpu'):

        #token = input("Enter your Hugging Face API token: ")
        #login(token=hf_token)
        user_info = whoami()
        print(f"Currently logged in as: {user_info['name']}")

        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        self.conch, self.eval_transform = titan.return_conch()
        self.conch.eval()

        if torch.cuda.is_available():
            self.conch = self.conch.to(device)
        self.device = device

        print(f"CONCH model successfully initialised on device {self.device}...\n")

    #@cache
    def extract_feats(self, imgs_batch: torch.Tensor):
        """
        Extracts features from WSI tiles using CONCH tile encoder.
        """
        #transformed_img = self.eval_transform(image)
        batch = imgs_batch.to(self.device)
        self.conch.eval()
        with torch.no_grad():
            features = self.conch(batch)
            assert len(features.shape) == 2 and features.shape[1] == 512, f"Unexpected feature shape: {features.shape}"

        return features

    
class FeatureExtractorVirchow:
    def __init__(self, device="cpu"):

        self.model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer="SwiGLUPacked",
            act_layer=torch.nn.SiLU,
        )
        self.model = self.model.eval().to(device)
        self.device = device

        print(f"Virchow2 model successfully initialised on device {self.device}...\n")

        # Define preprocessing transforms
        self.transforms = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )

    def extract_feats(self, imgs_batch):
        """
        Extract features from input images using Virchow tile encoder.

        Args:
            images (torch.Tensor): A batch of images, shape (B, C, H, W).

        Returns:
            torch.Tensor: Embeddings for the batch, shape (B, 2560).
        """

        # Apply preprocessing transforms
        if not isinstance(images, torch.Tensor):  # Handle PIL Images
            images = torch.stack([self.transforms(img) for img in images])

        # Forward pass through the model
        with torch.no_grad():
            output = self.model(images)  # shape: (B, 261, 1280)

        class_token = output[:, 0]  # shape: (B, 1280)
        patch_tokens = output[:, 5:]  # Ignore tokens 1-4 (register tokens)

        features = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)  # shape: (B, 2560)
        return features

