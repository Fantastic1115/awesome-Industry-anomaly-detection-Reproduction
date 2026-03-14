from __future__ import annotations

from typing import Literal, Optional

import torch

from .hf_clip import HFCLIPBackbone
from .torchhub_dino import TorchHubDINOBackbone


BackboneName = Literal[
    "clip-vit-l14-336",
    "dinov2-vit-l16-reg",
    "dinov3-vit-l16",
]


def build_backbone(name: str, device: torch.device, image_size: Optional[int] = None):
    name = name.strip().lower()
    if name == "clip-vit-l14-336":
        return HFCLIPBackbone(
            model_name="openai/clip-vit-large-patch14-336",
            device=device,
            image_size=image_size,
        )
    if name == "dinov2-vit-l16-reg":
        return TorchHubDINOBackbone(repo="facebookresearch/dinov2", model_name="dinov2_vitl16_reg", device=device)
    if name == "dinov3-vit-l16":
        return TorchHubDINOBackbone(repo="facebookresearch/dinov3", model_name="dinov3_vitl16", device=device)
    raise ValueError(f"Unknown backbone: {name}")
