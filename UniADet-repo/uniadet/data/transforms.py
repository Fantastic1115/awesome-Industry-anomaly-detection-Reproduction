from __future__ import annotations

from typing import Callable

import torchvision.transforms as T
from PIL import Image


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(backbone: str, image_size: int) -> Callable[[Image.Image], object]:
    backbone = backbone.lower()
    if backbone.startswith("clip"):
        mean, std = CLIP_MEAN, CLIP_STD
        interp = T.InterpolationMode.BICUBIC
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        interp = T.InterpolationMode.BICUBIC

    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=interp),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def build_mask_transform(image_size: int) -> Callable[[Image.Image], object]:
    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ]
    )

