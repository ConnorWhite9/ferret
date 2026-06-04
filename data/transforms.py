"""Image transforms for ResNet-50 target model inputs."""

from __future__ import annotations

from typing import Literal

import torch
import torchvision.transforms as T

from data.config import IMAGENET_MEAN, IMAGENET_STD

SplitName = Literal["train", "val"]


def build_transforms(image_size: int, split: SplitName) -> T.Compose:
    """Return torchvision transforms normalized for pretrained ResNet-50."""
    if split == "train":
        return T.Compose(
            [
                T.Resize(image_size),
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Map a normalized CHW tensor back to [0, 1] for foolbox bounds."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Map a [0, 1] CHW tensor to ImageNet-normalized space."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std
