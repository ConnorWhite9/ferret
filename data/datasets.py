"""Vision dataset wrappers and episode sample types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from data.config import AttackType, DataConfig, SplitName
from data.download import ensure_dataset_root


@dataclass(frozen=True)
class RawSample:
    """Clean image loaded from disk before episode sampling."""

    image: torch.Tensor
    label: int
    index: int
    relative_path: str


@dataclass(frozen=True)
class EpisodeSample:
    """Single episode starting state for the probing agent."""

    image: torch.Tensor
    clean_image: torch.Tensor
    label: int
    is_adversarial: bool
    attack_type: AttackType | None
    index: int
    relative_path: str


class VisionDataset(Dataset):
    """ImageFolder wrapper exposing stable paths for adversarial caching."""

    def __init__(self, config: DataConfig, split: SplitName, transform):
        root = ensure_dataset_root(config)
        split_root = root / split
        self._dataset = ImageFolder(split_root, transform=transform)
        self._root = split_root
        self.split = split

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> RawSample:
        image, label = self._dataset[index]
        absolute_path = self._dataset.samples[index][0]
        relative_path = str(Path(absolute_path).relative_to(self._root))
        return RawSample(
            image=image,
            label=label,
            index=index,
            relative_path=relative_path,
        )

    @property
    def class_to_idx(self) -> dict[str, int]:
        return self._dataset.class_to_idx
