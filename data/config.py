"""Configuration for Ferret vision data loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

AttackType = Literal["fgsm", "pgd", "cw"]
DatasetName = Literal["imagenette", "imagenet"]
SplitName = Literal["train", "val"]

DEFAULT_ATTACK_TYPES: tuple[AttackType, ...] = ("fgsm", "pgd", "cw")

# ResNet-50 / ImageNet normalization used by torchvision pretrained weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    """Runtime configuration for the vision data pipeline."""

    dataset: DatasetName = "imagenette"
    data_root: Path = field(default_factory=lambda: Path("data/raw"))
    cache_root: Path = field(default_factory=lambda: Path("data/cache"))
    image_size: int = 224
    batch_size: int = 32
    # Default 0 for macOS / sandbox safety (shared memory manager restriction).
    # Set to 4+ on Linux/GPU machines for throughput.
    num_workers: int = 0
    pin_memory: bool = True

    # Episode sampling: probability an episode starts from an adversarial input.
    adversarial_ratio: float = 0.5
    attack_types: Sequence[AttackType] = field(default_factory=lambda: DEFAULT_ATTACK_TYPES)

    # ImageNette ships at 160 or 320 px; we resize to image_size in transforms.
    imagenette_size: Literal[160, 320] = 320

    # ImageNet-1K root must contain train/ and val/ ImageFolder layouts.
    imagenet_root: Path | None = None

    # Adversarial generation settings (foolbox).
    pgd_steps: int = 20
    pgd_epsilon: float = 8 / 255
    fgsm_epsilon: float = 8 / 255
    cw_steps: int = 1000
    precompute_adversarial: bool = True
    adversarial_batch_size: int = 8

    seed: int = 42
    download: bool = True

    def validate(self) -> None:
        if not 0.0 <= self.adversarial_ratio <= 1.0:
            raise ValueError("adversarial_ratio must be in [0, 1]")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.image_size < 1:
            raise ValueError("image_size must be >= 1")
        if not self.attack_types:
            raise ValueError("attack_types must not be empty")
        if self.dataset == "imagenet" and self.imagenet_root is None:
            raise ValueError("imagenet_root is required when dataset='imagenet'")
