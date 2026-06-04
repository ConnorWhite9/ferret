"""Episode sampling and DataLoader construction."""

from __future__ import annotations

import random
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from data.adversarial import AdversarialGenerator, _collate_raw_samples
from data.config import AttackType, DataConfig, SplitName
from data.datasets import EpisodeSample, RawSample, VisionDataset
from data.transforms import build_transforms


class EpisodeDataset(Dataset):
    """
    Samples episode starting states from clean or cached adversarial inputs.

    Each __getitem__ randomly chooses clean vs adversarial according to
    config.adversarial_ratio, and uniformly selects an attack type when
    adversarial. Requires a precomputed cache for adversarial samples.
    """

    def __init__(
        self,
        config: DataConfig,
        split: SplitName,
        generator: AdversarialGenerator | None = None,
        seed: int | None = None,
    ):
        self.config = config
        self.split = split
        self.generator = generator
        transform = build_transforms(config.image_size, split)
        self.base = VisionDataset(config, split, transform=transform)
        self.rng = random.Random(seed if seed is not None else config.seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> EpisodeSample:
        raw = self.base[index]
        use_adversarial = (
            self.generator is not None
            and self.config.adversarial_ratio > 0.0
            and self.rng.random() < self.config.adversarial_ratio
        )

        if not use_adversarial:
            return EpisodeSample(
                image=raw.image,
                clean_image=raw.image,
                label=raw.label,
                is_adversarial=False,
                attack_type=None,
                index=raw.index,
                relative_path=raw.relative_path,
            )

        attack_type = self.rng.choice(list(self.config.attack_types))
        if not self.generator.is_cached(self.split, attack_type, raw.relative_path):
            raise FileNotFoundError(
                f"Missing cached adversarial example for {raw.relative_path} "
                f"({attack_type}). Call pipeline.ensure_adversarial_cache() first."
            )

        adv_image = self.generator.read_cached(self.split, attack_type, raw.relative_path)
        return EpisodeSample(
            image=adv_image,
            clean_image=raw.image,
            label=raw.label,
            is_adversarial=True,
            attack_type=attack_type,
            index=raw.index,
            relative_path=raw.relative_path,
        )

    def sample_episode(self) -> EpisodeSample:
        """Random episode for env.reset()."""
        index = self.rng.randrange(len(self.base))
        return self[index]


class FerretDataPipeline:
    """High-level API for dataset loading, caching, and episode sampling."""

    def __init__(
        self,
        config: DataConfig | None = None,
        model: torch.nn.Module | None = None,
        device: str | torch.device | None = None,
    ):
        self.config = config or DataConfig()
        self.config.validate()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device) if model is not None else None
        self.generator = (
            AdversarialGenerator(self.model, self.config, self.device)
            if self.model is not None
            else None
        )
        self._episode_sets: dict[SplitName, EpisodeDataset] = {}

    def vision_dataset(self, split: SplitName) -> VisionDataset:
        transform = build_transforms(self.config.image_size, split)
        return VisionDataset(self.config, split, transform=transform)

    def ensure_adversarial_cache(self, split: SplitName) -> None:
        if self.generator is None:
            raise ValueError("A target model is required to build the adversarial cache.")
        if not self.config.precompute_adversarial:
            return

        dataset = self.vision_dataset(split)
        for attack_type in self.config.attack_types:
            self.generator.precompute_cache(dataset, attack_type)

    def episode_dataset(self, split: SplitName) -> EpisodeDataset:
        if split not in self._episode_sets:
            seed = self.config.seed + (0 if split == "train" else 10_000)
            self._episode_sets[split] = EpisodeDataset(
                self.config,
                split,
                generator=self.generator,
                seed=seed,
            )
        return self._episode_sets[split]

    def build_dataloader(
        self,
        split: SplitName = "train",
        shuffle: bool | None = None,
    ) -> DataLoader:
        if shuffle is None:
            shuffle = split == "train"

        if self.config.adversarial_ratio > 0.0 and self.generator is not None:
            if self.config.precompute_adversarial:
                self.ensure_adversarial_cache(split)
            dataset: Dataset = self.episode_dataset(split)
            collate_fn = collate_episodes
        else:
            dataset = self.vision_dataset(split)

            def collate_fn(batch: list[RawSample]) -> dict:
                return collate_raw_to_episodes(_collate_raw_samples(batch))

        pin_memory = self.config.pin_memory and torch.cuda.is_available()
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=split == "train",
        )

    def sample_episode(self, split: SplitName = "train") -> EpisodeSample:
        dataset = self.episode_dataset(split)
        if self.config.adversarial_ratio > 0.0 and self.generator is not None:
            self.ensure_adversarial_cache(split)
        return dataset.sample_episode()

    def iter_episodes(self, split: SplitName = "train") -> Iterator[EpisodeSample]:
        loader = self.build_dataloader(split=split)
        for batch in loader:
            batch_size = batch["image"].shape[0]
            for i in range(batch_size):
                yield EpisodeSample(
                    image=batch["image"][i],
                    clean_image=batch["clean_image"][i],
                    label=int(batch["label"][i].item()),
                    is_adversarial=bool(batch["is_adversarial"][i].item()),
                    attack_type=batch["attack_type"][i],
                    index=int(batch["index"][i].item()),
                    relative_path=batch["relative_path"][i],
                )


def collate_episodes(batch: list[EpisodeSample]) -> dict:
    return {
        "image": torch.stack([sample.image for sample in batch]),
        "clean_image": torch.stack([sample.clean_image for sample in batch]),
        "label": torch.tensor([sample.label for sample in batch], dtype=torch.long),
        "is_adversarial": torch.tensor(
            [sample.is_adversarial for sample in batch], dtype=torch.bool
        ),
        "attack_type": [sample.attack_type for sample in batch],
        "index": torch.tensor([sample.index for sample in batch], dtype=torch.long),
        "relative_path": [sample.relative_path for sample in batch],
    }


def collate_raw_to_episodes(batch: dict) -> dict:
    """Convert a raw-sample batch into episode batch fields (clean-only)."""
    return {
        "image": batch["image"],
        "clean_image": batch["image"],
        "label": batch["label"],
        "is_adversarial": torch.zeros(len(batch["label"]), dtype=torch.bool),
        "attack_type": [None for _ in batch["label"]],
        "index": batch["index"],
        "relative_path": batch["relative_path"],
    }
