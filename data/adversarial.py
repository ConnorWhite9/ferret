"""Adversarial example generation and on-disk caching."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.config import AttackType, DataConfig, SplitName
from data.download import stable_sample_key
from data.datasets import RawSample, VisionDataset
from data.transforms import build_transforms, denormalize, normalize


class AdversarialGenerator:
    """Generate FGSM / PGD / CW examples with foolbox and cache to disk."""

    def __init__(self, model: nn.Module, config: DataConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self._fmodel = self._build_foolbox_model(model)

    @staticmethod
    def _build_foolbox_model(model: nn.Module):
        import foolbox as fb

        model.eval()
        bounds = (0.0, 1.0)
        preprocessing = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "axis": -3,
        }
        return fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

    def _epsilon_for(self, attack_type: AttackType) -> float | None:
        if attack_type in ("fgsm", "pgd"):
            return self.config.fgsm_epsilon if attack_type == "fgsm" else self.config.pgd_epsilon
        return None

    def cache_dir(self, split: SplitName, attack_type: AttackType) -> Path:
        dataset_tag = (
            f"imagenette{self.config.imagenette_size}"
            if self.config.dataset == "imagenette"
            else "imagenet"
        )
        return self.config.cache_root / dataset_tag / split / attack_type

    def cache_path(self, split: SplitName, attack_type: AttackType, relative_path: str) -> Path:
        key = stable_sample_key(relative_path, attack_type)
        return self.cache_dir(split, attack_type) / f"{key}.pt"

    def is_cached(self, split: SplitName, attack_type: AttackType, relative_path: str) -> bool:
        return self.cache_path(split, attack_type, relative_path).exists()

    def read_cached(
        self, split: SplitName, attack_type: AttackType, relative_path: str
    ) -> torch.Tensor:
        path = self.cache_path(split, attack_type, relative_path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        return payload["adversarial"]

    def write_cached(
        self,
        split: SplitName,
        attack_type: AttackType,
        relative_path: str,
        clean_image: torch.Tensor,
        adversarial_image: torch.Tensor,
        label: int,
    ) -> None:
        path = self.cache_path(split, attack_type, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "clean": clean_image.cpu(),
                "adversarial": adversarial_image.cpu(),
                "label": label,
                "attack_type": attack_type,
                "relative_path": relative_path,
            },
            path,
        )

    def _attack(self, attack_type: AttackType):
        import foolbox as fb

        if attack_type == "fgsm":
            return fb.attacks.LinfFastGradientAttack()
        if attack_type == "pgd":
            return fb.attacks.LinfPGD(steps=self.config.pgd_steps, random_start=True)
        if attack_type == "cw":
            return fb.attacks.L2CarliniWagnerAttack(steps=self.config.cw_steps)
        raise ValueError(f"Unknown attack type: {attack_type}")

    @staticmethod
    def _criterion(labels: torch.Tensor):
        from foolbox.criteria import Misclassification

        return Misclassification(labels)

    def generate_batch(
        self,
        clean_images: torch.Tensor,
        labels: torch.Tensor,
        attack_type: AttackType,
    ) -> torch.Tensor:
        """Generate adversarial images in normalized tensor space."""
        images_01 = denormalize(clean_images.to(self.device))
        labels = labels.to(self.device)
        attack = self._attack(attack_type)
        criterion = self._criterion(labels)
        epsilon = self._epsilon_for(attack_type)
        if epsilon is not None:
            _, advs, _ = attack(self._fmodel, images_01, criterion, epsilons=epsilon)
        else:
            _, advs, _ = attack(self._fmodel, images_01, criterion)
        return normalize(advs.detach())

    def generate_one(
        self,
        clean_image: torch.Tensor,
        label: int,
        attack_type: AttackType,
    ) -> torch.Tensor:
        batch = clean_image.unsqueeze(0)
        labels = torch.tensor([label], dtype=torch.long)
        return self.generate_batch(batch, labels, attack_type)[0].cpu()

    def cache_complete(self, dataset: VisionDataset, attack_type: AttackType) -> bool:
        for index in range(len(dataset)):
            sample = dataset[index]
            if not self.is_cached(dataset.split, attack_type, sample.relative_path):
                return False
        return True

    def precompute_cache(self, dataset: VisionDataset, attack_type: AttackType) -> None:
        """Generate and persist adversarial examples for every image in a split."""
        if self.cache_complete(dataset, attack_type):
            return

        loader = DataLoader(
            dataset,
            batch_size=self.config.adversarial_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
            collate_fn=_collate_raw_samples,
        )

        self.model.eval()
        for batch in tqdm(loader, desc=f"Caching {attack_type} ({dataset.split})"):
            images = batch["image"]
            labels = batch["label"]
            paths = batch["relative_path"]
            indices = batch["index"]

            pending_mask = [
                not self.is_cached(dataset.split, attack_type, path) for path in paths
            ]
            if not any(pending_mask):
                continue

            adv_batch = self.generate_batch(images, labels, attack_type)
            for i, needed in enumerate(pending_mask):
                if not needed:
                    continue
                self.write_cached(
                    dataset.split,
                    attack_type,
                    paths[i],
                    images[i],
                    adv_batch[i],
                    int(labels[i].item()),
                )


def _collate_raw_samples(batch: list[RawSample]) -> dict:
    return {
        "image": torch.stack([sample.image for sample in batch]),
        "label": torch.tensor([sample.label for sample in batch], dtype=torch.long),
        "index": torch.tensor([sample.index for sample in batch], dtype=torch.long),
        "relative_path": [sample.relative_path for sample in batch],
    }
