"""Dataset download helpers."""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path
from typing import Literal
from urllib.request import urlretrieve

from data.config import DataConfig

IMAGENETTE_URLS: dict[int, str] = {
    160: "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
    320: "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
}


def imagenette_dirname(size: Literal[160, 320]) -> str:
    return f"imagenette2-{size}"


def imagenette_root(config: DataConfig) -> Path:
    return config.data_root / imagenette_dirname(config.imagenette_size)


def _download_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size)
        print(f"\rDownloading {destination.name}: {pct:3d}%", end="", flush=True)

    urlretrieve(url, destination, reporthook=_report)
    print()


def _extract_archive(archive_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)


def ensure_imagenette(config: DataConfig) -> Path:
    """Download and extract ImageNette if missing."""
    root = imagenette_root(config)
    if (root / "train").is_dir() and (root / "val").is_dir():
        return root

    url = IMAGENETTE_URLS[config.imagenette_size]
    archive_name = f"{imagenette_dirname(config.imagenette_size)}.tgz"
    archive_path = config.data_root / archive_name

    if not archive_path.exists():
        print(f"Fetching ImageNette ({config.imagenette_size}px) from {url}")
        _download_with_progress(url, archive_path)

    print(f"Extracting {archive_path} -> {config.data_root}")
    _extract_archive(archive_path, config.data_root)

    if not (root / "train").is_dir():
        raise RuntimeError(f"ImageNette extraction failed; missing {root / 'train'}")

    return root


def ensure_dataset_root(config: DataConfig) -> Path:
    """Resolve and optionally download the configured dataset root."""
    config.validate()

    if config.dataset == "imagenette":
        if not config.download:
            root = imagenette_root(config)
            if not (root / "train").is_dir():
                raise FileNotFoundError(
                    f"ImageNette not found at {root}. "
                    "Set download=True or place the dataset manually."
                )
            return root
        return ensure_imagenette(config)

    assert config.imagenet_root is not None
    root = config.imagenet_root
    for split in ("train", "val"):
        if not (root / split).is_dir():
            raise FileNotFoundError(
                f"ImageNet split '{split}' not found at {root / split}. "
                "ImageNet-1K must be downloaded separately."
            )
    return root


def stable_sample_key(relative_path: str, attack_type: str) -> str:
    """Deterministic cache key for an (image, attack) pair."""
    digest = hashlib.sha256(f"{relative_path}:{attack_type}".encode()).hexdigest()
    return digest[:16]
