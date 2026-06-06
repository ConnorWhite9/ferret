"""Smoke tests for eval/baselines.py — CPU only, synthetic data."""

import torch
import pytest

from eval.baselines import (
    FeatureSqueezing,
    MahalanobisDetector,
    _bit_depth_reduce,
    _median_smooth,
)


def _fake_encoder():
    """Minimal VisionEncoder stand-in that returns deterministic logits."""
    import torch.nn as nn
    from ferret.constants import NUM_CLASSES

    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3 * 224 * 224, NUM_CLASSES)
            self.device = torch.device("cpu")
            self.feature_dim = 2048

        @torch.inference_mode()
        def logits(self, image):
            x = image.view(image.shape[0], -1) if image.dim() == 4 else image.view(1, -1)
            return self.linear(x)

        @torch.inference_mode()
        def encode(self, image):
            x = image.view(1, -1)
            return torch.zeros(1, self.feature_dim)

    return FakeEncoder()


def _random_image() -> torch.Tensor:
    return torch.rand(3, 224, 224)


def test_bit_depth_reduce():
    img = _random_image()
    reduced = _bit_depth_reduce(img, bits=4)
    assert reduced.shape == img.shape
    # All values should snap to multiples of 1/15
    steps = 15.0
    assert torch.all((reduced * steps).round() / steps == reduced)


def test_median_smooth():
    img = _random_image()
    smooth = _median_smooth(img)
    assert smooth.shape == img.shape


def test_feature_squeezing_fit_and_detect(monkeypatch):
    enc = _fake_encoder()
    fs = FeatureSqueezing(enc, threshold=0.5)
    clean_images = [_random_image() for _ in range(10)]
    fs.fit(clean_images, percentile=95.0)
    # Threshold should have been updated from fit
    record = fs.detect(_random_image(), is_adversarial=False)
    assert 0.0 <= record.confidence <= 1.0
    assert record.probes_used == 1
    assert record.is_adversarial is False


def test_mahalanobis_fit_and_detect():
    enc = _fake_encoder()
    maha = MahalanobisDetector(enc, threshold=0.5)
    clean_images = [_random_image() for _ in range(20)]
    clean_labels = [i % 10 for i in range(20)]
    maha.fit(clean_images, clean_labels)
    record = maha.detect(_random_image(), is_adversarial=True, attack_type="fgsm")
    assert 0.0 <= record.confidence <= 1.0
    assert record.attack_type == "fgsm"


def test_mahalanobis_requires_fit():
    enc = _fake_encoder()
    maha = MahalanobisDetector(enc)
    with pytest.raises(RuntimeError, match="fit"):
        maha._raw_score(_random_image())
