"""
Single-pass adversarial detection baselines.

Spec §1.2 primary comparisons:
  - Feature Squeezing  (Xu et al. 2018)
  - Mahalanobis Distance  (Lee et al. 2018)

Both implement the same interface:
    detector.fit(clean_images, clean_labels)   # optional
    score = detector.score(image)              # float ∈ [0, 1]; higher = more adversarial
    record = detector.detect(image, ...)       # EpisodeRecord
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.metrics import EpisodeRecord
from policy.vision_encoder import VisionEncoder


# ---------------------------------------------------------------------------
# Feature Squeezing
# ---------------------------------------------------------------------------

def _median_smooth(image: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """
    Apply 2D smoothing via average pooling (fast approx to median filter).
    kernel must be odd so that padding = kernel//2 preserves spatial size.
    """
    if kernel % 2 == 0:
        kernel += 1  # force odd
    padding = kernel // 2
    return F.avg_pool2d(
        image.unsqueeze(0),
        kernel_size=kernel,
        stride=1,
        padding=padding,
    ).squeeze(0)


def _bit_depth_reduce(image: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Reduce each channel to `bits` bits in [0, 1] space."""
    steps = 2**bits - 1
    return (image * steps).round() / steps


def _logit_distance(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """L1 distance between softmax distributions of two logit vectors."""
    pa = torch.softmax(logits_a.float(), dim=-1)
    pb = torch.softmax(logits_b.float(), dim=-1)
    return float((pa - pb).abs().sum().item())


class FeatureSqueezing:
    """
    Detects adversarial inputs by comparing model output on original vs
    squeezed (lower-bit / smoothed) versions of the input.

    Large disagreement → adversarial.
    """

    def __init__(
        self,
        encoder: VisionEncoder,
        squeezers: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
        threshold: float = 0.5,
    ):
        self.encoder = encoder
        self.squeezers: list[Callable[[torch.Tensor], torch.Tensor]] = squeezers or [
            _median_smooth,
            _bit_depth_reduce,
        ]
        self.threshold = threshold
        self._threshold_fitted: float = threshold

    def fit(
        self,
        clean_images: list[torch.Tensor],
        *,
        percentile: float = 95.0,
    ) -> None:
        """
        Set threshold as the `percentile`-th score on clean inputs.
        Scores on clean data should be low; pick threshold at high percentile
        so FP rate is controlled.
        """
        scores = [self._raw_score(img) for img in clean_images]
        self._threshold_fitted = float(np.percentile(scores, percentile))

    def _raw_score(self, image: torch.Tensor) -> float:
        orig_logits = self.encoder.logits(image).squeeze(0).cpu()
        max_dist = 0.0
        for squeeze in self.squeezers:
            squeezed = squeeze(image)
            sq_logits = self.encoder.logits(squeezed).squeeze(0).cpu()
            dist = _logit_distance(orig_logits, sq_logits)
            max_dist = max(max_dist, dist)
        return max_dist

    def score(self, image: torch.Tensor) -> float:
        """Return normalized score in [0, 1] using sigmoid around threshold."""
        raw = self._raw_score(image)
        # sigmoid centred at fitted threshold; scale = 5 gives ~0.99 at 3×threshold
        scale = 5.0 / max(self._threshold_fitted, 1e-6)
        return float(torch.sigmoid(torch.tensor((raw - self._threshold_fitted) * scale)).item())

    def detect(
        self,
        image: torch.Tensor,
        is_adversarial: bool,
        attack_type: str | None = None,
    ) -> EpisodeRecord:
        raw = self._raw_score(image)
        flagged = raw >= self._threshold_fitted
        sig_score = float(torch.sigmoid(torch.tensor((raw - self._threshold_fitted) * 5.0 / max(self._threshold_fitted, 1e-6))).item())
        return EpisodeRecord(
            confidence=sig_score,
            flagged=flagged,
            is_adversarial=is_adversarial,
            probes_used=1,
            attack_type=attack_type,
        )


# ---------------------------------------------------------------------------
# Mahalanobis Distance
# ---------------------------------------------------------------------------

class MahalanobisDetector:
    """
    Class-conditional Gaussian detector over penultimate-layer features.

    Lee et al. (2018) "A Simple Unified Framework for Detecting Out-of-Distribution
    Samples and Adversarial Attacks."

    Score = min over classes of Mahalanobis distance from class mean, using
    a shared (tied) precision matrix estimated from clean training features.
    """

    def __init__(self, encoder: VisionEncoder, threshold: float = 0.5):
        self.encoder = encoder
        self.threshold = threshold
        self._class_means: dict[int, torch.Tensor] = {}
        self._precision: torch.Tensor | None = None
        self._threshold_fitted: float = threshold
        self._fitted = False

    def fit(
        self,
        clean_images: list[torch.Tensor],
        clean_labels: list[int],
        *,
        threshold_percentile: float = 5.0,
    ) -> None:
        """
        Compute per-class means and shared precision matrix from clean features.
        Threshold at `threshold_percentile`-th score on clean data so that most
        clean inputs score low (inliers have small Mahalanobis distance).
        """
        feats_by_class: dict[int, list[torch.Tensor]] = {}
        with torch.no_grad():
            for img, lbl in zip(clean_images, clean_labels):
                feat = self.encoder.encode(img).squeeze(0).cpu()
                feats_by_class.setdefault(lbl, []).append(feat)

        all_feats: list[torch.Tensor] = []
        for lbl, feats in feats_by_class.items():
            feat_stack = torch.stack(feats)
            mean = feat_stack.mean(dim=0)
            self._class_means[lbl] = mean
            all_feats.extend([f - mean for f in feats])

        # Shared covariance estimate
        residuals = torch.stack(all_feats)  # [N, D]
        cov = (residuals.T @ residuals) / max(len(all_feats) - 1, 1)
        # Regularise to keep invertible
        cov += torch.eye(cov.shape[0]) * 1e-4
        self._precision = torch.linalg.inv(cov).float()
        self._fitted = True

        # Calibrate threshold on clean set
        clean_scores = [self._raw_score(img) for img in clean_images]
        # Low score → inlier → pick a high percentile so we flag few clean
        self._threshold_fitted = float(np.percentile(clean_scores, 100 - threshold_percentile))

    def _raw_score(self, image: torch.Tensor) -> float:
        if not self._fitted:
            raise RuntimeError("MahalanobisDetector.fit() must be called before detect().")
        with torch.no_grad():
            feat = self.encoder.encode(image).squeeze(0).cpu().float()

        min_dist = float("inf")
        prec = self._precision
        for mean in self._class_means.values():
            diff = feat - mean.float()
            dist = float((diff @ prec @ diff).item())
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def score(self, image: torch.Tensor) -> float:
        raw = self._raw_score(image)
        scale = 1.0 / max(self._threshold_fitted, 1e-6)
        return float(torch.sigmoid(torch.tensor((raw - self._threshold_fitted) * scale)).item())

    def detect(
        self,
        image: torch.Tensor,
        is_adversarial: bool,
        attack_type: str | None = None,
    ) -> EpisodeRecord:
        raw = self._raw_score(image)
        flagged = raw >= self._threshold_fitted
        sig = self.score(image)
        return EpisodeRecord(
            confidence=sig,
            flagged=flagged,
            is_adversarial=is_adversarial,
            probes_used=1,
            attack_type=attack_type,
        )


# ---------------------------------------------------------------------------
# Shared baseline fitting helper
# ---------------------------------------------------------------------------

def fit_baselines(
    encoder: VisionEncoder,
    fit_images: list[torch.Tensor],
    fit_labels: list[int],
    *,
    fs_percentile: float = 95.0,
    maha_percentile: float = 5.0,
) -> tuple[FeatureSqueezing, MahalanobisDetector]:
    """Fit both baselines on a held-out set of clean images."""
    fs = FeatureSqueezing(encoder)
    fs.fit(fit_images, percentile=fs_percentile)

    maha = MahalanobisDetector(encoder)
    maha.fit(fit_images, fit_labels, threshold_percentile=maha_percentile)

    return fs, maha
