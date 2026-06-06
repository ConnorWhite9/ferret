"""Unit tests for eval/metrics.py — no GPU, no data required."""

import numpy as np
import pytest

from eval.metrics import EpisodeRecord, compute_metrics, _roc_auc, _fpr_at_tpr95


def _make_records(
    n_adv: int,
    n_clean: int,
    adv_conf: float = 0.8,
    clean_conf: float = 0.2,
    threshold: float = 0.5,
    attack_type: str = "fgsm",
) -> list[EpisodeRecord]:
    records = []
    for _ in range(n_adv):
        records.append(
            EpisodeRecord(
                confidence=adv_conf,
                flagged=adv_conf >= threshold,
                is_adversarial=True,
                probes_used=3,
                attack_type=attack_type,
            )
        )
    for _ in range(n_clean):
        records.append(
            EpisodeRecord(
                confidence=clean_conf,
                flagged=clean_conf >= threshold,
                is_adversarial=False,
                probes_used=3,
                attack_type=None,
            )
        )
    return records


def test_perfect_detector():
    records = _make_records(50, 50, adv_conf=0.9, clean_conf=0.1)
    m = compute_metrics("perfect", records)
    assert m.accuracy == pytest.approx(1.0)
    assert m.tpr == pytest.approx(1.0)
    assert m.fpr == pytest.approx(0.0)
    assert m.roc_auc == pytest.approx(1.0, abs=0.02)


def test_random_detector():
    rng = np.random.default_rng(0)
    records = [
        EpisodeRecord(
            confidence=float(rng.random()),
            flagged=bool(rng.random() > 0.5),
            is_adversarial=bool(rng.random() > 0.5),
            probes_used=5,
            attack_type="fgsm" if rng.random() > 0.5 else None,
        )
        for _ in range(200)
    ]
    m = compute_metrics("random", records)
    assert 0.3 < m.roc_auc < 0.7  # near 0.5 for random


def test_per_attack_breakdown():
    pgd_records = _make_records(30, 30, attack_type="pgd")
    fgsm_records = _make_records(30, 0, attack_type="fgsm")
    all_records = pgd_records + fgsm_records
    m = compute_metrics("mixed", all_records)
    attack_types = {a.attack_type for a in m.per_attack}
    assert "fgsm" in attack_types
    assert "pgd" in attack_types


def test_roc_auc_all_positive():
    labels = np.ones(10, dtype=int)
    scores = np.linspace(0, 1, 10)
    result = _roc_auc(labels, scores)
    assert np.isnan(result)


def test_fpr_at_tpr95_perfect():
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    scores  = np.array([0.9, 0.85, 0.8, 0.75, 0.1, 0.05, 0.02, 0.01])
    fpr95 = _fpr_at_tpr95(labels, scores)
    assert fpr95 == pytest.approx(0.0, abs=0.01)


def test_compute_metrics_empty():
    # Empty records should return gracefully with nan/0 values, not crash
    m = compute_metrics("empty", [])
    assert isinstance(m.accuracy, float)
    assert m.n_total == 0
