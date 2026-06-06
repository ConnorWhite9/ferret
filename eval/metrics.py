"""
Evaluation metrics for Ferret adversarial detection.

Spec §10.1:
  - ROC-AUC per attack type (FGSM / PGD / CW) and combined
  - FPR@TPR95 — false positive rate at 95% true positive rate
  - Pareto: detection accuracy vs mean queries used
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class EpisodeRecord:
    """
    Minimal unit of eval data.  All detectors produce one per episode.
    confidence  : detector's final score in [0, 1]
    flagged     : binary prediction (threshold applied)
    is_adversarial : ground truth
    attack_type : "fgsm" | "pgd" | "cw" | None (clean)
    probes_used : number of queries consumed (1 for single-pass baselines)
    """

    confidence: float
    flagged: bool
    is_adversarial: bool
    probes_used: int
    attack_type: str | None


@dataclass
class AttackMetrics:
    attack_type: str
    n: int
    roc_auc: float
    fpr_at_tpr95: float
    tpr: float
    fpr: float
    accuracy: float
    mean_probes: float


@dataclass
class DetectorMetrics:
    detector: str
    n_total: int
    n_adversarial: int
    n_clean: int
    roc_auc: float
    fpr_at_tpr95: float
    tpr: float
    fpr: float
    accuracy: float
    mean_probes: float
    per_attack: list[AttackMetrics] = field(default_factory=list)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC via trapezoidal rule (no sklearn dependency)."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    n_pos = int(labels_sorted.sum())
    n_neg = len(labels_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1 - labels_sorted)
    tpr = tps / n_pos
    fpr = fps / n_neg
    # prepend (0, 0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))


def _fpr_at_tpr95(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    FPR at the highest score threshold that achieves TPR >= 0.95.
    (Higher threshold → stricter → fewer FP, so we take the first hit
    as we sweep thresholds from high to low.)
    Returns 1.0 if TPR 0.95 is never reached.
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    for thresh in np.sort(np.unique(scores))[::-1]:
        pred = (scores >= thresh).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        if tpr >= 0.95:
            return float(fpr)
    return 1.0


def _binary_metrics(
    records: Sequence[EpisodeRecord],
) -> tuple[float, float, float, float, float, float]:
    """Returns (roc_auc, fpr_at_tpr95, tpr, fpr, accuracy, mean_probes)."""
    if not records:
        return (float("nan"),) * 5 + (0.0,)  # type: ignore[return-value]

    labels = np.array([int(r.is_adversarial) for r in records])
    scores = np.array([r.confidence for r in records])
    preds = np.array([int(r.flagged) for r in records])

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    n_pos = tp + fn
    n_neg = fp + tn

    tpr = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)
    acc = (tp + tn) / max(len(records), 1)
    auc = _roc_auc(labels, scores)
    fpr95 = _fpr_at_tpr95(labels, scores)
    mean_probes = float(np.mean([r.probes_used for r in records]))
    return auc, fpr95, tpr, fpr, acc, mean_probes


def compute_metrics(
    detector_name: str,
    records: Sequence[EpisodeRecord],
) -> DetectorMetrics:
    """Full metric report for one detector across all records."""
    n_adv = sum(1 for r in records if r.is_adversarial)
    n_clean = sum(1 for r in records if not r.is_adversarial)
    auc, fpr95, tpr, fpr, acc, mean_probes = _binary_metrics(records)

    per_attack: list[AttackMetrics] = []
    attack_types = sorted(
        {r.attack_type for r in records if r.is_adversarial and r.attack_type is not None}
    )
    for atk in attack_types:
        # For each attack type: adversarial records of that type + all clean records
        subset = [
            r for r in records if not r.is_adversarial or r.attack_type == atk
        ]
        a_auc, a_fpr95, a_tpr, a_fpr, a_acc, a_probes = _binary_metrics(subset)
        per_attack.append(
            AttackMetrics(
                attack_type=str(atk),
                n=len(subset),
                roc_auc=a_auc,
                fpr_at_tpr95=a_fpr95,
                tpr=a_tpr,
                fpr=a_fpr,
                accuracy=a_acc,
                mean_probes=a_probes,
            )
        )

    return DetectorMetrics(
        detector=detector_name,
        n_total=len(records),
        n_adversarial=n_adv,
        n_clean=n_clean,
        roc_auc=auc,
        fpr_at_tpr95=fpr95,
        tpr=tpr,
        fpr=fpr,
        accuracy=acc,
        mean_probes=mean_probes,
        per_attack=per_attack,
    )


def print_metrics(m: DetectorMetrics) -> None:
    w = 60
    print("=" * w)
    print(f"  {m.detector}")
    print("=" * w)
    print(f"  n={m.n_total}  adv={m.n_adversarial}  clean={m.n_clean}")
    print(f"  ROC-AUC        {m.roc_auc:.4f}")
    print(f"  FPR@TPR95      {m.fpr_at_tpr95:.4f}")
    print(f"  TPR            {m.tpr:.4f}")
    print(f"  FPR            {m.fpr:.4f}")
    print(f"  Accuracy       {m.accuracy:.4f}")
    print(f"  Mean probes    {m.mean_probes:.2f}")
    if m.per_attack:
        print()
        print("  Per-attack breakdown:")
        header = f"  {'attack':<8}  {'n':>5}  {'AUC':>6}  {'FPR95':>6}  {'TPR':>6}  {'FPR':>6}"
        print(header)
        for a in m.per_attack:
            print(
                f"  {a.attack_type:<8}  {a.n:>5}  {a.roc_auc:>6.3f}"
                f"  {a.fpr_at_tpr95:>6.3f}  {a.tpr:>6.3f}  {a.fpr:>6.3f}"
            )
    print()
