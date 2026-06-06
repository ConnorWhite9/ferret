"""
Pareto frontier utilities for Ferret evaluation.

Primary Ferret metric (spec §10.1):
    Pareto frontier of (detection accuracy, mean queries used)
    plotted against Feature Squeezing and Mahalanobis baselines.

Exports:
    - CSV with per-detector Pareto points
    - matplotlib figure (saved to file or shown inline)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from eval.metrics import DetectorMetrics


@dataclass
class ParetoPoint:
    detector: str
    accuracy: float
    mean_probes: float


def extract_pareto_points(metrics: Sequence[DetectorMetrics]) -> list[ParetoPoint]:
    """One Pareto point per detector (accuracy, mean_probes)."""
    return [
        ParetoPoint(
            detector=m.detector,
            accuracy=m.accuracy,
            mean_probes=m.mean_probes,
        )
        for m in metrics
    ]


def pareto_frontier(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """
    Extract the Pareto-optimal subset:
    higher accuracy AND fewer probes is better.
    Returns points not dominated by any other.
    """
    pts = list(points)
    frontier = []
    for p in pts:
        dominated = any(
            q.accuracy >= p.accuracy and q.mean_probes <= p.mean_probes and q is not p
            for q in pts
        )
        if not dominated:
            frontier.append(p)
    return sorted(frontier, key=lambda p: p.mean_probes)


def save_pareto_csv(points: Sequence[ParetoPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["detector", "accuracy", "mean_probes"])
        writer.writeheader()
        for p in points:
            writer.writerow(
                {"detector": p.detector, "accuracy": f"{p.accuracy:.6f}", "mean_probes": f"{p.mean_probes:.4f}"}
            )


def plot_pareto(
    all_points: Sequence[ParetoPoint],
    frontier: Sequence[ParetoPoint],
    output_path: Path | None = None,
    title: str = "Ferret — Pareto Frontier (accuracy vs queries)",
) -> None:
    """Generate a Pareto plot. Saves to `output_path` if given, else shows."""
    try:
        import matplotlib
        if output_path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install matplotlib to generate Pareto plots: pip install matplotlib") from exc

    fig, ax = plt.subplots(figsize=(8, 5))

    # All points
    for p in all_points:
        ax.scatter(p.mean_probes, p.accuracy, zorder=3, s=80)
        ax.annotate(
            p.detector,
            (p.mean_probes, p.accuracy),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )

    # Frontier line
    if frontier:
        fx = [p.mean_probes for p in frontier]
        fy = [p.accuracy for p in frontier]
        ax.plot(fx, fy, "--", color="gray", linewidth=1.2, label="frontier")

    ax.set_xlabel("Mean probes used")
    ax.set_ylabel("Detection accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
