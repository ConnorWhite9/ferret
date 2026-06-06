"""Extract and log multi-objective rewards from vectorized Gymnasium envs."""

from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from ferret.constants import MORL_OBJECTIVE_NAMES


def extract_reward_vectors(infos: dict[str, Any], num_envs: int) -> np.ndarray:
    """
    Build [num_envs, 4] reward vectors from SyncVectorEnv step infos.

    Handles batched arrays and per-env final_info fallbacks.
    """
    vectors = np.zeros((num_envs, 4), dtype=np.float32)
    if not infos:
        return vectors

    if "reward_vector" in infos:
        raw = infos["reward_vector"]
        if isinstance(raw, np.ndarray):
            if raw.shape == (num_envs, 4):
                return raw.astype(np.float32)
            if raw.shape == (num_envs,):
                return vectors

    if "final_info" in infos:
        for env_idx, info in enumerate(infos["final_info"]):
            if info and "reward_vector" in info:
                vectors[env_idx] = np.asarray(info["reward_vector"], dtype=np.float32)

    return vectors


def extract_episode_metrics(infos: dict[str, Any], num_envs: int) -> dict[str, np.ndarray]:
    """Pull episode-level diagnostics when episodes terminate."""
    metrics: dict[str, np.ndarray] = {}
    if "final_info" not in infos:
        return metrics

    probes = []
    confidences = []
    flagged = []
    for info in infos["final_info"]:
        if not info:
            probes.append(np.nan)
            confidences.append(np.nan)
            flagged.append(np.nan)
            continue
        probes.append(info.get("probes_used", np.nan))
        confidences.append(info.get("confidence", np.nan))
        decision = info.get("decision", "")
        flagged.append(1.0 if decision == "flag" else 0.0)

    if probes:
        metrics["probes_used"] = np.array(probes, dtype=np.float32)
        metrics["confidence"] = np.array(confidences, dtype=np.float32)
        metrics["flagged"] = np.array(flagged, dtype=np.float32)
    return metrics


def log_morl_metrics(
    writer: SummaryWriter,
    reward_vectors: np.ndarray,
    global_step: int,
    episode_metrics: dict[str, np.ndarray] | None = None,
) -> None:
    """Write per-objective TensorBoard scalars."""
    # reward_vectors may be (T, N, 4) or (N, 4) — flatten to (4,)
    means = reward_vectors.reshape(-1, len(MORL_OBJECTIVE_NAMES)).mean(axis=0)
    for idx, name in enumerate(MORL_OBJECTIVE_NAMES):
        writer.add_scalar(f"morl/{name}", float(means[idx]), global_step)

    if episode_metrics:
        for key, values in episode_metrics.items():
            finite = values[np.isfinite(values)]
            if finite.size > 0:
                writer.add_scalar(f"episode/{key}", float(finite.mean()), global_step)
