"""Shared episode state used by the Gym env and LangGraph agent."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from data.config import AttackType
from ferret.constants import MAX_BUDGET, NUM_CLASSES


@dataclass
class EpisodeState:
    image: torch.Tensor
    clean_image: torch.Tensor
    label: int | None
    is_adversarial: bool | None
    attack_type: AttackType | None
    input_embedding: torch.Tensor
    baseline_logits: torch.Tensor
    preference: np.ndarray
    remaining_budget: int = MAX_BUDGET
    confidence: float = 0.0
    probe_grid: list[int] = field(default_factory=list)
    probe_pert: list[int] = field(default_factory=list)
    probe_mag: list[int] = field(default_factory=list)
    response_logits: list[torch.Tensor] = field(default_factory=list)


def build_policy_obs(episode: EpisodeState, max_budget: int = MAX_BUDGET) -> dict[str, np.ndarray]:
    """Build the dict observation consumed by FerretPolicy / FerretAgent."""
    history_len = len(episode.probe_grid)

    probe_grid = np.full(max_budget, -1, dtype=np.int32)
    probe_pert = np.full(max_budget, -1, dtype=np.int32)
    probe_mag = np.full(max_budget, -1, dtype=np.int32)
    response_logits = np.zeros((max_budget, NUM_CLASSES), dtype=np.float32)

    if history_len > 0:
        probe_grid[:history_len] = np.array(episode.probe_grid, dtype=np.int32)
        probe_pert[:history_len] = np.array(episode.probe_pert, dtype=np.int32)
        probe_mag[:history_len] = np.array(episode.probe_mag, dtype=np.int32)
        response_logits[:history_len] = torch.stack(episode.response_logits).numpy()

    return {
        "input_embedding": episode.input_embedding.numpy().astype(np.float32),
        "preference": episode.preference.astype(np.float32),
        "remaining_budget": np.array(
            [episode.remaining_budget / max_budget],
            dtype=np.float32,
        ),
        "confidence": np.array([episode.confidence], dtype=np.float32),
        "probe_grid": probe_grid,
        "probe_pert": probe_pert,
        "probe_mag": probe_mag,
        "response_logits": response_logits,
        "history_len": np.array([history_len], dtype=np.int32),
    }
