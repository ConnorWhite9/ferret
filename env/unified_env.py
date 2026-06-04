"""Factory for Ferret Gymnasium environments across modalities."""

from __future__ import annotations

from typing import Literal

import gymnasium as gym

from data.pipeline import FerretDataPipeline
from env.vision_env import FerretVisionEnv, make_vision_env
from policy.vision_encoder import VisionEncoder
from reward.morl_reward import MORLReward, RewardNormalizer

Modality = Literal["vision", "language"]


def make_ferret_env(
    modality: Modality,
    target_model: VisionEncoder,
    data_pipeline: FerretDataPipeline,
    *,
    split: str = "train",
    seed: int = 0,
    reward_fn: MORLReward | None = None,
    reward_normalizer: RewardNormalizer | None = None,
    max_budget: int | None = None,
) -> gym.Env:
    """
    Unified entry point for RL training environments.

    Phase 1–2: vision only. Language env is reserved for Phase 2+.
    """
    if modality == "vision":
        kwargs = {
            "target_model": target_model,
            "data_pipeline": data_pipeline,
            "split": split,
            "seed": seed,
            "reward_fn": reward_fn,
            "reward_normalizer": reward_normalizer,
        }
        if max_budget is not None:
            kwargs["max_budget"] = max_budget
        return make_vision_env(**kwargs)

    raise NotImplementedError(f"Modality '{modality}' is not implemented yet.")


__all__ = ["FerretVisionEnv", "make_ferret_env", "make_vision_env"]
