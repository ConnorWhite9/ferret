"""Gymnasium environment for vision sequential adversarial probing."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from agents.confidence import ConfidenceAggregator
from agents.decision import DecisionNode, DecisionOutcome
from agents.probe_executor import ProbeExecutor
from data.pipeline import FerretDataPipeline
from ferret.constants import (
    MAX_BUDGET,
    NUM_CLASSES,
    NUM_GRID_CELLS,
    NUM_MAGNITUDES,
    NUM_PERT_TYPES,
    NUM_PROBE_ACTIONS,
    PREFERENCE_DIM,
)
from ferret.episode import EpisodeState, build_policy_obs
from policy.vision_encoder import VisionEncoder
from reward.morl_reward import MORLReward, RewardNormalizer, sample_preference_vector, scalarize


class FerretVisionEnv(gym.Env):
    """
    Sequential probing RL environment.

    Each step applies one probe action, updates confidence, and may terminate
    when the decision node flags or abstains. Returns a 4-d reward vector in
    info plus a preference-scalarized reward for PPO.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        target_model: VisionEncoder,
        data_pipeline: FerretDataPipeline,
        split: str = "train",
        max_budget: int = MAX_BUDGET,
        reward_fn: MORLReward | None = None,
        reward_normalizer: RewardNormalizer | None = None,
        decision_node: DecisionNode | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.target_model = target_model
        self.data_pipeline = data_pipeline
        self.split = split
        self.max_budget = max_budget
        self.reward_fn = reward_fn or MORLReward(max_budget=max_budget)
        self.reward_normalizer = reward_normalizer or RewardNormalizer()
        self.decision_node = decision_node or DecisionNode()
        self.probe_executor = ProbeExecutor(target_model)
        self.confidence = ConfidenceAggregator()
        self._rng = np.random.default_rng(seed)
        self._episode: EpisodeState | None = None

        self.action_space = spaces.Discrete(NUM_PROBE_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "input_embedding": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(target_model.feature_dim,),
                    dtype=np.float32,
                ),
                "preference": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(PREFERENCE_DIM,),
                    dtype=np.float32,
                ),
                "remaining_budget": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "confidence": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "probe_grid": spaces.Box(
                    low=-1,
                    high=NUM_GRID_CELLS - 1,
                    shape=(max_budget,),
                    dtype=np.int32,
                ),
                "probe_pert": spaces.Box(
                    low=-1,
                    high=NUM_PERT_TYPES - 1,
                    shape=(max_budget,),
                    dtype=np.int32,
                ),
                "probe_mag": spaces.Box(
                    low=-1,
                    high=NUM_MAGNITUDES - 1,
                    shape=(max_budget,),
                    dtype=np.int32,
                ),
                "response_logits": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(max_budget, NUM_CLASSES),
                    dtype=np.float32,
                ),
                "history_len": spaces.Box(
                    low=0,
                    high=max_budget,
                    shape=(1,),
                    dtype=np.int32,
                ),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        sample = self.data_pipeline.sample_episode(self.split)  # type: ignore[arg-type]
        # Always compute embedding from the episode image (adversarial or clean).
        _, features = self.target_model(sample.image)
        input_embedding = features.squeeze(0).detach().cpu()
        # Baseline logits MUST come from the clean image so the confidence score
        # measures divergence from clean-model behaviour, not from an adversarial
        # baseline (which would invert the signal — clean shifts > adversarial shifts).
        clean_logits, _ = self.target_model(sample.clean_image)
        baseline_logits = clean_logits.squeeze(0).detach().cpu()
        preference = sample_preference_vector(self._rng)

        self._episode = EpisodeState(
            image=sample.image.cpu(),
            clean_image=sample.clean_image.cpu(),
            label=sample.label,
            is_adversarial=sample.is_adversarial,
            attack_type=sample.attack_type,
            input_embedding=input_embedding,
            baseline_logits=baseline_logits,
            preference=preference,
            remaining_budget=self.max_budget,
        )
        self.confidence.reset(baseline_logits)
        self._episode.confidence = self.confidence.score

        obs = build_policy_obs(self._episode, self.max_budget)
        info = {
            "preference": preference.copy(),
            "is_adversarial": sample.is_adversarial,
            "attack_type": sample.attack_type,
        }
        return obs, info

    def step(self, action: int):
        if self._episode is None:
            raise RuntimeError("Environment must be reset before step.")

        episode = self._episode
        confidence_before = episode.confidence

        probe = self.probe_executor.execute(episode.image, action, label=episode.label)
        episode.image = probe.probed_image
        episode.remaining_budget -= 1

        step_index = len(episode.probe_grid)
        episode.probe_grid.append(probe.action.grid_cell)
        episode.probe_pert.append(probe.action.perturbation_type)
        episode.probe_mag.append(probe.action.magnitude)
        episode.response_logits.append(probe.logits)

        confidence_after = self.confidence.update(probe.logits, step_index)
        episode.confidence = confidence_after

        reward_vector = self.reward_fn.step_reward(
            confidence_before,
            confidence_after,
            episode.remaining_budget,
        )

        decision = self.decision_node.evaluate(confidence_after, episode.remaining_budget)
        terminated = decision != DecisionOutcome.CONTINUE
        truncated = False

        if terminated:
            flagged = decision == DecisionOutcome.FLAG
            terminal = self.reward_fn.terminal_reward(
                flagged=flagged,
                is_adversarial=episode.is_adversarial,
                confidence=confidence_after,
                remaining_budget=episode.remaining_budget,
                attack_type=episode.attack_type,
            )
            reward_vector = reward_vector + terminal

        raw_vector = reward_vector.as_array()
        normalized_vector = self.reward_normalizer.normalize(raw_vector, update=True)
        scalar_reward = scalarize(normalized_vector, episode.preference)

        obs = build_policy_obs(self._episode, self.max_budget)
        info = {
            "reward_vector": raw_vector,
            "reward_vector_normalized": normalized_vector,
            "preference": episode.preference.copy(),
            "decision": decision.value,
            "is_adversarial": episode.is_adversarial,
            "attack_type": episode.attack_type,
            "confidence": confidence_after,
            "probes_used": len(episode.probe_grid),
        }
        return obs, scalar_reward, terminated, truncated, info


def stack_observations(observations: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack dict observations from vectorized environments."""
    return {key: np.stack([obs[key] for obs in observations], axis=0) for key in observations[0]}


def make_vision_env(
    target_model: VisionEncoder,
    data_pipeline: FerretDataPipeline,
    split: str = "train",
    seed: int = 0,
    reward_fn: MORLReward | None = None,
    reward_normalizer: RewardNormalizer | None = None,
    max_budget: int = MAX_BUDGET,
) -> FerretVisionEnv:
    return FerretVisionEnv(
        target_model=target_model,
        data_pipeline=data_pipeline,
        split=split,
        seed=seed,
        max_budget=max_budget,
        reward_fn=reward_fn,
        reward_normalizer=reward_normalizer,
    )
