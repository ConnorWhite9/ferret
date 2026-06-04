"""Multi-objective reward shaping and preference-weighted scalarization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.config import AttackType
from ferret.constants import (
    DEFAULT_FP_BETA,
    DEFAULT_GEN_GAMMA,
    DEFAULT_LAMBDA_EFF,
    LAMBDA_EFF_END,
    LAMBDA_EFF_START,
    MAX_BUDGET,
)


@dataclass
class RewardVector:
    """Four MORL objectives aligned with spec section 4.3."""

    accuracy: float = 0.0
    efficiency: float = 0.0
    false_positive: float = 0.0
    generalization: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.accuracy, self.efficiency, self.false_positive, self.generalization],
            dtype=np.float32,
        )

    def __add__(self, other: RewardVector) -> RewardVector:
        return RewardVector(
            accuracy=self.accuracy + other.accuracy,
            efficiency=self.efficiency + other.efficiency,
            false_positive=self.false_positive + other.false_positive,
            generalization=self.generalization + other.generalization,
        )


RARE_ATTACK_WEIGHTS: dict[AttackType | None, float] = {
    "fgsm": 0.0,
    "pgd": 0.05,
    "cw": 0.15,
    None: 0.0,
}


def sample_preference_vector(rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample a 4-d preference vector from a uniform Dirichlet."""
    generator = rng or np.random.default_rng()
    weights = generator.dirichlet(np.ones(4, dtype=np.float64))
    return weights.astype(np.float32)


def scalarize(reward_vector: RewardVector | np.ndarray, preference: np.ndarray) -> float:
    """Linear scalarization used by PPO: dot(preference, reward_vector)."""
    vector = reward_vector.as_array() if isinstance(reward_vector, RewardVector) else reward_vector
    return float(np.dot(preference, vector))


class RewardNormalizer:
    """Running mean/std normalization per objective (spec section 4 intro)."""

    def __init__(self, dim: int = 4, epsilon: float = 1e-8):
        self.dim = dim
        self.epsilon = epsilon
        self.count = epsilon
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)

    def update(self, reward_vector: np.ndarray) -> None:
        self.count += 1.0
        delta = reward_vector - self.mean
        self.mean += delta / self.count
        delta2 = reward_vector - self.mean
        self.var += delta * delta2

    def normalize(self, reward_vector: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
            self.update(reward_vector)
        std = np.sqrt(self.var / self.count) + self.epsilon
        return ((reward_vector - self.mean) / std).astype(np.float32)


class MORLReward:
    """Compute step and terminal reward vectors for the vision env."""

    def __init__(
        self,
        lambda_eff: float = DEFAULT_LAMBDA_EFF,
        fp_beta: float = DEFAULT_FP_BETA,
        gen_gamma: float = DEFAULT_GEN_GAMMA,
        max_budget: int = MAX_BUDGET,
    ):
        self.lambda_eff = lambda_eff
        self.fp_beta = fp_beta
        self.gen_gamma = gen_gamma
        self.max_budget = max_budget

    def step_reward(
        self,
        confidence_before: float,
        confidence_after: float,
        remaining_budget: int,
    ) -> RewardVector:
        info_gain = confidence_after - confidence_before
        query_penalty = -self.lambda_eff * (1.0 / max(remaining_budget, 1))
        return RewardVector(efficiency=info_gain + query_penalty)

    def terminal_reward(
        self,
        flagged: bool,
        is_adversarial: bool,
        confidence: float,
        remaining_budget: int,
        attack_type: AttackType | None,
    ) -> RewardVector:
        predicted_adversarial = flagged
        correct = predicted_adversarial == is_adversarial

        accuracy = 1.0 if correct else -0.5
        calibration = confidence if correct else -confidence
        accuracy += calibration

        efficiency = 0.1 * (remaining_budget / self.max_budget)

        false_positive = 0.0
        if flagged and not is_adversarial:
            false_positive = -self.fp_beta

        generalization = 0.0
        if is_adversarial and flagged:
            generalization = self.gen_gamma * RARE_ATTACK_WEIGHTS.get(attack_type, 0.0)

        return RewardVector(
            accuracy=accuracy,
            efficiency=efficiency,
            false_positive=false_positive,
            generalization=generalization,
        )


class LambdaSchedule:
    """
    Anneal query-efficiency penalty λ during training (spec §4.4).

    Starts small so the policy explores the full budget, then ramps up to
    penalize late probes.
    """

    def __init__(self, start: float = LAMBDA_EFF_START, end: float = LAMBDA_EFF_END):
        self.start = start
        self.end = end

    def __call__(self, progress: float) -> float:
        progress = float(np.clip(progress, 0.0, 1.0))
        return self.start + progress * (self.end - self.start)

    def apply(self, reward_fn: MORLReward, progress: float) -> float:
        value = self(progress)
        reward_fn.lambda_eff = value
        return value
