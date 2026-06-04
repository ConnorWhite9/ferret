"""
Preference-conditioned MORL scalarization (MORL-Baselines–compatible pattern).

Ferret uses linear scalarization with a sampled preference vector per episode,
matching the envelope / conditioned policy approach from MORL-Baselines without
requiring a separate training stack on top of CleanRL PPO.

Reference: https://github.com/LucasAlegre/morl-baselines
"""

from __future__ import annotations

import numpy as np

from ferret.constants import MORL_OBJECTIVE_NAMES, PREFERENCE_DIM
from reward.morl_reward import RewardNormalizer, RewardVector, sample_preference_vector, scalarize


class PreferenceConditionedScalarization:
    """
    Wraps reward normalization + dot(preference, reward_vector) for PPO.

    Equivalent role to MORL-Baselines linear scalarization with a
    preference-conditioned policy input.
    """

    def __init__(self, normalizer: RewardNormalizer | None = None):
        self.normalizer = normalizer or RewardNormalizer()

    def sample_preference(self, rng: np.random.Generator | None = None) -> np.ndarray:
        return sample_preference_vector(rng)

    def scalarize(
        self,
        reward_vector: RewardVector | np.ndarray,
        preference: np.ndarray,
        *,
        update_normalizer: bool = True,
    ) -> tuple[float, np.ndarray]:
        raw = reward_vector.as_array() if isinstance(reward_vector, RewardVector) else reward_vector
        normalized = self.normalizer.normalize(raw, update=update_normalizer)
        return scalarize(normalized, preference), normalized

    @staticmethod
    def objective_names() -> tuple[str, ...]:
        return MORL_OBJECTIVE_NAMES

    @staticmethod
    def preference_dim() -> int:
        return PREFERENCE_DIM
