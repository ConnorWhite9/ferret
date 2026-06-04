"""Rule-based adversarial confidence aggregator."""

from __future__ import annotations

import torch


class ConfidenceAggregator:
    """
    Weighted accumulator over L2 logit shifts from the episode baseline.

    The score is squashed to [0, 1] with a sigmoid so the decision node can
    threshold it without learning (prevents reward hacking).
    """

    def __init__(self, temperature: float = 4.0):
        self.temperature = temperature
        self._raw_score = 0.0
        self._baseline_logits: torch.Tensor | None = None

    def reset(self, baseline_logits: torch.Tensor) -> float:
        self._baseline_logits = baseline_logits.detach().cpu().float()
        self._raw_score = 0.0
        return self.score

    @property
    def score(self) -> float:
        return float(torch.sigmoid(torch.tensor(self._raw_score / self.temperature)).item())

    def update(self, probe_logits: torch.Tensor, step_index: int) -> float:
        if self._baseline_logits is None:
            raise RuntimeError("ConfidenceAggregator.reset must be called before update.")
        shift = torch.norm(
            probe_logits.detach().cpu().float() - self._baseline_logits,
            p=2,
        ).item()
        weight = 1.0 / (step_index + 1)
        self._raw_score += weight * shift
        return self.score
