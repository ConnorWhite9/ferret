"""LangGraph state schema and runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, TypedDict

import numpy as np
import torch

from agents.confidence import ConfidenceAggregator
from agents.decision import DecisionNode
from agents.probe_executor import ProbeExecutor
from ferret.episode import EpisodeState
from policy.trunk import FerretAgent
from policy.vision_encoder import VisionEncoder

Modality = Literal["vision", "language"]


class FerretState(TypedDict, total=False):
    """Graph state passed between nodes."""

    episode: EpisodeState
    modality: Modality
    last_action: Optional[int]
    decision: Optional[str]
    done: bool
    flagged: Optional[bool]
    probes_used: int
    # Optional inputs for input_node (inference).
    image: Optional[torch.Tensor]
    clean_image: Optional[torch.Tensor]
    label: Optional[int]
    preference: Optional[np.ndarray]
    # Eval-only ground truth metadata.
    is_adversarial: Optional[bool]
    attack_type: Optional[str]


@dataclass
class FerretGraphDeps:
    """Shared runtime dependencies injected via LangGraph configurable."""

    target_model: VisionEncoder
    agent: FerretAgent
    probe_executor: ProbeExecutor
    confidence: ConfidenceAggregator
    decision_node: DecisionNode
    max_budget: int
    device: torch.device
    deterministic_policy: bool = True

    @classmethod
    def from_models(
        cls,
        target_model: VisionEncoder,
        agent: FerretAgent,
        *,
        max_budget: int | None = None,
        device: torch.device | None = None,
        decision_threshold: float | None = None,
        deterministic_policy: bool = True,
    ) -> FerretGraphDeps:
        from ferret.constants import DECISION_THRESHOLD, MAX_BUDGET

        device = device or target_model.device
        agent = agent.to(device)
        agent.eval()
        decision_node = DecisionNode(
            threshold=decision_threshold if decision_threshold is not None else DECISION_THRESHOLD
        )
        return cls(
            target_model=target_model,
            agent=agent,
            probe_executor=ProbeExecutor(target_model),
            confidence=ConfidenceAggregator(),
            decision_node=decision_node,
            max_budget=max_budget or MAX_BUDGET,
            device=device,
            deterministic_policy=deterministic_policy,
        )


def get_deps(config: dict | None = None, deps: FerretGraphDeps | None = None) -> FerretGraphDeps:
    """Resolve deps from explicit injection (preferred) or LangGraph configurable."""
    if deps is not None:
        return deps
    if config:
        configurable = config.get("configurable") or {}
        found = configurable.get("deps")
        if found is not None:
            return found
    raise ValueError("FerretGraphDeps were not provided to the graph node.")
