"""LangGraph state machine for Ferret inference-time sequential probing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from ferret.constants import PREFERENCE_DIM
from ferret.episode import EpisodeState, build_policy_obs
from graph.nodes import (
    confidence_node,
    decision_node,
    input_node,
    modality_router,
    policy_node,
    probe_node,
    route_after_decision,
)
from graph.state import FerretGraphDeps, FerretState
from policy.trunk import FerretAgent, PolicyConfig
from policy.vision_encoder import VisionEncoder

try:
    from langgraph.graph import END, StateGraph
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install langgraph: pip install langgraph") from exc


@dataclass
class DetectionResult:
    """Outcome of a full probing episode."""

    flagged: bool
    decision: str
    confidence: float
    probes_used: int
    probe_history: list[tuple[int, int, int]]
    preference: np.ndarray
    episode: EpisodeState


def _bind(node_fn, deps: FerretGraphDeps):
    def bound(state: FerretState) -> FerretState:
        return node_fn(state, deps=deps)

    return bound


def build_ferret_graph(deps: FerretGraphDeps) -> StateGraph:
    """Construct the Ferret probing StateGraph (uncompiled)."""
    graph = StateGraph(FerretState)

    graph.add_node("input", _bind(input_node, deps))
    graph.add_node("router", _bind(modality_router, deps))
    graph.add_node("policy", _bind(policy_node, deps))
    graph.add_node("probe", _bind(probe_node, deps))
    graph.add_node("confidence", _bind(confidence_node, deps))
    graph.add_node("decision", _bind(decision_node, deps))

    graph.set_entry_point("input")
    graph.add_edge("input", "router")
    graph.add_edge("router", "policy")
    graph.add_edge("policy", "probe")
    graph.add_edge("probe", "confidence")
    graph.add_edge("confidence", "decision")
    graph.add_conditional_edges("decision", route_after_decision, ["policy", END])

    return graph


def compile_ferret_agent(deps: FerretGraphDeps):
    """Compile the LangGraph application with injected dependencies."""
    return build_ferret_graph(deps).compile()


def load_policy_checkpoint(
    agent: FerretAgent,
    checkpoint_path: str | Path,
    device: torch.device,
) -> FerretAgent:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    state_dict = payload["agent"] if isinstance(payload, dict) and "agent" in payload else payload
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


class FerretDetector:
    """High-level inference API over the LangGraph probing loop."""

    def __init__(self, deps: FerretGraphDeps):
        self.deps = deps
        self.app = compile_ferret_agent(deps)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = None,
        policy_config: PolicyConfig | None = None,
        deterministic_policy: bool = True,
        decision_threshold: float | None = None,
    ) -> FerretDetector:
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        target_model = VisionEncoder(device=device)
        agent = FerretAgent(policy_config)
        load_policy_checkpoint(agent, checkpoint_path, device)
        deps = FerretGraphDeps.from_models(
            target_model,
            agent,
            device=device,
            deterministic_policy=deterministic_policy,
            decision_threshold=decision_threshold,
        )
        return cls(deps)

    @classmethod
    def from_models(
        cls,
        target_model: VisionEncoder,
        agent: FerretAgent,
        *,
        deterministic_policy: bool = True,
        decision_threshold: float | None = None,
    ) -> FerretDetector:
        deps = FerretGraphDeps.from_models(
            target_model,
            agent,
            deterministic_policy=deterministic_policy,
            decision_threshold=decision_threshold,
        )
        return cls(deps)

    def _run_config(self) -> dict:
        return {"configurable": {"deps": self.deps}}

    def detect(
        self,
        image: torch.Tensor,
        *,
        clean_image: torch.Tensor | None = None,
        label: int | None = None,
        preference: np.ndarray | None = None,
        is_adversarial: bool | None = None,
        attack_type: str | None = None,
    ) -> DetectionResult:
        """Run the full probe loop until flag or abstain."""
        final_state = self.app.invoke(
            self._initial_state(
                image=image,
                clean_image=clean_image,
                label=label,
                preference=preference,
                is_adversarial=is_adversarial,
                attack_type=attack_type,
            ),
            config=self._run_config(),
        )
        return self._result_from_state(final_state)

    def stream(
        self,
        image: torch.Tensor,
        *,
        clean_image: torch.Tensor | None = None,
        label: int | None = None,
        preference: np.ndarray | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream per-node outputs for debugging or UI."""
        yield from self.app.stream(
            self._initial_state(
                image=image,
                clean_image=clean_image,
                label=label,
                preference=preference,
            ),
            config=self._run_config(),
            stream_mode="updates",
        )

    def build_observation(self, episode: EpisodeState) -> dict[str, np.ndarray]:
        return build_policy_obs(episode, max_budget=self.deps.max_budget)

    @staticmethod
    def _initial_state(
        *,
        image: torch.Tensor,
        clean_image: torch.Tensor | None = None,
        label: int | None = None,
        preference: np.ndarray | None = None,
        is_adversarial: bool | None = None,
        attack_type: str | None = None,
    ) -> FerretState:
        if preference is not None:
            preference = np.asarray(preference, dtype=np.float32)
            if preference.shape != (PREFERENCE_DIM,):
                raise ValueError(f"preference must have shape ({PREFERENCE_DIM},)")
        return {
            "image": image,
            "clean_image": clean_image,
            "label": label,
            "preference": preference,
            "is_adversarial": is_adversarial,
            "attack_type": attack_type,
            "modality": "vision",
        }

    @staticmethod
    def _result_from_state(state: FerretState) -> DetectionResult:
        episode = state["episode"]
        if episode is None:
            raise RuntimeError("Graph finished without episode state.")
        flagged = bool(state.get("flagged"))
        decision = state.get("decision") or "abstain"
        history = list(zip(episode.probe_grid, episode.probe_pert, episode.probe_mag))
        return DetectionResult(
            flagged=flagged,
            decision=decision,
            confidence=episode.confidence,
            probes_used=len(episode.probe_grid),
            probe_history=history,
            preference=episode.preference.copy(),
            episode=episode,
        )


def build_detector(
    checkpoint_path: str | Path | None = None,
    *,
    agent: FerretAgent | None = None,
    target_model: VisionEncoder | None = None,
    device: str | torch.device | None = None,
) -> FerretDetector:
    if checkpoint_path is not None:
        return FerretDetector.from_checkpoint(checkpoint_path, device=device)
    if agent is not None and target_model is not None:
        return FerretDetector.from_models(target_model, agent)
    raise ValueError("Provide checkpoint_path or (agent, target_model).")
