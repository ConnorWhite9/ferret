"""Confidence aggregator node — rule-based adversarial score update."""

from __future__ import annotations

from graph.state import FerretGraphDeps, FerretState, get_deps


def confidence_node(state: FerretState, deps: FerretGraphDeps | None = None, config: dict | None = None) -> FerretState:
    deps = get_deps(config, deps=deps)
    episode = state["episode"]
    if episode is None or not episode.response_logits:
        raise RuntimeError("confidence_node requires episode with at least one probe response.")

    step_index = len(episode.response_logits) - 1
    latest_logits = episode.response_logits[-1]
    episode.confidence = deps.confidence.update(latest_logits, step_index)
    return {"episode": episode}
