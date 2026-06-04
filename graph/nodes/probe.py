"""Probe executor node — apply perturbation and query target model."""

from __future__ import annotations

from graph.state import FerretGraphDeps, FerretState, get_deps


def probe_node(state: FerretState, deps: FerretGraphDeps | None = None, config: dict | None = None) -> FerretState:
    deps = get_deps(config, deps=deps)
    episode = state["episode"]
    action = state.get("last_action")
    if episode is None or action is None:
        raise RuntimeError("probe_node requires episode and last_action.")

    result = deps.probe_executor.execute(episode.image, action, label=episode.label)
    episode.image = result.probed_image
    episode.remaining_budget -= 1
    episode.probe_grid.append(result.action.grid_cell)
    episode.probe_pert.append(result.action.perturbation_type)
    episode.probe_mag.append(result.action.magnitude)
    episode.response_logits.append(result.logits)

    return {
        "episode": episode,
        "probes_used": len(episode.probe_grid),
    }
