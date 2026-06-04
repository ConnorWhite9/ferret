"""Probing policy node — frozen RL policy selects the next probe action."""

from __future__ import annotations

import torch

from ferret.episode import build_policy_obs
from graph.state import FerretGraphDeps, FerretState, get_deps


def policy_node(state: FerretState, deps: FerretGraphDeps | None = None, config: dict | None = None) -> FerretState:
    deps = get_deps(config, deps=deps)
    episode = state["episode"]
    if episode is None:
        raise RuntimeError("policy_node requires initialized episode.")

    obs = build_policy_obs(episode, max_budget=deps.max_budget)
    with torch.no_grad():
        if deps.deterministic_policy:
            batch = _obs_to_device(obs, deps.device)
            logits, _ = deps.agent.policy.forward(batch)
            action = int(logits.argmax(dim=-1).item())
        else:
            action_t, _, _, _ = deps.agent.get_action_and_value(obs)
            action = int(action_t.item())

    return {"last_action": action}


def _obs_to_device(obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for key, value in obs.items():
        tensor = torch.as_tensor(value, device=device)
        if key in {"probe_grid", "probe_pert", "probe_mag", "history_len"}:
            tensor = tensor.long()
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.float()
            if key == "input_embedding" and tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            elif key == "preference" and tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            elif key in {"remaining_budget", "confidence", "history_len"} and tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            elif key in {"probe_grid", "probe_pert", "probe_mag"} and tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            elif key == "response_logits" and tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
        batch[key] = tensor
    return batch
