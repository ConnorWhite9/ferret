"""Input node — initialize episode from user image or defaults."""

from __future__ import annotations

import numpy as np
import torch

from ferret.episode import EpisodeState
from graph.state import FerretGraphDeps, FerretState, get_deps
from reward.morl_reward import sample_preference_vector


def input_node(state: FerretState, deps: FerretGraphDeps | None = None, config: dict | None = None) -> FerretState:
    """
    Initialize probing episode state.

    Expects either:
    - `state["image"]` (+ optional clean_image, label, preference), or
    - raises if no image is provided (caller must supply tensor).
    """
    deps = get_deps(config, deps=deps)

    image = state.get("image")
    if image is None:
        raise ValueError("input_node requires state['image'] as a CHW torch.Tensor.")

    if image.dim() == 4:
        image = image[0]
    image = image.detach().cpu().float()
    clean_image = state.get("clean_image")
    if clean_image is None:
        clean_image = image.clone()
    else:
        clean_image = clean_image.detach().cpu().float()
        if clean_image.dim() == 4:
            clean_image = clean_image[0]

    preference = state.get("preference")
    if preference is None:
        preference = sample_preference_vector()
    else:
        preference = np.asarray(preference, dtype=np.float32)
        if preference.shape != (4,):
            raise ValueError("preference must be a 4-d vector.")

    label = state.get("label")
    logits, features = deps.target_model(image)
    baseline_logits = logits.squeeze(0).detach().cpu()
    input_embedding = features.squeeze(0).detach().cpu()

    episode = EpisodeState(
        image=image,
        clean_image=clean_image,
        label=label,
        is_adversarial=state.get("is_adversarial"),
        attack_type=state.get("attack_type"),  # type: ignore[arg-type]
        input_embedding=input_embedding,
        baseline_logits=baseline_logits,
        preference=preference,
        remaining_budget=deps.max_budget,
    )
    deps.confidence.reset(baseline_logits)
    episode.confidence = deps.confidence.score

    return {
        "episode": episode,
        "modality": state.get("modality"),
        "last_action": None,
        "decision": None,
        "done": False,
        "flagged": None,
        "probes_used": 0,
    }
