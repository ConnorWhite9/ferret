"""Checkpoint save / load with full training state for resume."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch
import torch.optim as optim

from policy.trunk import FerretAgent


def save_checkpoint(
    path: Path,
    agent: FerretAgent,
    optimizer: optim.Optimizer,
    args,
    global_step: int,
    update: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "update": update,
            "args": dataclasses.asdict(args),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    agent: FerretAgent,
    optimizer: optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> tuple[int, int, dict]:
    """
    Load checkpoint into agent (and optionally optimizer).

    Returns:
        global_step, update, saved_args_dict
    """
    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=False)

    # Backwards-compatible: plain state-dict or full payload
    if isinstance(payload, dict) and "agent" in payload:
        agent.load_state_dict(payload["agent"])
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        global_step = int(payload.get("global_step", 0))
        update = int(payload.get("update", 0))
        saved_args = payload.get("args", {})
    else:
        agent.load_state_dict(payload)
        global_step = 0
        update = 0
        saved_args = {}

    return global_step, update, saved_args


def latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the most recently modified policy.pt under run_dir, or None."""
    candidates = sorted(run_dir.rglob("policy.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None
