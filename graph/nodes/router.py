"""Modality router — select vision vs language backends (Phase 2)."""

from __future__ import annotations

from graph.state import FerretState, Modality


def modality_router(state: FerretState, deps=None, config: dict | None = None) -> FerretState:
    """
    Detect modality and attach routing metadata.

    Phase 1: vision-only. Language path is reserved for Phase 2.
    """
    modality: Modality = state.get("modality") or "vision"
    if modality != "vision":
        raise NotImplementedError(f"Modality '{modality}' is not implemented yet.")
    return {"modality": modality}
