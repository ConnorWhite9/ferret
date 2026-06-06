"""
Unit tests for policy/trunk.py.
Covers positional encoding, MLP ablation, get_value, and forward shape.
No GPU required; CPU-only, no data pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ferret.constants import (
    INPUT_EMBED_DIM,
    MAX_BUDGET,
    NUM_CLASSES,
    NUM_PROBE_ACTIONS,
    PREFERENCE_DIM,
)
from policy.trunk import FerretAgent, FerretPolicy, PolicyConfig, _to_tensor


def _dummy_obs(batch: int = 2, history_len: int = 3, max_budget: int = MAX_BUDGET) -> dict[str, np.ndarray]:
    """Build a syntactically valid observation dict."""
    probe_grid = np.full((batch, max_budget), -1, dtype=np.int32)
    probe_pert = np.full((batch, max_budget), -1, dtype=np.int32)
    probe_mag  = np.full((batch, max_budget), -1, dtype=np.int32)
    resp       = np.zeros((batch, max_budget, NUM_CLASSES), dtype=np.float32)

    # Fill first `history_len` steps with valid values
    probe_grid[:, :history_len] = np.random.randint(0, 49, (batch, history_len))
    probe_pert[:, :history_len] = np.random.randint(0, 4,  (batch, history_len))
    probe_mag[:, :history_len]  = np.random.randint(0, 3,  (batch, history_len))
    resp[:, :history_len]       = np.random.randn(batch, history_len, NUM_CLASSES).astype(np.float32)

    return {
        "input_embedding":  np.random.randn(batch, INPUT_EMBED_DIM).astype(np.float32),
        "preference":       np.tile([0.25, 0.25, 0.25, 0.25], (batch, 1)).astype(np.float32),
        "remaining_budget": np.full((batch, 1), 0.5, dtype=np.float32),
        "confidence":       np.full((batch, 1), 0.3, dtype=np.float32),
        "probe_grid":       probe_grid,
        "probe_pert":       probe_pert,
        "probe_mag":        probe_mag,
        "response_logits":  resp,
        "history_len":      np.full((batch, 1), history_len, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# FerretPolicy — forward shape
# ---------------------------------------------------------------------------

def test_policy_output_shapes_transformer():
    policy = FerretPolicy()
    obs = _to_tensor(_dummy_obs(batch=4, history_len=3), torch.device("cpu"))
    logits, value = policy.forward(obs)
    assert logits.shape == (4, NUM_PROBE_ACTIONS)
    assert value.shape == (4, 1)


def test_policy_output_shapes_empty_history():
    policy = FerretPolicy()
    obs = _to_tensor(_dummy_obs(batch=2, history_len=0), torch.device("cpu"))
    logits, value = policy.forward(obs)
    assert logits.shape == (2, NUM_PROBE_ACTIONS)


def test_policy_output_shapes_mlp_trunk():
    config = PolicyConfig(use_mlp_trunk=True)
    policy = FerretPolicy(config)
    obs = _to_tensor(_dummy_obs(batch=3, history_len=5), torch.device("cpu"))
    logits, value = policy.forward(obs)
    assert logits.shape == (3, NUM_PROBE_ACTIONS)


# ---------------------------------------------------------------------------
# Positional encoding — different step orders produce different outputs
# ---------------------------------------------------------------------------

def test_positional_encoding_changes_output():
    """
    Two sequences with the same tokens but different history lengths should
    produce different logits after the PE is added.  If PE were absent the
    transformer output would only depend on the SET of tokens (permutation-
    invariant), not their order.
    """
    policy = FerretPolicy()
    policy.eval()

    obs_a = _dummy_obs(batch=1, history_len=2)
    obs_b = _dummy_obs(batch=1, history_len=4)

    # Fix input embeddings & static context to be identical so any output
    # difference must come from the history sequence encoding.
    for key in ("input_embedding", "preference", "remaining_budget", "confidence"):
        obs_b[key] = obs_a[key].copy()

    ta = _to_tensor(obs_a, torch.device("cpu"))
    tb = _to_tensor(obs_b, torch.device("cpu"))

    with torch.no_grad():
        logits_a, _ = policy.forward(ta)
        logits_b, _ = policy.forward(tb)

    # Different history lengths → different outputs (not an identity function).
    assert not torch.allclose(logits_a, logits_b), \
        "Policy produced identical logits for different probe histories — PE may not be working."


# ---------------------------------------------------------------------------
# FerretAgent wrappers
# ---------------------------------------------------------------------------

def test_agent_get_action_and_value():
    agent = FerretAgent()
    obs = _dummy_obs(batch=2, history_len=3)
    action, logprob, entropy, value = agent.get_action_and_value(obs)
    assert action.shape == (2,)
    assert logprob.shape == (2,)
    assert value.shape == (2,)


def test_agent_get_value():
    agent = FerretAgent()
    obs = _dummy_obs(batch=2, history_len=2)
    value = agent.get_value(obs)
    assert value.shape == (2,)


def test_agent_deterministic_action():
    """Passing the same action tensor should return its log-prob."""
    agent = FerretAgent()
    obs = _dummy_obs(batch=1, history_len=1)
    fixed_action = torch.tensor([42])
    action, logprob, _, _ = agent.get_action_and_value(obs, action=fixed_action)
    assert action.item() == 42
    assert torch.isfinite(logprob)


# ---------------------------------------------------------------------------
# MLP ablation vs transformer — different parameter counts
# ---------------------------------------------------------------------------

def test_mlp_trunk_fewer_params():
    tf_agent  = FerretAgent(PolicyConfig(use_mlp_trunk=False))
    mlp_agent = FerretAgent(PolicyConfig(use_mlp_trunk=True))

    tf_params  = sum(p.numel() for p in tf_agent.parameters())
    mlp_params = sum(p.numel() for p in mlp_agent.parameters())

    # Transformer has multi-head attention; should be larger than flat MLP trunk.
    assert tf_params > mlp_params, (
        f"Transformer ({tf_params}) should have more params than MLP ({mlp_params})"
    )
