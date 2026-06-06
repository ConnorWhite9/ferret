"""Smoke tests for checkpoint save/load round-trip."""

import tempfile
from pathlib import Path

import torch
import torch.optim as optim

from policy.trunk import FerretAgent
from train.checkpoint import save_checkpoint, load_checkpoint
from train.ppo_train import Args


def test_checkpoint_roundtrip():
    agent = FerretAgent()
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    args = Args(exp_name="test", seed=7)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "policy.pt"
        save_checkpoint(path, agent, optimizer, args, global_step=1000, update=5)
        assert path.exists()

        agent2 = FerretAgent()
        opt2 = optim.Adam(agent2.parameters(), lr=1e-3)
        gs, upd, saved_args = load_checkpoint(path, agent2, opt2)

        assert gs == 1000
        assert upd == 5
        assert saved_args["seed"] == 7

        # Weights should match
        for p1, p2 in zip(agent.parameters(), agent2.parameters()):
            assert torch.allclose(p1.data, p2.data)
