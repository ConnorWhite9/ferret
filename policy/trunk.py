"""Transformer policy trunk and CleanRL-compatible agent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from ferret.constants import (
    INPUT_EMBED_DIM,
    MAX_BUDGET,
    NUM_CLASSES,
    NUM_GRID_CELLS,
    NUM_MAGNITUDES,
    NUM_PERT_TYPES,
    NUM_PROBE_ACTIONS,
    POLICY_D_MODEL,
    POLICY_MLP_HIDDEN,
    POLICY_NHEAD,
    POLICY_NLAYERS,
    PREFERENCE_DIM,
)


@dataclass
class PolicyConfig:
    d_model: int = POLICY_D_MODEL
    nhead: int = POLICY_NHEAD
    num_layers: int = POLICY_NLAYERS
    mlp_hidden: int = POLICY_MLP_HIDDEN
    max_budget: int = MAX_BUDGET
    num_actions: int = NUM_PROBE_ACTIONS


def _to_tensor(obs: dict[str, np.ndarray | torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in obs.items():
        if isinstance(value, torch.Tensor):
            tensor = value.to(device)
        else:
            tensor = torch.as_tensor(value, device=device)

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

        if key in {"probe_grid", "probe_pert", "probe_mag", "history_len"}:
            tensor = tensor.long()
        else:
            tensor = tensor.float()
        out[key] = tensor
    return out


class FerretPolicy(nn.Module):
    """
    Shared trunk from spec section 6.1:
    probe/response history -> transformer -> concat(static) -> MLP actor/critic.
    """

    def __init__(self, config: PolicyConfig | None = None):
        super().__init__()
        self.config = config or PolicyConfig()
        d = self.config.d_model

        self.grid_embed = nn.Embedding(NUM_GRID_CELLS + 1, d, padding_idx=NUM_GRID_CELLS)
        self.pert_embed = nn.Embedding(NUM_PERT_TYPES + 1, d, padding_idx=NUM_PERT_TYPES)
        self.mag_embed = nn.Embedding(NUM_MAGNITUDES + 1, d, padding_idx=NUM_MAGNITUDES)
        self.response_proj = nn.Linear(NUM_CLASSES, d)
        self.empty_token = nn.Parameter(torch.zeros(1, 1, d))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.config.nhead,
            dim_feedforward=d * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers,
        )

        static_in = INPUT_EMBED_DIM + 1 + PREFERENCE_DIM + 1
        self.static_proj = nn.Sequential(
            nn.Linear(static_in, d),
            nn.GELU(),
        )

        head_in = d * 2
        self.actor = nn.Sequential(
            nn.Linear(head_in, self.config.mlp_hidden),
            nn.Tanh(),
            nn.Linear(self.config.mlp_hidden, self.config.num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(head_in, self.config.mlp_hidden),
            nn.Tanh(),
            nn.Linear(self.config.mlp_hidden, 1),
        )

    def _embed_history(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        grid = obs["probe_grid"].long()
        pert = obs["probe_pert"].long()
        mag = obs["probe_mag"].long()
        responses = obs["response_logits"].float()

        grid = torch.where(grid < 0, torch.full_like(grid, NUM_GRID_CELLS), grid)
        pert = torch.where(pert < 0, torch.full_like(pert, NUM_PERT_TYPES), pert)
        mag = torch.where(mag < 0, torch.full_like(mag, NUM_MAGNITUDES), mag)

        probe_tokens = self.grid_embed(grid) + self.pert_embed(pert) + self.mag_embed(mag)
        response_tokens = self.response_proj(responses)
        seq = probe_tokens + response_tokens

        history_len = obs["history_len"].long().squeeze(-1)
        max_len = seq.shape[1]
        positions = torch.arange(max_len, device=seq.device).unsqueeze(0)
        padding_mask = positions >= history_len.unsqueeze(1)
        return seq, padding_mask

    def _pool_sequence(self, seq: torch.Tensor, padding_mask: torch.Tensor, history_len: torch.Tensor) -> torch.Tensor:
        batch_size = seq.shape[0]
        pooled = self.empty_token.reshape(1, -1).expand(batch_size, -1).clone()
        has_history = history_len > 0
        if has_history.any():
            idx = (history_len[has_history] - 1).view(-1, 1, 1).expand(-1, 1, seq.shape[-1])
            gathered = seq[has_history].gather(1, idx).squeeze(1)
            pooled[has_history] = gathered
        return pooled

    def _static_context(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        static = torch.cat(
            [
                obs["input_embedding"],
                obs["remaining_budget"],
                obs["preference"],
                obs["confidence"],
            ],
            dim=-1,
        )
        return self.static_proj(static)

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        history_len = obs["history_len"].long().squeeze(-1)
        batch_size = history_len.shape[0]

        if (history_len == 0).all():
            pooled = self.empty_token.reshape(-1).expand(batch_size, -1)
        else:
            seq, padding_mask = self._embed_history(obs)
            # Transformer requires at least one non-masked token per batch row.
            no_history = history_len == 0
            if no_history.any():
                padding_mask = padding_mask.clone()
                padding_mask[no_history, 0] = False
            encoded = self.transformer(seq, src_key_padding_mask=padding_mask)
            pooled = self._pool_sequence(encoded, padding_mask, history_len)
            if no_history.any():
                empty = self.empty_token.reshape(-1)
                pooled = pooled.clone()
                pooled[no_history] = empty

        context = torch.cat([pooled, self._static_context(obs)], dim=-1)
        return self.actor(context), self.critic(context)

    def get_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        _, value = self.forward(obs)
        return value.squeeze(-1)

    def get_action_and_value(
        self,
        obs: dict[str, np.ndarray | torch.Tensor],
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        batch = _to_tensor(obs, device)
        logits, value = self.forward(batch)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)


class FerretAgent(nn.Module):
    """CleanRL wrapper around FerretPolicy."""

    def __init__(self, config: PolicyConfig | None = None):
        super().__init__()
        self.policy = FerretPolicy(config)

    def get_action_and_value(
        self,
        obs: dict[str, np.ndarray | torch.Tensor],
        action: torch.Tensor | None = None,
    ):
        return self.policy.get_action_and_value(obs, action=action)
