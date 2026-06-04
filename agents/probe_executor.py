"""Vision probe executor — applies perturbations and queries the target model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ferret.constants import (
    CELL_SIZE,
    GRID_SIDE,
    MAGNITUDE_SCALES,
    NUM_MAGNITUDES,
    NUM_PERT_TYPES,
)
from policy.vision_encoder import VisionEncoder


@dataclass(frozen=True)
class ProbeAction:
    grid_cell: int
    perturbation_type: int
    magnitude: int


@dataclass(frozen=True)
class ProbeResult:
    action: ProbeAction
    logits: torch.Tensor
    probed_image: torch.Tensor


def decode_action(action: int) -> ProbeAction:
    if action < 0:
        raise ValueError(f"Invalid action index: {action}")
    magnitude = action % NUM_MAGNITUDES
    action //= NUM_MAGNITUDES
    perturbation_type = action % NUM_PERT_TYPES
    action //= NUM_PERT_TYPES
    grid_cell = action
    return ProbeAction(grid_cell, perturbation_type, magnitude)


def encode_action(action: ProbeAction) -> int:
    return (
        action.grid_cell * NUM_PERT_TYPES * NUM_MAGNITUDES
        + action.perturbation_type * NUM_MAGNITUDES
        + action.magnitude
    )


def _cell_slice(grid_cell: int) -> tuple[slice, slice]:
    row = grid_cell // GRID_SIDE
    col = grid_cell % GRID_SIDE
    y0 = row * CELL_SIZE
    x0 = col * CELL_SIZE
    return slice(y0, y0 + CELL_SIZE), slice(x0, x0 + CELL_SIZE)


class ProbeExecutor:
    """Apply a discrete probe action to an image and query the target model."""

    def __init__(self, target_model: VisionEncoder):
        self.target_model = target_model

    def execute(
        self,
        image: torch.Tensor,
        action: int | ProbeAction,
        label: int | None = None,
    ) -> ProbeResult:
        probe_action = action if isinstance(action, ProbeAction) else decode_action(action)
        perturbed = self._apply_probe(image, probe_action, label)
        logits = self.target_model.logits(perturbed).squeeze(0).detach().cpu()
        return ProbeResult(
            action=probe_action,
            logits=logits,
            probed_image=perturbed.detach().cpu(),
        )

    def _apply_probe(
        self,
        image: torch.Tensor,
        action: ProbeAction,
        label: int | None,
    ) -> torch.Tensor:
        image = image.clone()
        ys, xs = _cell_slice(action.grid_cell)
        scale = MAGNITUDE_SCALES[action.magnitude]

        if action.perturbation_type == 0:
            return self._fgsm_step(image, ys, xs, scale, label)
        if action.perturbation_type == 1:
            return self._gaussian_noise(image, ys, xs, scale)
        if action.perturbation_type == 2:
            return self._occlusion(image, ys, xs)
        if action.perturbation_type == 3:
            return self._mean_fill(image, ys, xs)
        raise ValueError(f"Unknown perturbation type: {action.perturbation_type}")

    def _fgsm_step(
        self,
        image: torch.Tensor,
        ys: slice,
        xs: slice,
        scale: float,
        label: int | None,
    ) -> torch.Tensor:
        if label is None:
            return self._gaussian_noise(image, ys, xs, scale)

        work = image.detach().clone().requires_grad_(True)
        logits = self.target_model.logits_with_grad(work)
        loss = F.cross_entropy(logits, torch.tensor([label], device=logits.device))
        loss.backward()
        grad = work.grad.detach()
        step = scale * grad.sign()
        work = work + step
        patch = work[:, ys, xs] - image[:, ys, xs]
        out = image.clone()
        out[:, ys, xs] = image[:, ys, xs] + patch
        return out.detach()

    def _gaussian_noise(
        self,
        image: torch.Tensor,
        ys: slice,
        xs: slice,
        scale: float,
    ) -> torch.Tensor:
        out = image.clone()
        patch = image[:, ys, xs]
        noise = torch.randn_like(patch) * scale
        out[:, ys, xs] = patch + noise
        return out

    def _occlusion(self, image: torch.Tensor, ys: slice, xs: slice) -> torch.Tensor:
        out = image.clone()
        out[:, ys, xs] = 0.0
        return out

    def _mean_fill(self, image: torch.Tensor, ys: slice, xs: slice) -> torch.Tensor:
        out = image.clone()
        fill = image.mean(dim=(1, 2), keepdim=True).expand_as(image[:, ys, xs])
        out[:, ys, xs] = fill
        return out
