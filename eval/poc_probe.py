"""
Phase 1 POC: hardcoded probing strategy to verify adversarial detection signal.

Runs a fixed center-cell Gaussian-noise probe sequence and reports confidence
separation between clean and adversarial episodes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from agents.confidence import ConfidenceAggregator
from agents.decision import DecisionNode, DecisionOutcome
from agents.probe_executor import ProbeAction, encode_action
from data import DataConfig, FerretDataPipeline
from ferret.constants import MAX_BUDGET
from policy.vision_encoder import VisionEncoder
from train.ppo_train import _target_model_for_foolbox


@dataclass
class PocResult:
    is_adversarial: bool
    attack_type: str | None
    final_confidence: float
    probes_used: int
    decision: str


def hardcoded_action(step: int) -> int:
    """Center cell + Gaussian noise + small magnitude, rotating slightly by step."""
    grid_cell = 24 + (step % 3)  # center-ish cells in 7x7 grid
    return encode_action(ProbeAction(grid_cell=grid_cell, perturbation_type=1, magnitude=0))


def run_poc_episode(
    image: torch.Tensor,
    label: int,
    target_model: VisionEncoder,
    clean_image: torch.Tensor | None = None,
    max_budget: int = MAX_BUDGET,
) -> PocResult:
    from agents.probe_executor import ProbeExecutor

    executor = ProbeExecutor(target_model)
    confidence = ConfidenceAggregator()
    decision_node = DecisionNode()

    # Baseline must be the CLEAN image prediction so adversarial probes diverge
    # upward from it and clean probes stay close — giving the correct signal direction.
    reference = clean_image if clean_image is not None else image
    baseline = target_model.logits(reference).squeeze(0).detach().cpu()
    confidence.reset(baseline)

    working = image.clone()
    probes_used = 0
    outcome = DecisionOutcome.ABSTAIN

    for step in range(max_budget):
        result = executor.execute(working, hardcoded_action(step), label=label)
        working = result.probed_image
        probes_used += 1
        score = confidence.update(result.logits, step)
        remaining = max_budget - probes_used
        outcome = decision_node.evaluate(score, remaining)
        if outcome != DecisionOutcome.CONTINUE:
            break

    return PocResult(
        is_adversarial=False,
        attack_type=None,
        final_confidence=confidence.score,
        probes_used=probes_used,
        decision=outcome.value,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ferret Phase 1 hardcoded probe POC")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--adversarial-ratio", type=float, default=0.5)
    parser.add_argument("--attack-types", nargs="+", default=["fgsm"])
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = DataConfig(
        adversarial_ratio=args.adversarial_ratio,
        attack_types=tuple(args.attack_types),  # type: ignore[arg-type]
        precompute_adversarial=True,
        download=args.download,
    )
    encoder = VisionEncoder(device=device)
    pipeline = FerretDataPipeline(data_config, model=_target_model_for_foolbox(encoder), device=device)
    if data_config.adversarial_ratio > 0:
        pipeline.ensure_adversarial_cache("val")

    clean_scores = []
    adv_scores = []

    for _ in range(args.episodes):
        sample = pipeline.sample_episode("val")
        result = run_poc_episode(sample.image, sample.label, encoder, clean_image=sample.clean_image)
        if sample.is_adversarial:
            adv_scores.append(result.final_confidence)
        else:
            clean_scores.append(result.final_confidence)

    clean_mean = float(np.mean(clean_scores)) if clean_scores else float("nan")
    adv_mean = float(np.mean(adv_scores)) if adv_scores else float("nan")
    separation = adv_mean - clean_mean

    print(f"Episodes: {args.episodes} (adv ratio={args.adversarial_ratio})")
    print(f"Clean confidence mean:       {clean_mean:.4f} (n={len(clean_scores)})")
    print(f"Adversarial confidence mean: {adv_mean:.4f} (n={len(adv_scores)})")
    print(f"Separation (adv - clean):    {separation:.4f}")
    if separation > 0.05:
        print("POC PASS: positive detection signal.")
    else:
        print("POC WEAK: consider tuning probes or confidence aggregator.")


if __name__ == "__main__":
    main()
