"""
Ablation study runner — spec §10.4.

Ablations implemented:
  A1. MORL (Dirichlet preference) vs single-objective (fixed equal weights)
  A2. Transformer trunk vs MLP trunk  (use_mlp_trunk=True)
  A3. Sequential probing vs single-probe baseline  (max_budget varied)
  A4. Budget sizes: 5, 10, 20 probes

Each ablation variant trains a fresh policy from the same random seed (or loads
a checkpoint), evaluates it, and saves a JSON row to eval/results/ablations/.

Usage:
  # Run all ablations with a trained checkpoint (eval-only, no re-training):
  python -m eval.ablation --checkpoint runs/.../policy.pt --episodes 200

  # Or specify a subset:
  python -m eval.ablation --ablations A1 A3 --checkpoint ...
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data import DataConfig, FerretDataPipeline
from eval.metrics import DetectorMetrics, EpisodeRecord, compute_metrics, print_metrics
from eval.pareto import ParetoPoint, extract_pareto_points, pareto_frontier, save_pareto_csv
from ferret.constants import MAX_BUDGET, NUM_PROBE_ACTIONS
from graph import FerretDetector
from policy.trunk import FerretAgent, PolicyConfig
from policy.vision_encoder import VisionEncoder
from reward.morl_reward import MORLReward, sample_preference_vector
from train.ppo_train import _target_model_for_foolbox


BALANCED_PREFERENCE = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------

def _run_agent_episodes(
    detector: FerretDetector,
    pipeline: FerretDataPipeline,
    episodes: int,
    preference: np.ndarray | None = None,
    split: str = "val",
) -> list[EpisodeRecord]:
    records = []
    for _ in tqdm(range(episodes), leave=False):
        sample = pipeline.sample_episode(split)  # type: ignore[arg-type]
        outcome = detector.detect(
            sample.image,
            label=sample.label,
            preference=preference,
        )
        records.append(
            EpisodeRecord(
                confidence=outcome.confidence,
                flagged=outcome.flagged,
                is_adversarial=bool(sample.is_adversarial),
                probes_used=outcome.probes_used,
                attack_type=sample.attack_type,
            )
        )
    return records


def _run_single_probe(
    encoder: VisionEncoder,
    pipeline: FerretDataPipeline,
    episodes: int,
    seed: int = 0,
    split: str = "val",
) -> list[EpisodeRecord]:
    """Single-probe baseline: one random action, then decide based on confidence."""
    from agents.confidence import ConfidenceAggregator
    from agents.decision import DecisionNode, DecisionOutcome
    from agents.probe_executor import ProbeExecutor

    rng = np.random.default_rng(seed)
    executor = ProbeExecutor(encoder)
    confidence_agg = ConfidenceAggregator()
    decision_node = DecisionNode()
    records = []

    for _ in tqdm(range(episodes), leave=False):
        sample = pipeline.sample_episode(split)  # type: ignore[arg-type]
        _, features = encoder(sample.image)
        baseline_logits = encoder.logits(sample.image).squeeze(0).detach().cpu()
        confidence_agg.reset(baseline_logits)

        action = int(rng.integers(0, NUM_PROBE_ACTIONS))
        probe = executor.execute(sample.image, action, label=sample.label)
        confidence_after = confidence_agg.update(probe.logits, step_index=0)

        # Force decision after exactly 1 probe.
        decision = decision_node.evaluate(confidence_after, remaining_budget=0)
        flagged = decision == DecisionOutcome.FLAG

        records.append(
            EpisodeRecord(
                confidence=confidence_after,
                flagged=flagged,
                is_adversarial=bool(sample.is_adversarial),
                probes_used=1,
                attack_type=sample.attack_type,
            )
        )
    return records


def _make_pipeline(
    attack_types: tuple[str, ...],
    encoder: VisionEncoder,
    device: torch.device,
    num_workers: int = 0,
) -> FerretDataPipeline:
    cfg = DataConfig(
        adversarial_ratio=0.5,
        attack_types=attack_types,
        precompute_adversarial=True,
        download=False,
        num_workers=num_workers,
    )
    return FerretDataPipeline(cfg, model=_target_model_for_foolbox(encoder), device=device)


# ---------------------------------------------------------------------------
# Individual ablations
# ---------------------------------------------------------------------------

def ablation_a1_morl_vs_fixed(
    checkpoint: str,
    episodes: int,
    pipeline: FerretDataPipeline,
    device: torch.device,
    split: str = "val",
) -> dict[str, DetectorMetrics]:
    """A1: MORL (random Dirichlet preference) vs fixed balanced preference."""
    results = {}

    # Fixed balanced preference (single-objective proxy)
    det_fixed = FerretDetector.from_checkpoint(checkpoint, device=device)
    records_fixed = _run_agent_episodes(det_fixed, pipeline, episodes, preference=BALANCED_PREFERENCE, split=split)
    results["A1_fixed_weights"] = compute_metrics("A1_fixed_weights", records_fixed)

    # MORL: random preference per episode (already handled in detect() when preference=None)
    det_morl = FerretDetector.from_checkpoint(checkpoint, device=device)
    records_morl = _run_agent_episodes(det_morl, pipeline, episodes, preference=None, split=split)
    results["A1_morl"] = compute_metrics("A1_morl", records_morl)

    return results


def ablation_a2_transformer_vs_mlp(
    checkpoint: str,
    episodes: int,
    pipeline: FerretDataPipeline,
    device: torch.device,
    split: str = "val",
) -> dict[str, DetectorMetrics]:
    """
    A2: Transformer trunk vs MLP trunk.

    Loads the trained transformer policy and compares against a fresh MLP-trunk
    policy with random weights (shows what the transformer actually learned vs
    a naive flat aggregator). For a fair comparison, re-train the MLP variant —
    but for quick inspection, random MLP shows the floor.
    """
    results = {}

    # Transformer (trained)
    det_tf = FerretDetector.from_checkpoint(checkpoint, device=device)
    records_tf = _run_agent_episodes(det_tf, pipeline, episodes, split=split)
    results["A2_transformer"] = compute_metrics("A2_transformer", records_tf)

    # MLP trunk (untrained — random-weights floor)
    mlp_config = PolicyConfig(use_mlp_trunk=True)
    mlp_agent = FerretAgent(mlp_config).to(device)
    encoder = VisionEncoder(device=device)
    det_mlp = FerretDetector.from_models(encoder, mlp_agent, deterministic_policy=False)
    records_mlp = _run_agent_episodes(det_mlp, pipeline, episodes, split=split)
    results["A2_mlp_random"] = compute_metrics("A2_mlp_random", records_mlp)

    return results


def ablation_a3_sequential_vs_single(
    checkpoint: str,
    episodes: int,
    encoder: VisionEncoder,
    pipeline: FerretDataPipeline,
    device: torch.device,
    split: str = "val",
    seed: int = 0,
) -> dict[str, DetectorMetrics]:
    """A3: Sequential learned probing vs hardcoded single-probe baseline."""
    results = {}

    det = FerretDetector.from_checkpoint(checkpoint, device=device)
    records_seq = _run_agent_episodes(det, pipeline, episodes, split=split)
    results["A3_sequential"] = compute_metrics("A3_sequential", records_seq)

    records_single = _run_single_probe(encoder, pipeline, episodes, seed=seed, split=split)
    results["A3_single_probe"] = compute_metrics("A3_single_probe", records_single)

    return results


def ablation_a4_budget_sizes(
    checkpoint: str,
    episodes: int,
    pipeline: FerretDataPipeline,
    device: torch.device,
    budgets: tuple[int, ...] = (5, 10, 20),
    split: str = "val",
) -> dict[str, DetectorMetrics]:
    """
    A4: Budget sizes 5 / 10 / 20.

    The trained policy is allowed fewer or more probes by modifying the
    decision_node threshold rather than re-training (lower budget → fewer
    steps before forced decision). For a rigorous study, re-train each.
    For quick inspection, this shows sensitivity to budget.
    """
    from agents.decision import DecisionNode

    results = {}
    for budget in budgets:
        # Force early termination by adjusting the decision threshold to be
        # lower (fires sooner) or setting max_budget on the deps object.
        det = FerretDetector.from_checkpoint(checkpoint, device=device)
        det.deps.max_budget = budget

        records = _run_agent_episodes(det, pipeline, episodes, split=split)
        name = f"A4_budget_{budget}"
        results[name] = compute_metrics(name, records)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ABLATION_REGISTRY = {
    "A1": ablation_a1_morl_vs_fixed,
    "A2": ablation_a2_transformer_vs_mlp,
    "A3": ablation_a3_sequential_vs_single,
    "A4": ablation_a4_budget_sizes,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ferret ablation study (spec §10.4)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy.pt")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--ablations", nargs="+", default=list(ABLATION_REGISTRY), choices=list(ABLATION_REGISTRY))
    parser.add_argument("--attack-types", nargs="+", default=["fgsm"])
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    tag = args.tag or f"ablation_{int(time.time())}"
    out_dir = Path("eval/results") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VisionEncoder(device=device)
    pipeline = _make_pipeline(tuple(args.attack_types), encoder, device)
    pipeline.ensure_adversarial_cache(args.split)

    all_metrics: dict[str, DetectorMetrics] = {}

    for ablation_id in args.ablations:
        print(f"\n{'='*60}")
        print(f"  Running ablation {ablation_id}")
        print(f"{'='*60}")

        if ablation_id == "A1":
            all_metrics.update(
                ablation_a1_morl_vs_fixed(args.checkpoint, args.episodes, pipeline, device, args.split)
            )
        elif ablation_id == "A2":
            all_metrics.update(
                ablation_a2_transformer_vs_mlp(args.checkpoint, args.episodes, pipeline, device, args.split)
            )
        elif ablation_id == "A3":
            all_metrics.update(
                ablation_a3_sequential_vs_single(args.checkpoint, args.episodes, encoder, pipeline, device, args.split, args.seed)
            )
        elif ablation_id == "A4":
            all_metrics.update(
                ablation_a4_budget_sizes(args.checkpoint, args.episodes, pipeline, device, split=args.split)
            )

    for m in all_metrics.values():
        print_metrics(m)

    # Save JSON
    payload = {
        "tag": tag,
        "checkpoint": args.checkpoint,
        "ablations": args.ablations,
        "episodes": args.episodes,
        "split": args.split,
        "results": {k: dataclasses.asdict(v) for k, v in all_metrics.items()},
    }
    json_path = out_dir / "ablations.json"
    json_path.write_text(json.dumps(payload, indent=2))

    # Pareto CSV + plot across ablation variants
    pareto_pts = extract_pareto_points(list(all_metrics.values()))
    save_pareto_csv(pareto_pts, out_dir / "ablations_pareto.csv")

    if args.plot:
        from eval.pareto import plot_pareto
        try:
            frontier = pareto_frontier(pareto_pts)
            plot_pareto(pareto_pts, frontier, output_path=out_dir / "ablations_pareto.png", title="Ferret Ablations — Pareto")
            print(f"Pareto plot → {out_dir / 'ablations_pareto.png'}")
        except ImportError as exc:
            print(f"Skipping plot: {exc}")

    print(f"\nAblation results → {out_dir}/")


if __name__ == "__main__":
    main()
