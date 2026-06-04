"""
Evaluation utilities: random vs learned policy, Pareto-style metrics.

Phase 3 will add Feature Squeezing and Mahalanobis baselines.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from data import DataConfig, FerretDataPipeline
from env.vision_env import FerretVisionEnv
from ferret.constants import MORL_OBJECTIVE_NAMES, NUM_PROBE_ACTIONS
from graph import FerretDetector
from policy.vision_encoder import VisionEncoder
from train.ppo_train import _target_model_for_foolbox


@dataclass
class EpisodeMetrics:
    flagged: bool
    correct: bool
    probes_used: int
    confidence: float
    is_adversarial: bool
    attack_type: str | None
    reward_vector: list[float]


@dataclass
class BenchmarkSummary:
    policy: str
    accuracy: float
    mean_probes: float
    mean_confidence: float
    fpr: float
    tpr: float
    pareto_points: list[dict[str, float]]


def run_random_policy(env: FerretVisionEnv, episodes: int, seed: int = 0) -> list[EpisodeMetrics]:
    rng = np.random.default_rng(seed)
    results: list[EpisodeMetrics] = []

    for _ in range(episodes):
        env.reset()
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            action = int(rng.integers(0, NUM_PROBE_ACTIONS))
            _, _, terminated, truncated, info = env.step(action)

        episode = env._episode  # noqa: SLF001
        flagged = info.get("decision") == "flag"
        is_adv = bool(episode.is_adversarial)
        correct = flagged == is_adv
        results.append(
            EpisodeMetrics(
                flagged=flagged,
                correct=correct,
                probes_used=int(info.get("probes_used", 0)),
                confidence=float(info.get("confidence", 0.0)),
                is_adversarial=is_adv,
                attack_type=episode.attack_type,
                reward_vector=list(info.get("reward_vector", np.zeros(4))),
            )
        )
    return results


def run_learned_policy(
    detector: FerretDetector,
    pipeline: FerretDataPipeline,
    episodes: int,
    split: str = "val",
) -> list[EpisodeMetrics]:
    results: list[EpisodeMetrics] = []
    for _ in range(episodes):
        sample = pipeline.sample_episode(split)  # type: ignore[arg-type]
        outcome = detector.detect(
            sample.image,
            label=sample.label,
            preference=sample_preference_default(),
        )
        flagged = outcome.flagged
        is_adv = bool(sample.is_adversarial)
        results.append(
            EpisodeMetrics(
                flagged=flagged,
                correct=flagged == is_adv,
                probes_used=outcome.probes_used,
                confidence=outcome.confidence,
                is_adversarial=is_adv,
                attack_type=sample.attack_type,
                reward_vector=[0.0, 0.0, 0.0, 0.0],
            )
        )
    return results


def sample_preference_default() -> np.ndarray:
    """Balanced deployment preference for eval."""
    return np.array([0.35, 0.35, 0.15, 0.15], dtype=np.float32)


def summarize(policy_name: str, episodes: list[EpisodeMetrics]) -> BenchmarkSummary:
    if not episodes:
        raise ValueError("No episodes to summarize.")

    correct = [e.correct for e in episodes]
    adv = [e for e in episodes if e.is_adversarial]
    clean = [e for e in episodes if not e.is_adversarial]

    tp = sum(1 for e in adv if e.flagged)
    fn = sum(1 for e in adv if not e.flagged)
    fp = sum(1 for e in clean if e.flagged)
    tn = sum(1 for e in clean if not e.flagged)

    tpr = tp / max(len(adv), 1)
    fpr = fp / max(len(clean), 1)

    pareto_points = [
        {
            "accuracy": float(np.mean(correct)),
            "mean_probes": float(np.mean([e.probes_used for e in episodes])),
            "confidence": float(np.mean([e.confidence for e in episodes])),
        }
    ]

    return BenchmarkSummary(
        policy=policy_name,
        accuracy=float(np.mean(correct)),
        mean_probes=float(np.mean([e.probes_used for e in episodes])),
        mean_confidence=float(np.mean([e.confidence for e in episodes])),
        fpr=fpr,
        tpr=tpr,
        pareto_points=pareto_points,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ferret policy benchmark")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--adversarial-ratio", type=float, default=0.5)
    parser.add_argument("--attack-types", nargs="+", default=["fgsm"])
    parser.add_argument("--output", type=str, default="eval/results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = DataConfig(
        adversarial_ratio=args.adversarial_ratio,
        attack_types=tuple(args.attack_types),  # type: ignore[arg-type]
        precompute_adversarial=True,
        download=False,
    )
    encoder = VisionEncoder(device=device)
    pipeline = FerretDataPipeline(data_config, model=_target_model_for_foolbox(encoder), device=device)
    pipeline.ensure_adversarial_cache("val")

    env = FerretVisionEnv(encoder, pipeline, split="val", seed=0)
    summaries = [summarize("random", run_random_policy(env, args.episodes))]

    if args.checkpoint:
        detector = FerretDetector.from_checkpoint(args.checkpoint, device=device)
        learned = run_learned_policy(detector, pipeline, args.episodes)
        summaries.append(summarize("learned", learned))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "objectives": list(MORL_OBJECTIVE_NAMES),
        "summaries": [asdict(s) for s in summaries],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output_path}")
    for summary in summaries:
        print(
            f"[{summary.policy}] accuracy={summary.accuracy:.3f} "
            f"mean_probes={summary.mean_probes:.2f} tpr={summary.tpr:.3f} fpr={summary.fpr:.3f}"
        )


if __name__ == "__main__":
    main()
