"""
Unified Ferret evaluation benchmark.

Runs all detectors through the same episode loop:
  - Random policy (lower bound)
  - Learned Ferret policy  (from checkpoint)
  - Feature Squeezing      (single-pass baseline)
  - Mahalanobis Distance   (single-pass baseline)

Outputs per eval run:
  - eval/results/<tag>/metrics.json   — full metric report
  - eval/results/<tag>/pareto.csv     — Pareto points
  - eval/results/<tag>/pareto.png     — Pareto plot
  - eval/results/<tag>/episodes.jsonl — per-episode records

Usage:
  python -m eval.benchmark --checkpoint runs/.../policy.pt --episodes 200
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
from data.datasets import EpisodeSample
from env.vision_env import FerretVisionEnv
from eval.baselines import FeatureSqueezing, MahalanobisDetector, fit_baselines
from eval.metrics import DetectorMetrics, EpisodeRecord, compute_metrics, print_metrics
from eval.pareto import extract_pareto_points, pareto_frontier, plot_pareto, save_pareto_csv
from ferret.constants import NUM_PROBE_ACTIONS
from graph import FerretDetector
from policy.vision_encoder import VisionEncoder
from train.ppo_train import _target_model_for_foolbox


# ---------------------------------------------------------------------------
# Per-detector runners
# ---------------------------------------------------------------------------

def _run_random(
    env: FerretVisionEnv,
    episodes: int,
    seed: int = 0,
) -> list[EpisodeRecord]:
    rng = np.random.default_rng(seed)
    records: list[EpisodeRecord] = []
    for _ in tqdm(range(episodes), desc="random"):
        env.reset()
        done = False
        info: dict = {}
        while not done:
            action = int(rng.integers(0, NUM_PROBE_ACTIONS))
            _, _, term, trunc, info = env.step(action)
            done = term or trunc
        ep = env._episode  # noqa: SLF001
        records.append(
            EpisodeRecord(
                confidence=float(info.get("confidence", 0.0)),
                flagged=info.get("decision") == "flag",
                is_adversarial=bool(ep.is_adversarial),
                probes_used=int(info.get("probes_used", 0)),
                attack_type=ep.attack_type,
            )
        )
    return records


def _run_ferret(
    detector: FerretDetector,
    pipeline: FerretDataPipeline,
    episodes: int,
    split: str = "val",
    preference: np.ndarray | None = None,
) -> list[EpisodeRecord]:
    if preference is None:
        preference = np.array([0.35, 0.35, 0.15, 0.15], dtype=np.float32)
    records: list[EpisodeRecord] = []
    for _ in tqdm(range(episodes), desc="ferret"):
        sample: EpisodeSample = pipeline.sample_episode(split)  # type: ignore[arg-type]
        outcome = detector.detect(sample.image, label=sample.label, preference=preference)
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


def _run_baseline(
    name: str,
    pipeline: FerretDataPipeline,
    baseline: FeatureSqueezing | MahalanobisDetector,
    episodes: int,
    split: str = "val",
) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for _ in tqdm(range(episodes), desc=name):
        sample: EpisodeSample = pipeline.sample_episode(split)  # type: ignore[arg-type]
        record = baseline.detect(
            sample.image,
            is_adversarial=bool(sample.is_adversarial),
            attack_type=sample.attack_type,
        )
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_metrics_json(
    results: dict[str, DetectorMetrics],
    path: Path,
    meta: dict | None = None,
) -> None:
    payload = {
        "meta": meta or {},
        "results": {
            name: dataclasses.asdict(m) for name, m in results.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _save_episodes_jsonl(
    records_by_detector: dict[str, list[EpisodeRecord]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for det, records in records_by_detector.items():
            for r in records:
                row = dataclasses.asdict(r)
                row["detector"] = det
                f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ferret unified evaluation benchmark")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to policy.pt")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--fit-episodes", type=int, default=50, help="Clean episodes for baseline fitting")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--adversarial-ratio", type=float, default=0.5)
    parser.add_argument("--attack-types", nargs="+", default=["fgsm"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None, help="Run tag for output dir (default: timestamp)")
    parser.add_argument("--no-baselines", action="store_true", help="Skip Feature Squeezing + Mahalanobis")
    parser.add_argument("--plot", action="store_true", help="Generate Pareto PNG")
    args = parser.parse_args()

    tag = args.tag or f"eval_{int(time.time())}"
    out_dir = Path("eval/results") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = DataConfig(
        adversarial_ratio=args.adversarial_ratio,
        attack_types=tuple(args.attack_types),  # type: ignore[arg-type]
        precompute_adversarial=True,
        download=False,
        num_workers=0,
    )
    encoder = VisionEncoder(device=device)
    pipeline = FerretDataPipeline(
        data_config,
        model=_target_model_for_foolbox(encoder),
        device=device,
    )
    pipeline.ensure_adversarial_cache(args.split)

    all_records: dict[str, list[EpisodeRecord]] = {}

    # -- Random policy --
    env = FerretVisionEnv(encoder, pipeline, split=args.split, seed=args.seed)
    all_records["random"] = _run_random(env, args.episodes, seed=args.seed)

    # -- Learned Ferret policy --
    if args.checkpoint:
        detector = FerretDetector.from_checkpoint(args.checkpoint, device=device)
        all_records["ferret"] = _run_ferret(detector, pipeline, args.episodes, split=args.split)

    # -- Single-pass baselines --
    if not args.no_baselines:
        print(f"Fitting baselines on {args.fit_episodes} clean episodes…")
        fit_cfg = DataConfig(adversarial_ratio=0.0, download=False, num_workers=0)
        fit_pipeline = FerretDataPipeline(fit_cfg, model=_target_model_for_foolbox(encoder), device=device)
        fit_samples = [fit_pipeline.sample_episode(args.split) for _ in range(args.fit_episodes)]
        fit_images = [s.image for s in fit_samples]
        fit_labels = [s.label for s in fit_samples]

        fs, maha = fit_baselines(encoder, fit_images, fit_labels)
        all_records["feature_squeezing"] = _run_baseline("feature_squeezing", pipeline, fs, args.episodes, split=args.split)
        all_records["mahalanobis"] = _run_baseline("mahalanobis", pipeline, maha, args.episodes, split=args.split)

    # -- Compute metrics --
    metrics: dict[str, DetectorMetrics] = {
        name: compute_metrics(name, records) for name, records in all_records.items()
    }

    for m in metrics.values():
        print_metrics(m)

    # -- Save outputs --
    meta = {
        "tag": tag,
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "split": args.split,
        "adversarial_ratio": args.adversarial_ratio,
        "attack_types": args.attack_types,
        "seed": args.seed,
    }
    _save_metrics_json(metrics, out_dir / "metrics.json", meta=meta)
    _save_episodes_jsonl(all_records, out_dir / "episodes.jsonl")

    pareto_pts = extract_pareto_points(list(metrics.values()))
    frontier = pareto_frontier(pareto_pts)
    save_pareto_csv(pareto_pts, out_dir / "pareto.csv")

    if args.plot:
        try:
            plot_pareto(pareto_pts, frontier, output_path=out_dir / "pareto.png")
            print(f"Pareto plot → {out_dir / 'pareto.png'}")
        except ImportError as exc:
            print(f"Skipping plot: {exc}")

    print(f"\nResults → {out_dir}/")


if __name__ == "__main__":
    main()
