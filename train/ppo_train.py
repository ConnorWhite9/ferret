"""CleanRL PPO with preference-conditioned MORL scalarization."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter

from data import DataConfig, FerretDataPipeline
from env.vision_env import make_vision_env
from ferret.constants import MAX_BUDGET, NUM_PROBE_ACTIONS
from policy.trunk import FerretAgent, PolicyConfig
from policy.vision_encoder import VisionEncoder
from reward.morl_reward import LambdaSchedule, MORLReward, RewardNormalizer
from train.checkpoint import load_checkpoint, save_checkpoint
from train.morl_logging import extract_episode_metrics, extract_reward_vectors, log_morl_metrics
from train.morl_scalarization import PreferenceConditionedScalarization


@dataclass
class Args:
    exp_name: str = "ferret_ppo"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ferret"
    wandb_entity: str | None = None

    total_timesteps: int = 500_000
    learning_rate: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    adversarial_ratio: float = 0.5
    attack_types: tuple[str, ...] = ("fgsm", "pgd", "cw")
    precompute_adversarial: bool = True
    dataset: str = "imagenette"
    download_data: bool = True
    max_budget: int = MAX_BUDGET
    batch_size: int = 32

    lambda_anneal: bool = True
    lambda_start: float = 0.01
    lambda_end: float = 0.05

    # Resume / checkpointing
    resume: str | None = None
    checkpoint_every: int = 50  # save every N updates

    # Periodic val eval during training
    eval_every: int = 0   # 0 = disabled; >0 = eval every N updates
    eval_episodes: int = 50


def _alloc_obs_buffer(
    obs_space: spaces.Dict,
    num_steps: int,
    num_envs: int,
) -> dict[str, np.ndarray]:
    buffers: dict[str, np.ndarray] = {}
    for key, space in obs_space.spaces.items():
        buffers[key] = np.zeros(
            (num_steps, num_envs, *space.shape),
            dtype=space.dtype if hasattr(space, "dtype") else np.float32,
        )
    return buffers


def _slice_obs(obs_buffer: dict[str, np.ndarray], step: int) -> dict[str, np.ndarray]:
    return {key: value[step] for key, value in obs_buffer.items()}


def _obs_to_torch(
    obs: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor | np.ndarray]:
    # FerretAgent converts internally; keep numpy for compatibility.
    return obs


def train(args: Args) -> None:
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = args.batch_size // args.num_minibatches
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    data_config = DataConfig(
        dataset=args.dataset,  # type: ignore[arg-type]
        adversarial_ratio=args.adversarial_ratio,
        attack_types=args.attack_types,  # type: ignore[arg-type]
        precompute_adversarial=args.precompute_adversarial,
        download=args.download_data,
        # macOS/fork safety: DataLoader workers must be 0 when spawned from SyncVectorEnv
        num_workers=0,
    )
    target_model = VisionEncoder(device=device)
    pipeline = FerretDataPipeline(
        data_config,
        model=_target_model_for_foolbox(target_model),
        device=device,
    )
    if data_config.adversarial_ratio > 0.0 and data_config.precompute_adversarial:
        print("Precomputing adversarial cache (may take a while on first run)...")
        pipeline.ensure_adversarial_cache("train")
        if args.eval_every > 0:
            pipeline.ensure_adversarial_cache("val")

    morl = PreferenceConditionedScalarization(normalizer=RewardNormalizer())
    shared_reward = MORLReward(max_budget=args.max_budget, lambda_eff=args.lambda_start)
    lambda_schedule = LambdaSchedule(start=args.lambda_start, end=args.lambda_end)

    def make_env(rank: int):
        def _init():
            return make_vision_env(
                target_model=target_model,
                data_pipeline=pipeline,
                split="train",
                seed=args.seed + rank,
                reward_fn=shared_reward,
                reward_normalizer=morl.normalizer,
                max_budget=args.max_budget,
            )

        return _init

    envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    assert envs.single_action_space.n == NUM_PROBE_ACTIONS

    agent = FerretAgent(PolicyConfig(max_budget=args.max_budget)).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    start_global_step = 0
    start_update = 1
    if args.resume:
        resume_path = Path(args.resume)
        print(f"Resuming from {resume_path}")
        start_global_step, resume_update, _ = load_checkpoint(
            resume_path, agent, optimizer, device=device
        )
        start_update = resume_update + 1

    obs_buffer = _alloc_obs_buffer(envs.single_observation_space, args.num_steps, args.num_envs)
    actions = np.zeros((args.num_steps, args.num_envs), dtype=np.int64)
    logprobs = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    rewards = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    dones = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    values = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    reward_vectors = np.zeros((args.num_steps, args.num_envs, 4), dtype=np.float32)

    global_step = start_global_step
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = np.zeros(args.num_envs, dtype=np.float32)

    checkpoint_dir = Path(f"runs/{run_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    num_updates = args.total_timesteps // args.batch_size
    episode_metrics_accum: dict[str, list[float]] = {}

    for update in range(start_update, num_updates + 1):
        progress = (update - 1.0) / max(num_updates - 1, 1)
        if args.anneal_lr:
            frac = 1.0 - progress
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate
        if args.lambda_anneal:
            current_lambda = lambda_schedule.apply(shared_reward, progress)
            writer.add_scalar("morl/lambda_eff", current_lambda, global_step)

        for step in range(args.num_steps):
            global_step += args.num_envs
            for key in obs_buffer:
                obs_buffer[key][step] = next_obs[key]

            with torch.no_grad():
                obs_tensor = _obs_to_torch(next_obs, device)
                action, logprob, _, value = agent.get_action_and_value(obs_tensor)
                values[step] = value.cpu().numpy()
                actions[step] = action.cpu().numpy()
                logprobs[step] = logprob.cpu().numpy()

            next_obs, reward, terminations, truncations, infos = envs.step(actions[step])
            done = np.logical_or(terminations, truncations)
            rewards[step] = reward
            dones[step] = done.astype(np.float32)

            reward_vectors[step] = extract_reward_vectors(infos, args.num_envs)
            step_metrics = extract_episode_metrics(infos, args.num_envs)
            # NOTE: loop var intentionally named ep_vals to avoid shadowing the
            # critic `values` buffer defined above.
            for key, ep_vals in step_metrics.items():
                finite = ep_vals[np.isfinite(ep_vals)]
                if finite.size > 0:
                    episode_metrics_accum.setdefault(key, []).extend(finite.tolist())

            next_done = done.astype(np.float32)

        with torch.no_grad():
            obs_tensor = _obs_to_torch(next_obs, device)
            next_value = agent.get_action_and_value(obs_tensor)[3].cpu().numpy()
            advantages = np.zeros_like(rewards)
            last_gae = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
                advantages[t] = last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
            returns = advantages + values

        flat_obs = {
            key: arr.reshape((-1, *arr.shape[2:]))
            for key, arr in obs_buffer.items()
        }
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = args.batch_size
        minibatch_size = args.minibatch_size
        indices = np.arange(batch_size)

        clipfracs = []
        for _epoch in range(args.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = indices[start:end]
                mb_obs = {key: value[mb_inds] for key, value in flat_obs.items()}

                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    mb_obs,
                    action=torch.as_tensor(b_actions[mb_inds], device=device),
                )
                logratio = new_logprob - torch.as_tensor(b_logprobs[mb_inds], device=device)
                ratio = logratio.exp()

                mb_adv = torch.as_tensor(b_advantages[mb_inds], device=device)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_value = new_value.view(-1)
                mb_returns = torch.as_tensor(b_returns[mb_inds], device=device)
                mb_values = torch.as_tensor(b_values[mb_inds], device=device)
                if args.clip_vloss:
                    v_loss_unclipped = (new_value - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        new_value - mb_values,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values, b_returns
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        log_morl_metrics(writer, reward_vectors, global_step)

        if episode_metrics_accum:
            log_morl_metrics(
                writer,
                reward_vectors,
                global_step,
                episode_metrics={
                    key: np.array(vals, dtype=np.float32) for key, vals in episode_metrics_accum.items()
                },
            )
            episode_metrics_accum.clear()

        print(
            f"update={update}/{num_updates} global_step={global_step} "
            f"lambda={shared_reward.lambda_eff:.4f} "
            f"SPS={int(global_step / (time.time() - start_time))}"
        )

        # Periodic checkpoint
        if args.checkpoint_every > 0 and update % args.checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"policy_step{global_step}.pt"
            save_checkpoint(ckpt_path, agent, optimizer, args, global_step, update)

        # Periodic val eval
        if args.eval_every > 0 and update % args.eval_every == 0:
            _run_val_eval(agent, target_model, pipeline, args, device, writer, global_step)

    envs.close()
    writer.close()

    save_checkpoint(checkpoint_dir / "policy.pt", agent, optimizer, args, global_step, update)
    print(f"Saved final checkpoint to {checkpoint_dir / 'policy.pt'}")

    if args.track:
        wandb.finish()


def _run_val_eval(
    agent: FerretAgent,
    encoder: VisionEncoder,
    pipeline: FerretDataPipeline,
    args: Args,
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
) -> None:
    from eval.metrics import EpisodeRecord, compute_metrics
    from graph import FerretDetector

    agent.eval()
    detector = FerretDetector.from_models(encoder, agent, deterministic_policy=True)
    records: list[EpisodeRecord] = []
    for _ in range(args.eval_episodes):
        try:
            sample = pipeline.sample_episode("val")  # type: ignore[arg-type]
            outcome = detector.detect(sample.image, label=sample.label)
            records.append(
                EpisodeRecord(
                    confidence=outcome.confidence,
                    flagged=outcome.flagged,
                    is_adversarial=bool(sample.is_adversarial),
                    probes_used=outcome.probes_used,
                    attack_type=sample.attack_type,
                )
            )
        except Exception:
            continue
    agent.train()

    if not records:
        return
    m = compute_metrics("val", records)
    writer.add_scalar("eval/accuracy", m.accuracy, global_step)
    writer.add_scalar("eval/roc_auc", m.roc_auc, global_step)
    writer.add_scalar("eval/fpr_at_tpr95", m.fpr_at_tpr95, global_step)
    writer.add_scalar("eval/mean_probes", m.mean_probes, global_step)
    print(
        f"[val] accuracy={m.accuracy:.3f} roc_auc={m.roc_auc:.3f} "
        f"fpr95={m.fpr_at_tpr95:.3f} probes={m.mean_probes:.2f}"
    )


def _target_model_for_foolbox(encoder: VisionEncoder) -> torch.nn.Module:
    """Expose a single nn.Module for foolbox adversarial caching."""
    import torch.nn as nn

    class _Wrapper(nn.Module):
        def __init__(self, vision_encoder: VisionEncoder):
            super().__init__()
            self.encoder = vision_encoder

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Must use logits_with_grad so foolbox can backpropagate for FGSM/PGD.
            return self.encoder.logits_with_grad(x)

    return _Wrapper(encoder)


if __name__ == "__main__":
    train(tyro.cli(Args))
