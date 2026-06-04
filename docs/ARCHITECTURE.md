# Ferret System Architecture

Sequential adversarial probing agent: an RL-trained policy chooses where and how to perturb inputs under a fixed query budget, while a rule-based confidence aggregator makes detection decisions.

Technical spec: [`spec.md`](../spec.md)

---

## High-level overview

Ferret has two runtimes that share the same probe / confidence / decision logic:

| Runtime | Entry point | Purpose |
|---------|-------------|---------|
| **Training** | `train/ppo_train.py` + `env/vision_env.py` | Learn probing policy with CleanRL PPO + MORL scalarization |
| **Inference** | `graph/langgraph_agent.py` + `FerretDetector` | Deploy frozen policy as an explicit agent graph |

```mermaid
flowchart TB
    subgraph Data["Data layer"]
        DL[ImageNette / ImageNet]
        AC[Adversarial cache<br/>FGSM · PGD · CW]
        DL --> AC
    end

    subgraph Train["Training runtime"]
        ENV[FerretVisionEnv]
        PPO[CleanRL PPO]
        POL[FerretPolicy transformer]
        Data --> ENV
        ENV --> PPO
        PPO --> POL
    end

    subgraph Infer["Inference runtime"]
        LG[LangGraph StateGraph]
        DET[FerretDetector API]
        LG --> DET
    end

    subgraph Shared["Shared components"]
        PE[ProbeExecutor]
        TM[VisionEncoder ResNet-50]
        CA[ConfidenceAggregator]
        DN[DecisionNode]
        PE --> TM
        CA --> DN
    end

    ENV --> Shared
    LG --> Shared
    POL -.->|checkpoint| DET
```

---

## LangGraph agent (inference)

Each graph lap is one probe macro-step: policy → probe → confidence → decision → loop or end.

```mermaid
stateDiagram-v2
    [*] --> input: raw image + preference
    input --> router: modality
    router --> policy: build obs
    policy --> probe: action 0–587
    probe --> confidence: logit shift
    confidence --> decision: threshold

    decision --> policy: continue
    decision --> [*]: flag adversarial
    decision --> [*]: abstain
```

### Node map

| Graph node | Module | Responsibility |
|------------|--------|----------------|
| `input` | `graph/nodes/input.py` | Init `EpisodeState`, baseline logits, preference vector |
| `router` | `graph/nodes/router.py` | Vision vs language (vision only today) |
| `policy` | `graph/nodes/policy.py` | Frozen `FerretAgent` → discrete probe action |
| `probe` | `graph/nodes/probe.py` | `ProbeExecutor` + target model logits |
| `confidence` | `graph/nodes/confidence.py` | Rule-based L2 logit-shift score |
| `decision` | `graph/nodes/decision.py` | Flag / continue / abstain |

### Inference usage

```python
from graph import FerretDetector

detector = FerretDetector.from_checkpoint("runs/<run>/policy.pt")
result = detector.detect(image_tensor, preference=np.array([0.4, 0.3, 0.2, 0.1]))
# result.flagged, result.confidence, result.probes_used
```

---

## RL training loop

```mermaid
sequenceDiagram
    participant PPO as PPO trainer
    participant Env as FerretVisionEnv
    participant Pol as FerretPolicy
    participant MORL as MORL scalarization

    PPO->>Env: reset()
    Env-->>PPO: obs + preference ~ Dirichlet

    loop each probe step
        PPO->>Pol: get_action_and_value(obs)
        Pol-->>PPO: action
        PPO->>Env: step(action)
        Env-->>PPO: obs, reward_vector, done
        Note over MORL: scalar = dot(preference, norm(reward_vector))
    end
```

### Environment step (inlined graph)

`FerretVisionEnv.step()` runs the same chain as the LangGraph loop in one call for vectorized PPO:

```mermaid
flowchart LR
    A[action] --> B[ProbeExecutor]
    B --> C[VisionEncoder logits]
    C --> D[ConfidenceAggregator]
    D --> E{DecisionNode}
    E -->|continue| F[next obs]
    E -->|flag / abstain| G[terminal MORL reward]
```

---

## Policy network

```mermaid
flowchart LR
    subgraph History["Probe history sequence"]
        PG[grid embed]
        PP[pert embed]
        PM[mag embed]
        RL[logit proj 1000→d]
        PG --> TOK[+]
        PP --> TOK
        PM --> TOK
        RL --> TOK
    end

    TOK --> TR[3-layer Transformer]
    TR --> POOL[last token pool]

    subgraph Static["Static context"]
        IE[input embedding 2048]
        BUD[remaining budget]
        PREF[preference 4]
        CONF[confidence]
    end

    IE --> SP[static MLP]
    BUD --> SP
    PREF --> SP
    CONF --> SP

    POOL --> CAT[concat]
    SP --> CAT
    CAT --> ACTOR[MLP → 588 actions]
    CAT --> CRITIC[MLP → value]
```

| Parameter | Value |
|-----------|-------|
| Action space | 49 grid × 4 perturbation × 3 magnitude = **588** |
| Max probes per episode | **10** |
| Input embedding | Frozen ResNet-50 pooled features (2048-d) |

---

## MORL reward (4 objectives)

Preference vector **w** is sampled per episode from Dirichlet(1,…,1) and concatenated to the policy input. Scalar PPO reward = **w · normalize(reward_vector)**.

```mermaid
flowchart TB
    subgraph Step["Step rewards"]
        IG[info gain Δconfidence]
        QP[query penalty −λ / budget]
        IG --> EFF[efficiency objective]
        QP --> EFF
    end

    subgraph Terminal["Terminal rewards"]
        ACC[accuracy ±1 / −0.5 + calibration]
        EB[early decision bonus]
        FP[false positive −β]
        GEN[generalization +γ rare attacks]
    end

    Step --> RV[reward_vector 4-d]
    Terminal --> RV
    RV --> NORM[running mean/std]
    NORM --> DOT[dot with preference w]
    DOT --> PPO[PPO advantage]
```

| Index | Objective | Training weight in **w** |
|-------|-----------|---------------------------|
| 0 | Detection accuracy | w₁ |
| 1 | Query efficiency | w₂ |
| 2 | False positive rate | w₃ |
| 3 | Attack generalization | w₄ |

**λ annealing** (spec §4.4): `LambdaSchedule` ramps λ from `0.01` → `0.05` over training so early exploration uses the full budget.

Implementation note: we use **preference-conditioned linear scalarization** (`train/morl_scalarization.py`), which matches the MORL-Baselines pattern for conditioned policies without a second training stack.

---

## Data pipeline

```mermaid
flowchart TB
    subgraph Raw["data/raw"]
        IM[imagenette2-320/]
    end

    subgraph Cache["data/cache"]
        FGSM[fgsm/*.pt]
        PGD[pgd/*.pt]
        CW[cw/*.pt]
    end

    subgraph API["data/pipeline.py"]
        EP[EpisodeDataset]
        FP[FerretDataPipeline]
    end

    IM --> EP
    FGSM --> EP
    PGD --> EP
    CW --> EP
    EP --> FP
    FP --> ENV[FerretVisionEnv.reset]
```

- **Clean / adversarial mix** controlled by `adversarial_ratio`
- **Attack type** uniform over `{fgsm, pgd, cw}` when adversarial
- Precompute cache before training when `precompute_adversarial=True`

---

## Repository layout

```
ferret/
├── data/           # Download, datasets, adversarial cache, pipeline
├── env/            # FerretVisionEnv, unified factory
├── policy/         # VisionEncoder, FerretPolicy trunk
├── agents/         # Probe, confidence, decision
├── reward/         # MORL reward, λ schedule, normalizer
├── train/          # PPO, MORL scalarization, logging
├── graph/          # LangGraph nodes + FerretDetector
├── eval/           # POC probe, benchmark
├── ferret/         # Shared constants, EpisodeState
└── docs/           # This file
```

---

## Build phases (from spec)

| Phase | Status | Deliverable |
|-------|--------|-------------|
| 1 Vision POC | ✅ `eval/poc_probe.py` | Hardcoded probes, confidence separation |
| 2 RL policy | ✅ `train/ppo_train.py` | PPO + MORL on ImageNette |
| 3 Baselines | 🔲 | Feature Squeezing, Mahalanobis in `eval/benchmark.py` |
| 4 Unified / language | 🔲 | `language_env`, OLMo encoder |
| 5 Self-play | 🔲 | Attacker–detector co-training |

---

## How to run

### 1. Phase 1 POC (detection signal)

```bash
python -m eval.poc_probe --episodes 50 --attack-types fgsm --download
```

### 2. Training

```bash
# Fast dev run (FGSM cache only)
python -m train.ppo_train \
  --download-data \
  --attack-types fgsm \
  --total-timesteps 500000 \
  --num-envs 4 \
  --track  # optional wandb + TensorBoard
```

```bash
# Full attack mix (precompute PGD + CW first — slow)
python -m train.ppo_train \
  --attack-types fgsm pgd cw \
  --precompute-adversarial \
  --lambda-anneal
```

Logs: `runs/<exp_name>__<seed>__<ts>/` (TensorBoard + `policy.pt`)

### 3. Benchmark

```bash
python -m eval.benchmark --episodes 100 --checkpoint runs/<run>/policy.pt
```

### 4. Inference

```python
from graph import FerretDetector
detector = FerretDetector.from_checkpoint("runs/.../policy.pt")
result = detector.detect(image)
```

---

## Design constraints (spec-aligned)

| Constraint | Implementation |
|------------|----------------|
| Reward hacking via confidence | `ConfidenceAggregator` is rule-based, not learned |
| Budget collapse | Terminal efficiency bonus ∝ remaining budget |
| Action collapse | PPO entropy coefficient `ent_coef` |
| Attack overfitting | Mixed FGSM + PGD + CW from cache |
| Preference exploitation | Uniform Dirichlet sampling |
| λ too large early | λ annealing `0.01 → 0.05` |

---

## Future work

- Feature Squeezing + Mahalanobis baselines (Phase 3)
- Language modality + shared trunk transfer (Phase 4)
- Self-play attacker loop (Phase 5)
- Optional: native `morl-baselines` envelope training alongside current PPO
