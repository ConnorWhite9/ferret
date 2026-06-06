.PHONY: install install-dev poc train-fgsm train-full cache eval test lint fmt help

PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
CKPT   ?= ""

help:
	@echo "Ferret — Sequential Adversarial Probing Agent"
	@echo ""
	@echo "Setup"
	@echo "  make install      Create .venv and install runtime deps"
	@echo "  make install-dev  Install + dev extras (pytest, ruff)"
	@echo ""
	@echo "Data"
	@echo "  make cache        Precompute FGSM adversarial cache (train + val)"
	@echo ""
	@echo "Training"
	@echo "  make poc          Phase 1 — hardcoded probe signal check"
	@echo "  make train-fgsm   Phase 2 — PPO on FGSM only (fast)"
	@echo "  make train-full   Phase 2 — PPO on FGSM+PGD+CW (full)"
	@echo "  make resume CKPT=runs/.../policy.pt"
	@echo ""
	@echo "Evaluation"
	@echo "  make eval CKPT=runs/.../policy.pt"
	@echo "  make eval-random  Baseline-only eval (no checkpoint needed)"
	@echo ""
	@echo "Dev"
	@echo "  make test         Run pytest"
	@echo "  make lint         Ruff lint check"
	@echo "  make fmt          Ruff auto-format"

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install pytest pytest-cov ruff matplotlib scikit-learn pyyaml

# ── Data ─────────────────────────────────────────────────────────────────────

cache:
	$(PYTHON) -c "\
from data import DataConfig, FerretDataPipeline; \
from policy.vision_encoder import VisionEncoder; \
from train.ppo_train import _target_model_for_foolbox; \
import torch; \
enc = VisionEncoder(device=torch.device('cpu')); \
cfg = DataConfig(adversarial_ratio=0.5, attack_types=('fgsm',), download=True); \
p = FerretDataPipeline(cfg, model=_target_model_for_foolbox(enc), device=torch.device('cpu')); \
p.ensure_adversarial_cache('train'); \
p.ensure_adversarial_cache('val'); \
print('Cache ready.')"

# ── Training ─────────────────────────────────────────────────────────────────

poc:
	$(PYTHON) -m eval.poc_probe --episodes 50 --attack-types fgsm --download

train-fgsm:
	$(PYTHON) -m train.ppo_train \
		--exp-name ferret_fgsm \
		--seed 42 \
		--attack-types fgsm \
		--adversarial-ratio 0.5 \
		--precompute-adversarial \
		--download-data \
		--total-timesteps 500000 \
		--num-envs 4 \
		--lambda-anneal \
		--eval-every 100 \
		--checkpoint-every 50

train-full:
	$(PYTHON) -m train.ppo_train \
		--exp-name ferret_full \
		--seed 42 \
		--attack-types fgsm pgd cw \
		--adversarial-ratio 0.5 \
		--precompute-adversarial \
		--download-data \
		--total-timesteps 1000000 \
		--num-envs 4 \
		--lambda-anneal \
		--eval-every 100 \
		--checkpoint-every 50 \
		--track

resume:
	@test -n "$(CKPT)" || (echo "Usage: make resume CKPT=runs/.../policy.pt"; exit 1)
	$(PYTHON) -m train.ppo_train \
		--resume $(CKPT) \
		--attack-types fgsm \
		--adversarial-ratio 0.5 \
		--download-data \
		--num-envs 4 \
		--lambda-anneal

# ── Evaluation ────────────────────────────────────────────────────────────────

eval:
	@test -n "$(CKPT)" || (echo "Usage: make eval CKPT=runs/.../policy.pt"; exit 1)
	$(PYTHON) -m eval.benchmark \
		--checkpoint $(CKPT) \
		--episodes 200 \
		--fit-episodes 100 \
		--attack-types fgsm \
		--plot

eval-random:
	$(PYTHON) -m eval.benchmark \
		--episodes 200 \
		--fit-episodes 100 \
		--attack-types fgsm \
		--plot \
		--no-baselines

ablation:
	@test -n "$(CKPT)" || (echo "Usage: make ablation CKPT=runs/.../policy.pt"; exit 1)
	$(PYTHON) -m eval.ablation \
		--checkpoint $(CKPT) \
		--episodes 200 \
		--attack-types fgsm \
		--plot

# ── Dev ──────────────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	$(PYTHON) -m ruff check .

fmt:
	$(PYTHON) -m ruff format .
