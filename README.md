# Ferret

Sequential adversarial probing agent — RL-trained policy for efficient adversarial input detection.

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for system diagrams, module map, and run instructions.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Phase 1: verify probe signal
python -m eval.poc_probe --episodes 50 --attack-types fgsm --download

# Phase 2: train policy
python -m train.ppo_train --download-data --attack-types fgsm --num-envs 4
```
