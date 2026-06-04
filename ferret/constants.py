"""Shared Ferret constants for vision RL."""

NUM_GRID_CELLS = 49
NUM_PERT_TYPES = 4
NUM_MAGNITUDES = 3
NUM_PROBE_ACTIONS = NUM_GRID_CELLS * NUM_PERT_TYPES * NUM_MAGNITUDES  # 588

GRID_SIDE = 7
CELL_SIZE = 32
IMAGE_SIZE = 224

NUM_CLASSES = 1000
INPUT_EMBED_DIM = 2048

MAX_BUDGET = 10
PREFERENCE_DIM = 4

# Transformer policy defaults (spec section 5.2 / 6.1).
POLICY_D_MODEL = 128
POLICY_NHEAD = 4
POLICY_NLAYERS = 3
POLICY_MLP_HIDDEN = 256

# MORL reward defaults (spec section 4).
DEFAULT_LAMBDA_EFF = 0.05
DEFAULT_FP_BETA = 0.5
DEFAULT_GEN_GAMMA = 0.2
DECISION_THRESHOLD = 0.5

MAGNITUDE_SCALES = (0.01, 0.03, 0.06)

# MORL objective names (order matches RewardVector.as_array).
MORL_OBJECTIVE_NAMES = ("accuracy", "efficiency", "false_positive", "generalization")

# Query-efficiency λ schedule (spec §4.4): start small, ramp up during training.
LAMBDA_EFF_START = 0.01
LAMBDA_EFF_END = 0.05
