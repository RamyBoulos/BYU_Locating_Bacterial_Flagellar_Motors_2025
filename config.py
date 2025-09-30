"""Project configuration constants."""

import os

# Paths
BASE_DATA_DIR = "/data/horse/ws/rabo074f-team_project/byu-locating-bacterial-flagellar-motors-2025/train"
LABELS_CSV    = "/data/horse/ws/rabo074f-team_project/BYU2/train_labels.csv"
EVAL_DEBUG_DIR = "/data/horse/ws/rabo074f-team_project/BYU2/eval_debug"

# Patch/training
RESIZE_SHAPE = (128, 960, 960)  # (Z, Y, X) after resampling
NUM_EPOCHS = 100
PATCH_SIZE = (64, 128, 128)  # (Z, Y, X)
# SIGMA = 4                     # Gaussian heatmap sigma
BATCH_SIZE = 1
NUM_WORKERS = 2

# Model
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR        = "./logs"

# Evaluation
QUANTILE_THRESHOLD = 0.9997
MAX_PREDICTIONS_PER_TOMO = 10
DISTANCE_THRESHOLD_ANGSTROM = 1000.0  # in Angstroms
