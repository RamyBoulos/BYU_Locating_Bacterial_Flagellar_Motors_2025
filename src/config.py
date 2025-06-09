# src/config.py
import os

# Define base project directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data directories
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
SAMPLED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sampled")

# File paths
TRAIN_LABELS_PATH = os.path.join(RAW_DATA_DIR, "train_labels.csv")

SAMPLED_LABELS_PATH = os.path.join(SAMPLED_DATA_DIR, "sampled_train_labels.csv")

SAMPLED_TRAIN_DIR = os.path.join(SAMPLED_DATA_DIR, "sampled_train")

# File path for full labels CSV (from original dataset, not uploaded to GitHub)
FULL_LABELS_PATH = os.path.join(RAW_DATA_DIR, "train_labels.csv")

# Directory for full training tomograms (each tomo_id folder with .jpg slices)
# NOTE: This folder is not uploaded to the GitHub repository due to its large size.
#         You must manually set this path according to your local setup if needed.
FULL_DATA_TRAIN_DIR = os.path.join(PROJECT_ROOT, "..", "byu-locating-bacterial-flagellar-motors-2025", "train")

# Directory for externally stored sampled training images
EXTERNAL_SAMPLED_TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "sampled", "sampled_train")