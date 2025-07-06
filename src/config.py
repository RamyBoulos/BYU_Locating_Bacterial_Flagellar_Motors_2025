# src/config.py
import os
import glob

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

SAMPLED_TRAIN_DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "sampled_train")
TRAIN_DATASET_DIR  = os.path.join(PROJECT_ROOT, "data", "train")
USE_SAMPLED_TRAIN_DATASET = False

PREPROCESSED_DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "preprocessed_data_3d")  # Directory for 3D preprocessed data

# Automatically find the latest preprocessing log CSV (used during training)
log_files = sorted(glob.glob(os.path.join(PREPROCESSED_DATASET_DIR, "preprocessing_log_*.csv")))
PREPROCESSING_LOG_CSV = log_files[-1] if log_files else None

YOLO_DATA_DIR =  os.path.join(PROJECT_ROOT, "data", "yolo")
AUGMENTED_YOLO_DATA =  os.path.join(PROJECT_ROOT, "data", "augmented_yolo_data")

# Sampled tomogram directories (not tracked in Git, local path must exist)
SAMPLED_TOMO_DIR = os.path.join(PROJECT_ROOT, "sampled_train")

# Warn if SAMPLED_TOMO_DIR does not exist
if not os.path.exists(SAMPLED_TOMO_DIR):
    print(f"⚠️ WARNING: SAMPLED_TOMO_DIR not found: {SAMPLED_TOMO_DIR}")

# Example tomogram (for testing or reference)
EXAMPLE_TOMO_ID = "tomo_0a8f05"
EXAMPLE_TOMO_DIR = os.path.join(SAMPLED_TOMO_DIR, EXAMPLE_TOMO_ID)

# Target voxel spacing for 3D resampling (angstroms per voxel)
TARGET_VOXEL_SPACING = 15.6  # Median_Voxel_Value

# Heatmap generation parameters
GAUSSIAN_SIGMA = 2.0
GAUSSIAN_PATCH_RADIUS = 6

# Flag to determine whether to use log-filtered tomograms during training
USE_LOG_FILTER = True

# Load filtered tomogram IDs from latest log CSV if enabled
if USE_LOG_FILTER and PREPROCESSING_LOG_CSV:
    import pandas as pd
    try:
        df = pd.read_csv(PREPROCESSING_LOG_CSV)
        LOG_FILTERED_TOMO_IDS = df.loc[~df["skipped"], "tomo_id"].tolist()
    except Exception:
        LOG_FILTERED_TOMO_IDS = []
else:
    LOG_FILTERED_TOMO_IDS = None
