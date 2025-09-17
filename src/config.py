"""Configuration file for project paths and constants."""

# src/config.py
import os

# Define base project directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.join("/data/horse/ws/kein254g-team_project")


SRC = os.path.join(PROJECT_ROOT, "src")


# Data directories
RAW_DATA_DIR = os.path.join(WORKSPACE_ROOT, "data", "raw")
SAMPLED_DATA_DIR = os.path.join(WORKSPACE_ROOT, "data", "sampled")

# File paths
TRAIN_LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "train_labels.csv")
SAMPLED_LABELS_PATH = os.path.join(SAMPLED_DATA_DIR, "sampled_train_labels.csv")
DENORMALIZED_RESULTS = os.path.join(PROJECT_ROOT, "data", "raw", "denormalized_results.csv")


SAMPLED_TRAIN_DIR = os.path.join(SAMPLED_DATA_DIR, "sampled_train")

# File path for full labels CSV (from original dataset, not uploaded to GitHub)
FULL_LABELS_PATH = os.path.join(RAW_DATA_DIR, "train_labels.csv")

FULL_DATA_TRAIN_DIR = os.environ.get(
    "FULL_DATA_TRAIN_DIR",
    os.path.join(PROJECT_ROOT, "..", "byu-locating-bacterial-flagellar-motors-2025", "train"),
)

# Directory for externally stored sampled training images
EXTERNAL_SAMPLED_TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "sampled", "sampled_train")

SAMPLED_TRAIN_DATASET_DIR = os.path.join(WORKSPACE_ROOT, "data", "sampled_train")
TRAIN_DATASET_DIR = os.path.join(WORKSPACE_ROOT, "data", "train")
TRAIN_DATASET_HPC_DIR = os.path.join(WORKSPACE_ROOT, "train")

USE_SAMPLED_TRAIN_DATASET = False

PREPROCESSED_DATASET_DIR = os.path.join(WORKSPACE_ROOT, "data", "preprocessed_data")
YOLO_DATA_DIR = os.path.join(WORKSPACE_ROOT, "data", "yolo")
AUGMENTED_YOLO_DATA = os.path.join(PROJECT_ROOT, "data", "augmented_yolo_data")

YOLO_RESULT = os.path.join(WORKSPACE_ROOT, "data", "runs", "yolo", "train")
YOLO_RESULT_PREDICT = os.path.join(WORKSPACE_ROOT, "data", "runs", "yolo", "predict")
YOLO_TRAIN_RESULT = os.path.join(WORKSPACE_ROOT, "data", "runs", "yolo", "train")
YOLO_OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "data", "runs", "yolo", "predict", "val_predictions")


RTDETR_RESULT = os.path.join(WORKSPACE_ROOT, "rtdetr_batch", "exp2")
NMS_RESULTS_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "nms_results.csv")


RTDETR_RESULT_LABELS = os.path.join(RTDETR_RESULT, "labels")
