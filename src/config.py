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
TRAIN_LABELS_PATH = os.path.join(PROJECT_ROOT, "data","raw",  "train_labels.csv")
SAMPLED_LABELS_PATH = os.path.join(SAMPLED_DATA_DIR, "sampled_train_labels.csv")
DENORMALIZED_RESULTS = os.path.join(PROJECT_ROOT,"data","raw", "denormalized_results.csv")


SAMPLED_TRAIN_DIR = os.path.join(SAMPLED_DATA_DIR, "sampled_train")

# File path for full labels CSV (from original dataset, not uploaded to GitHub)
FULL_LABELS_PATH = os.path.join(RAW_DATA_DIR, "train_labels.csv")

# Directory for full training tomograms (each tomo_id folder with .jpg slices)
# NOTE: This folder is not uploaded to the GitHub repository due to its large size.
#         You must manually set this path according to your local setup if needed.
#         You can set the FULL_DATA_TRAIN_DIR environment variable to override the default.
FULL_DATA_TRAIN_DIR = os.environ.get(
	"FULL_DATA_TRAIN_DIR",
	os.path.join(PROJECT_ROOT, "..", "byu-locating-bacterial-flagellar-motors-2025", "train")
)

# Directory for externally stored sampled training images
EXTERNAL_SAMPLED_TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "sampled", "sampled_train")

SAMPLED_TRAIN_DATASET_DIR = os.path.join(WORKSPACE_ROOT, "data", "sampled_train")
TRAIN_DATASET_DIR  = os.path.join(WORKSPACE_ROOT, "data", "train")
TRAIN_DATASET_HPC_DIR  = os.path.join(WORKSPACE_ROOT, "train")

USE_SAMPLED_TRAIN_DATASET = False

PREPROCESSED_DATASET_DIR = os.path.join(WORKSPACE_ROOT,"data","preprocessed_data")
YOLO_DATA_DIR =  os.path.join(WORKSPACE_ROOT, "data", "yolo")
RTDETR_DATA_DIR =  os.path.join(WORKSPACE_ROOT, "data", "rtdetr")


YOLO_TRAIN_DIR =  os.path.join(YOLO_DATA_DIR, "images", "train")
YOLO_VAL_DIR =  os.path.join(YOLO_DATA_DIR, "images", "val")

AUGMENTED_YOLO_DATA =  os.path.join(PROJECT_ROOT, "data", "augmented_yolo_data")

YOLO_RESULT = os.path.join(WORKSPACE_ROOT, "data", "runs", "yolo", "train")
YOLO_RESULT_PREDICT = os.path.join(WORKSPACE_ROOT, "data", "runs", "yolo", "predict")
YOLO_WEIGHTS_PATH   = os.path.join(WORKSPACE_ROOT, "data", "runs", "yolo", "train", "motor_detection", "weights", "best.pt")
YOLO_WEIGHTS_TEST_PATH = os.path.join(WORKSPACE_ROOT, "data", "runs", "yolo", "train", "motor_detection_test", "weights" ,"best.pt")
YOLO_TRAIN_RESULT   = os.path.join(WORKSPACE_ROOT, "data", "runs", "yolo", "train")                   
YOLO_OUTPUT_DIR     =  os.path.join(WORKSPACE_ROOT, "data", "runs", "yolo", "predict", "val_predictions")


RTDETR_RESULT = os.path.join(WORKSPACE_ROOT, "data", "runs", "rtdetr", "train")
RTDETR_RESULT_PREDICT = os.path.join(WORKSPACE_ROOT, "data", "runs", "rtdetr", "predict")
RTDETR_WEIGHTS_PATH   = os.path.join(WORKSPACE_ROOT, "data", "runs", "rtdetr", "train", "motor_detection", "weights", "best.pt")
RTDETR_WEIGHTS_TEST_PATH = os.path.join(WORKSPACE_ROOT, "data", "runs","rtdetr", "train", "motor_detection_test", "weights" ,"best.pt")
RTDETR_TRAIN_RESULT   = os.path.join(WORKSPACE_ROOT, "data", "runs", "rtdetr", "train")                   
RTDETR_OUTPUT_DIR     =  os.path.join(WORKSPACE_ROOT, "data", "runs","rtdetr", "predict", "val_predictions")

TEST_DIR = os.path.join(WORKSPACE_ROOT, "test")


TEST_FLAT_DIR = os.path.join(WORKSPACE_ROOT, "test_flat")
RTDETR_RESULT = os.path.join(WORKSPACE_ROOT, "rtdetr_batch", "exp2")
RTDETR_PREDICTION_NORMALIZED_CSV = os.path.join(RAW_DATA_DIR, "predictions_normalized.csv")
RTDETR_PREDICTION_NORMALIZED_CSV = os.path.join(RAW_DATA_DIR, "denormalized_results.csv")
NMS_RESULTS_CSV = os.path.join(PROJECT_ROOT, "data","raw", "nms_results.csv")


RTDETR_PREDICTION_CSV = os.path.join(RAW_DATA_DIR, "predictions.csv")


RTDETR_RESULT_LABELS = os.path.join(RTDETR_RESULT, "labels")


