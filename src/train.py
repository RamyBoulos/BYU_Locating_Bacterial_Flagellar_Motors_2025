import sys
import yaml
from ultralytics import RTDETR, YOLO
from src.types.types import ModelType
import torch
import os



# Add project root to system path to allow src module imports
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root set to: {project_root}")

from src import config 

DATA_ROOT = os.getenv("DATA_ROOT", str(config.DATASET_DIR))
print(f"Using DATA_ROOT: {DATA_ROOT}")
data_cfg = {
    "path": DATA_ROOT,                 # <<< now points to staged copy
    "train": "images/train",
    "val":   "images/val",
    "test":  "images/test",
    "nc": 1,
    "names": ["motor"],
}

with open(f"{config.SRC}/data.yaml", "w") as f:
    yaml.dump(data_cfg, f)

print(f"Data configuration saved to {config.SRC}/data.yaml")
# is cuda available
print("CUDA available:", torch.cuda.is_available())


def _suggest_workers():
    # Respect SLURM, avoid oversubscription; YOLO uses one "workers" value for all loaders
    cpus = int(os.getenv("SLURM_CPUS_PER_TASK", "6"))
    print(f"SLURM_CPUS_PER_TASK={cpus}")
    # Leave 1 CPU for Python/CUDA, split roughly across train/val internally
    return max(1, min(6, max(1, cpus - 1) // 2))

def train_model(type: ModelType):
    workers = _suggest_workers()
    print(f"Using workers={workers}")
    device_arg = 0 if torch.cuda.is_available() else "cpu"  # single-GPU job

    if type == ModelType.RTDETR:
        model = RTDETR("rtdetr-l.pt") # rtdetr-l.pt for experimenting, rtdetr-x.pt for training
        project_path = config.RTDETR_TRAINING_RESULT
        project_name = "motor_rtdetr_l_1024"
    elif type == ModelType.YOLO:
        model = YOLO("yolov8n.pt")  # yolov8n.pt for experimenting, yolov8x.pt for training
        project_path = config.YOLO_TRAINING_RESULT
        project_name = "motor_yolo_1024"

    results = model.train(
        data=f"{config.SRC}/data.yaml",
        project=project_path,
        name=project_name,

        # --- runtime sizing (tune to fit your SLURM time) ---
        epochs=2,                # increase later; resume=True for continuation
        batch=8,                # was 16; slightly lower to avoid timeouts
        imgsz=896,             # 1024x1024 for final; 896 for faster iteration

        # --- performance knobs ---
        workers=6,         # <<< key fix for your warning
        device=device_arg,
        cache="disk",            # safer on shared clusters than RAM; try "ram" if fits
        rect=True,
        multi_scale=False,

        # --- optim schedule ---
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.1,
        cos_lr=True,
        warmup_epochs=3,
        patience=30,

        # --- data/aug (kept modest; mosaic/mixup off for stability) ---
        augment=True,
        hsv_h=0.0, hsv_s=0.1, hsv_v=0.2,
        degrees=0.0, translate=0.05, scale=0.10, shear=0.0, perspective=0.0,
        fliplr=0.5, flipud=0.5,
        mosaic=0.0, mixup=0.0, cutmix=0.0,

        # --- logging/saving ---
        save=True,
        save_period=1,
        verbose=True,
        # deterministic=False,  # optional Ultralytics flag (8.2+); we already disabled via torch
    )
    return results


if __name__ == "__main__":
    #train_model(type=ModelType.RTDETR)
    train_model(type=ModelType.YOLO)