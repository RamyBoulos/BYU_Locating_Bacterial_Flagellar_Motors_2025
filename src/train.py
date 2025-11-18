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

data_cfg = {
    "path": str(config.DATASET_DIR),                 
    "train": "images/train",
    "val":   "images/val",
    "nc": 1,
    "names": ["motor"],
}

with open(f"{config.SRC}/data.yaml", "w") as f:
    yaml.dump(data_cfg, f)

print(f"Dataset directory: {config.DATASET_DIR}")
print(f"Data configuration saved to {config.SRC}/data.yaml")
print("CUDA available:", torch.cuda.is_available())



def train_model(type: ModelType):

    if type == ModelType.RTDETR:
        model = RTDETR("rtdetr-x.pt") # rtdetr-l.pt for experimenting, rtdetr-x.pt for training
        project_path = config.RTDETR_TRAINING_RESULT
        project_name = "motor_rtdetr_x_1024"
    elif type == ModelType.YOLO:
        model = YOLO("yolov8x.pt")  # yolov8n.pt for experimenting, yolov8x.pt for training
        project_path = config.YOLO_TRAINING_RESULT
        project_name = "motor_yolo_1024"

    results = model.train(
    data=f"{config.SRC}/data.yaml",
    project=project_path,
    name=project_name,

    # --- optimization ---
    optimizer="AdamW",
    lr0=1e-4, lrf=0.1,           # start higher than 1e-5; 2e-4 is also a good try
    cos_lr=True, warmup_epochs=3,

    # --- schedule ---
    epochs=150, patience=20,     # shorter patience is usually enough

    # --- batch / io ---
    batch=16,
    imgsz=1024,
    device="cuda" if torch.cuda.is_available() else "cpu",
    cache="ram",
    workers=8,                   # adjust to node; <= #CPU cores

    # --- loss weights (tiny-object bias) ---
    box=9.0, cls=0.75, dfl=1.5,

    # --- augmentation (gentle for tiny targets) ---
    augment=True,
    hsv_h=0.0, hsv_s=0.05, hsv_v=0.1,  # set to 0 if images are grayscale
    degrees=10.0, translate=0.05, scale=0.10, shear=0.0, perspective=0.0,
    fliplr=0.5, flipud=0.5,

    # --- mosaic/mix ---
    mosaic=0.0, mixup=0.0, cutmix=0.0,  # tiny single-object: keep off

    # --- batching/layout ---
    rect=False,                 # IMPORTANT: more diversity during train
    multi_scale=False,

    # --- logging/saving ---
    save_period=1,              # reduce disk churn
    verbose=True, plots=True,
    seed=42
)
    print(results)
    return results


if __name__ == "__main__":
    train_model(type=ModelType.RTDETR)
    #train_model(type=ModelType.YOLO)