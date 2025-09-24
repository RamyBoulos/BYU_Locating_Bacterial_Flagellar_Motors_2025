from pathlib import Path
import sys
import yaml
from ultralytics import RTDETR, YOLO
from src.types.types import ModelType



# Ensure repository root is on sys.path so `src` is importable
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src import config 

data_cfg = {
    "path": config.DATASET_DIR,
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 1,  # number of classes
    "names": ["motor"],  # class names
}

with open(f"{config.SRC}/data.yaml", "w") as f:
    yaml.dump(data_cfg, f)

print(f"Data configuration saved to {config.SRC}/data.yaml")

def train_model(type: ModelType):
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
    epochs=10,
    patience=30,
    batch=16,
    imgsz=1024,
    device="cuda",
    optimizer="AdamW",
    lr0=1e-4,
    lrf=0.1,
    cos_lr=True,
    warmup_epochs=3,
    cache="disk",
    augment=True,
    hsv_h=0.0,
    hsv_s=0.1,
    hsv_v=0.2,
    degrees=0.0,
    translate=0.05,
    scale=0.10,
    shear=0.0,
    perspective=0.0,
    fliplr=0.5,
    flipud=0.5,
    mosaic=0.0,
    mixup=0.0,
    cutmix=0.0,
    rect=True,
    multi_scale=False,
    save_period=1,
    verbose=True,
)
    print(results)


#if __name__ == "__main__":
    #train_model(type=ModelType.RTDETR)
    #train_model(type=ModelType.YOLO)