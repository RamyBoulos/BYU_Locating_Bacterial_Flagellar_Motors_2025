from pathlib import Path
import sys
import yaml


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

print("data.yaml written successfully")