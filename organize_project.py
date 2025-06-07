import os
import shutil
from pathlib import Path

# === Base path (assumed current directory is repo root) ===
repo_root = "."

# === Folders to create ===
folders = [
    "data/sampled",
    "data/raw",
    "notebooks",
    "src/data_utils",
    "src/models",
    "src/training",
    "outputs/logs",
    "outputs/predictions",
    "outputs/checkpoints"
]

for folder in folders:
    path = os.path.join(repo_root, folder)
    os.makedirs(path, exist_ok=True)

# === Move small files (if they exist) ===
file_moves = {
    "../sampled_train_labels.csv": "data/sampled/sampled_train_labels.csv",
    "../sampled_train.zip": "data/sampled/sampled_train.zip",
    "../t_motors.ipynb": "notebooks/EDA.ipynb",
    "../f_motors.py": "src/data_utils/loader.py"
}

for src, dst in file_moves.items():
    dst_path = os.path.join(repo_root, dst)
    if os.path.exists(src):
        shutil.move(src, dst_path)
        print(f"✅ Moved {src} → {dst_path}")
    else:
        print(f"⚠️ File not found: {src}")

# === Create symlinks to large data (if available) ===
symlinks = {
    "data/raw/train": "../../byu-locating-bacterial-flagellar-motors-2025/train",
    "data/raw/test": "../../byu-locating-bacterial-flagellar-motors-2025/test",
    "data/raw/train_labels.csv": "../../byu-locating-bacterial-flagellar-motors-2025/train_labels.csv",
}

for link_name, target in symlinks.items():
    link_path = os.path.join(repo_root, link_name)
    target_path = os.path.abspath(os.path.join(repo_root, target))
    if not os.path.exists(target_path):
        print(f"❌ Skipping symlink, target not found: {target_path}")
        continue
    try:
        if not os.path.exists(link_path):
            os.symlink(target_path, link_path)
            print(f"🔗 Symlink created: {link_path} → {target_path}")
        else:
            print(f"✅ Symlink already exists: {link_path}")
    except OSError as e:
        print(f"❌ Failed to create symlink {link_path}: {e}")

# === Create YOLOv8-specific folders ===
yolo_folders = [
    "data/yolo/images/train",
    "data/yolo/images/val",
    "data/yolo/labels/train",
    "data/yolo/labels/val"
]

for folder in yolo_folders:
    path = os.path.join(repo_root, folder)
    os.makedirs(path, exist_ok=True)
    print(f"📁 Created: {path}")

# === Remind user about big data ===
print("\n🚫 Large folders like 'sampled_train/' were NOT moved to avoid Git issues.")
print("✅ Use .gitignore to exclude them from commits.")

# === Create requirements.txt for EDA + YOLOv8 ===
requirements = [
    "ultralytics",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "opencv-python",
    "Pillow",
    "tqdm",
    "scikit-learn",
    "albumentations",
    "pyyaml"
]

req_path = os.path.join(repo_root, "requirements.txt")
with open(req_path, "w") as f:
    f.write("\n".join(requirements))
print(f"\n✅ requirements.txt created at {req_path}")