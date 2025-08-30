import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
import pandas as pd


def ensure_project_root_in_sys_path(project_root: Optional[str] = None) -> str:
    """Ensure project root is in sys.path and return it."""
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def build_positive_samples(labels_df: pd.DataFrame, data_root: str) -> Dict[str, Tuple[float, float, int, int]]:
    """Return mapping from image path to (x, y, img_width, img_height) for positive samples."""
    positive_samples: Dict[str, Tuple[float, float, int, int]] = {}
    for _, row in labels_df.iterrows():
        if row["Motor axis 0"] == -1:
            continue
        tomo_id = row["tomo_id"]
        z = int(row["Motor axis 0"])
        y = row["Motor axis 1"]
        x = row["Motor axis 2"]
        img_width = row["Array shape (axis 2)"]
        img_height = row["Array shape (axis 1)"]
        img_name = f"slice_{z:04d}.jpg"
        img_path = os.path.join(data_root, tomo_id, img_name)
        positive_samples[img_path] = (x, y, img_width, img_height)
    return positive_samples


def collect_all_images(data_root: str, exts: Optional[List[str]] = None) -> List[str]:
    """Return list of all image paths under data_root filtered by extensions."""
    if exts is None:
        exts = [".jpg"]
    all_images: List[str] = []
    for root, _, files in os.walk(data_root):
        for fname in files:
            for ext in exts:
                if fname.lower().endswith(ext):
                    all_images.append(os.path.join(root, fname))
                    break
    return all_images


def balance_paths(df: pd.DataFrame, pipeline: Any) -> Tuple[Any, Any]:
    """Apply imbalanced-learn pipeline to balance dataset and return X_res, y_res."""
    X_res, y_res = pipeline.fit_resample(df[["path"]], df["label"])
    return X_res, y_res


def process_images(image_list: List[str], positive_samples: Dict[str, Tuple[float, float, int, int]], img_out_dir: str, lbl_out_dir: str, box_size: int) -> None:
    """Copy images to output dirs and write YOLO-format label files for positives."""
    for img_path in image_list:
        if not os.path.isfile(img_path):
            print(f"⚠️  Skipping missing image: {img_path}")
            continue
        tomo_id = os.path.basename(os.path.dirname(img_path))
        z = int(os.path.splitext(os.path.basename(img_path))[0].split("_")[-1])
        base_fn = f"{tomo_id}_slice_{z:04d}"
        dst_img = os.path.join(img_out_dir, f"{base_fn}.jpg")
        dst_lbl = os.path.join(lbl_out_dir, f"{base_fn}.txt")
        shutil.copy(img_path, dst_img)
        lines: List[str] = []
        if img_path in positive_samples:
            x, y, w, h = positive_samples[img_path]
            cx, cy = x / w, y / h
            bw, bh = (box_size * 2) / w, (box_size * 2) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(dst_lbl, "w") as f:
            f.write("\n".join(lines))
