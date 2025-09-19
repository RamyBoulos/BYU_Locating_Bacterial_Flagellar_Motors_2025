"""Utilities for parsing predictions and denormalizing YOLO outputs.

Functions
---------
- find_all_predicted_tomo_id_attributes: return rows for a tomo_id from denormalized results
- find_all_tomo_id_attributes: return rows for a tomo_id from training labels
- find_tomo_id_slice_attributes: return rows for a tomo_id and slice number
- find_width_height: get (width, height) for a tomo_id + slice number
- parse_txt: parse one YOLO .txt prediction file
- rescale_letterbox: denormalize predictions to pixel space and export CSV
"""

import glob
import os
import sys
from pathlib import Path
from typing import Union
import pandas as pd

# Ensure repository root is on sys.path so `src` is importable
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src import config  # noqa: E402


def parse_csv_row(line: str):
    """Parse a CSV row into its components."""
    parts = line.strip().split(",")
    return {
        "tomo_id": parts[0],
        "slice": int(parts[1]),
        "confidence": float(parts[2]),
        "x_center": float(parts[3]),
        "y_center": float(parts[4]),
        "width": float(parts[5]),
        "height": float(parts[6]),
    }


def train_test_validation_split():
    """Generate train, validation, and test CSV files from the full training labels."""

    df = pd.read_csv(config.TRAIN_LABELS_PATH)

    print(f"Original set size: {len(df)}")

    train_df = df.sample(frac=0.7, replace=False, random_state=1)
    validation_df = df.drop(train_df.index)
    test_df = validation_df.sample(frac=1 / 3, replace=False, random_state=1)
    validation_df = validation_df.drop(test_df.index)

    train_df.to_csv(config.TRAIN_CSV_PATH, index=False)
    validation_df.to_csv(config.VALIDATION_CSV_PATH, index=False)
    test_df.to_csv(config.TEST_CSV_PATH, index=False)


def find_all_predicted_tomo_id_attributes(tomo_id: str) -> Union[pd.DataFrame, None]:
    """Find all predicted attributes for a given tomo_id."""
    labels_df = pd.read_csv(config.DENORMALIZED_RESULTS)
    filtered_df = labels_df[labels_df["tomo_id"] == tomo_id]

    if filtered_df.empty:
        return None

    return filtered_df


def find_all_tomo_id_attributes(tomo_id: str) -> Union[pd.DataFrame, None]:
    """Find all attributes for a given tomo_id."""
    labels_df = pd.read_csv(config.TRAIN_LABELS_PATH)
    filtered_df = labels_df[labels_df["tomo_id"] == tomo_id]

    if filtered_df.empty:
        return None  # Return None if tomo_id not found

    return filtered_df


def find_tomo_id_slice_attributes(tomo_id, slice_num) -> Union[pd.DataFrame, None]:
    """Find attributes for a given tomo_id and slice_num."""
    print(config.TRAIN_LABELS_PATH)
    labels_df = pd.read_csv(config.TRAIN_LABELS_PATH)
    filtered_df = labels_df[(labels_df["tomo_id"] == tomo_id)]
    filtered_df = labels_df[(labels_df["Motor axis 0"] == slice_num)]

    if filtered_df.empty:
        return None  # Return None if tomo_id or slice_num not found

    return filtered_df


def find_width_height(tomo_id, slice_num):
    """Find width and height for a given tomo_id and slice_num."""
    attributes = find_tomo_id_slice_attributes(tomo_id, slice_num)

    if attributes is None or attributes.empty:
        print(f"No attributes found for tomo_id: {tomo_id}, slice_num: {slice_num}")
        return None, None

    width = attributes.iloc[0]["Array shape (axis 2)"]
    height = attributes.iloc[0]["Array shape (axis 1)"]
    print(f"Width: {width}, Height: {height}")
    return width, height


def parse_txt(file_path, f):
    """Parse a YOLO format .txt file and return its components."""
    file_name = os.path.basename(file_path)  # e.g. "image1.txt"
    file_id = os.path.splitext(file_name)[0]  # e.g. "image1"
    _, tomo_num, _, slice_idx = file_id.split("_")
    tomo_id = "tomo_" + str(tomo_num)
    _, confidence, x_center, y_center, width, height = f.readline().strip().split()
    return tomo_id, int(slice_idx), confidence, x_center, y_center, width, height


def rescale_letterbox():
    """Rescale and denormalize YOLO predictions to original image dimensions."""
    rows = []

    print(f"{config.RTDETR_RESULT_LABELS}/*.txt")
    txt_files = sorted(glob.glob(f"{config.RTDETR_RESULT_LABELS}/*.txt"))
    for file_path in txt_files:
        print(f"\n--- Reading: {file_path} ---")
        with open(file_path, "r", encoding="utf-8") as f:
            # print(parse_txt(file_path,f))
            tomo_id, slice_number, confidence, x_center, y_center, width, height = parse_txt(
                file_path, f
            )
            img_width, img_height = find_width_height(tomo_id, slice_number)

            if img_width is None or img_height is None:
                print(f"Skipping {tomo_id}, {slice_number} due to missing dimensions.")
                continue

            # denomarize
            x_center = float(x_center) * img_width
            y_center = float(y_center) * img_height
            width = float(width) * img_width
            height = float(height) * img_height

            # what about voxel spacing ?

            rows.append(
                {
                    "tomo_id": tomo_id,
                    "slice": slice_number,
                    "confidence": float(confidence),
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(config.DENORMALIZED_RESULTS, index=False)


if __name__ == "__main__":
    # print(find_all_tomo_id_attributes("tomo_00e047"))
    print(train_test_validation_split())
