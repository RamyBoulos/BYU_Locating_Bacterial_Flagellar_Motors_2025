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
import shutil

from src.types.types import DatasetType

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


def find_all_predicted_tomo_id_attributes(tomo_id: str) -> Union[pd.DataFrame, None]:
    """Find all predicted attributes for a given tomo_id."""
    labels_df = pd.read_csv(config.DENORMALIZED_RESULTS)
    filtered_df = labels_df[labels_df["tomo_id"] == tomo_id]

    if filtered_df.empty:
        return None

    return filtered_df


def find_all_tomo_ids(type: DatasetType) -> list:
    """Find all unique tomo_ids in the training labels."""
    labels_df = get_csv_data_frame(type)
    unique_tomo_ids = labels_df["tomo_id"].unique().tolist()
    return unique_tomo_ids


def find_all_train_tomo_ids() -> list:
    """Find all unique tomo_ids in the train labels."""
    labels_df = pd.read_csv(config.TRAIN_CSV_PATH)
    unique_tomo_ids = labels_df["tomo_id"].unique().tolist()
    return unique_tomo_ids


def get_csv_data_frame(csv_type: DatasetType) -> pd.DataFrame:
    """Get the CSV data frame for a specific dataset type."""
    if csv_type == DatasetType.TRAIN:
        return pd.read_csv(config.TRAIN_CSV_PATH)
    elif csv_type == DatasetType.VALID:
        return pd.read_csv(config.VALIDATION_CSV_PATH)
    elif csv_type == DatasetType.TEST:
        return pd.read_csv(config.TEST_CSV_PATH)
    elif csv_type == DatasetType.ALL:
        return pd.read_csv(config.FULL_LABELS_PATH)
    else:
        raise ValueError(f"Unknown dataset type: {csv_type}")


def find_all_tomo_id_attributes(tomo_id: str) -> Union[pd.DataFrame, None]:
    """Find all attributes for a given tomo_id."""
    labels_df = pd.read_csv(config.TRAIN_LABELS_PATH)
    filtered_df = labels_df[labels_df["tomo_id"] == tomo_id]

    if filtered_df.empty:
        return None  # Return None if tomo_id not found

    return filtered_df


def find_z_axis_length(tomo_id: str) -> int:
    """Find z axis length for a given tomo_id."""
    attributes = find_all_tomo_id_attributes(tomo_id)
    if attributes is not None and not attributes.empty:
        return int(attributes.iloc[0]["Array shape (axis 0)"])
    return -1


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


def generate_filename(tomo_id: str, slice_idx: float) -> str:
    """Generate a filename based on tomo_id and slice index."""
    # tomo_0a8f05_slice_0000.jpg
    # tomo_0a8f05_slice_0299.jpg
    if int(slice_idx) < 0:
        return ""

    return f"{tomo_id}_slice_{int(slice_idx):04d}.jpg"


def generate_absolute_path(tomo_id: str, slice_idx: float) -> str:
    """Generate a filename based on tomo_id and slice index."""
    # /data/horse/ws/kein254g-team_project/train/tomo_0a8f05/slice_0000.jpg
    # /data/horse/ws/kein254g-team_project/train/tomo_0a8f05/slice_0299.jpg
    if int(slice_idx) < 0:
        return ""

    return os.path.join(config.TRAIN_DATASET_HPC_DIR, tomo_id, f"slice_{int(slice_idx):04d}.jpg")


def generate_absolute_paths__for_tomo_id(tomo_id: str):
    """Generate filenames for all slices of a given tomo_id."""
    z_length = find_z_axis_length(tomo_id)
    if z_length == "":
        return []
    filenames = [
        # os.path.join(config.TRAIN_DATASET_HPC_DIR, tomo_id, f"slice_{i:04d}.jpg")
        generate_absolute_path(tomo_id, i)
        for i in range(int(z_length))
    ]

    return filenames


def add_absolute_paths_to_csv():
    "Read csv train and generate absolute paths for each tomo_id"

    df = pd.read_csv(config.TRAIN_CSV_PATH)
    df["absolute_path"] = df.apply(
        lambda row: generate_absolute_path(row["tomo_id"], row["Motor axis 0"]), axis=1
    )

    # append absolute path column to csv in same data frame df
    df.to_csv(config.TRAIN_CSV_PATH, index=False)


def target_file_paths(tomo_id: str, type: DatasetType = DatasetType.TRAIN) -> list:
    """Generate target file paths for YOLO format based on tomo_id."""
    z_length = find_z_axis_length(tomo_id)

    if z_length == "":
        return []
    target_paths = [
        os.path.join(fetch_dataset_directory(type), generate_filename(tomo_id, i))
        for i in range(int(z_length))
    ]
    return target_paths


def fetch_dataset_directory(type: DatasetType = DatasetType.TRAIN) -> str:
    dir_path = ""
    if type == DatasetType.TRAIN:
        dir_path = config.YOLO_TRAIN_FORMAT_DIR
    elif type == DatasetType.VALID:
        dir_path = config.YOLO_VAL_FORMAT_DIR
    elif type == DatasetType.TEST:
        dir_path = config.YOLO_TEST_FORMAT_DIR
    elif type == DatasetType.ALL:
        dir_path = config.YOLO_TRAIN_FORMAT_DIR
    else:
        raise ValueError(f"Unknown dataset type: {type}")
    return dir_path


def get_labels_directory(type: DatasetType) -> str:
    dir_path = ""
    if type == DatasetType.TRAIN:
        dir_path = config.TRAIN_LABELS_TXT_PATH
    elif type == DatasetType.VALID:
        dir_path = config.VAL_LABELS_TXT_PATH
    elif type == DatasetType.TEST:
        dir_path = config.TEST_LABELS_TXT_PATH
    else:
        raise ValueError(f"Unknown dataset type: {type}")
    return dir_path


def get_positive_tomo_id_slices(tomo_id: str):
    # fix here
    df = pd.read_csv(config.TRAIN_LABELS_PATH)
    tomo_id_df = df[df["tomo_id"] == tomo_id]
    positive_slices = tomo_id_df[tomo_id_df["Motor axis 0"] > 0]["Motor axis 0"].tolist()
    return positive_slices

def get_negative_tomo_id_slices(tomo_id: str):
    all_tomo_id_slices = get_all_tomo_id_slices(tomo_id)
    positive_slices = get_positive_tomo_id_slices(tomo_id)
    negative_slices = list(set(all_tomo_id_slices) - set(positive_slices))
    return negative_slices

def get_all_tomo_id_slices(tomo_id: str):
    slice_length = find_z_axis_length(tomo_id)
    generate_filename(tomo_id, slice_length)
    return [i for i in range(slice_length)]

def copy_files(positive_tomo_id_slices , negative_tomo_id_slices, tomo_id: str , type: DatasetType):
    dest_dir = fetch_dataset_directory(type)
    for slice_idx in positive_tomo_id_slices:
        print(generate_absolute_path(tomo_id, slice_idx))
        src = generate_absolute_path(tomo_id, slice_idx)
        dst = os.path.join(dest_dir, generate_filename(tomo_id, slice_idx))
      
        dst_path = Path(dst)

        if dst_path.exists():
            print(f"Skipping {dst} as it already exists.")
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f"Copy {src} to {dst}")


    for slice_idx in negative_tomo_id_slices:
        print(generate_absolute_path(tomo_id, slice_idx))
        src = generate_absolute_path(tomo_id, slice_idx)
        dst = os.path.join(dest_dir, generate_filename(tomo_id, slice_idx))

        dst_path = Path(dst)

        if dst_path.exists():
            print(f"Skipping {dst} as it already exists.")
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f"Copy {src} to {dst}")

    return 


def move_files_to_yolo_format(type: DatasetType = DatasetType.TRAIN):
    """Move files to YOLO format directory structure."""
    # This function is a placeholder and needs to be implemented based on specific requirements.
    tomos = find_all_tomo_ids(type)
    count = 0

    for tomo_id in tomos:
        src_files = generate_absolute_paths__for_tomo_id(tomo_id)
        dst_files = target_file_paths(tomo_id)
        for src, dst in zip(src_files, dst_files):
            dst_path = Path(dst)

            if dst_path.exists():
                print(f"Skipping {dst} as it already exists.")
                continue

            dst_path.parent.mkdir(parents=True, exist_ok=True)  # klasörü yoksa oluştur
            shutil.copy(src, dst)  # taşıma istiyorsan copy yerine move yaz
            print(f"{src} -> {dst}")
        count += 1
        print(f"Processed {count}/{len(tomos)}: {tomo_id}")




#if __name__ == "__main__":
    # print(find_all_tomo_id_attributes("tomo_00e047"))
    #move_files_to_yolo_format(type=DatasetType.VALID)
