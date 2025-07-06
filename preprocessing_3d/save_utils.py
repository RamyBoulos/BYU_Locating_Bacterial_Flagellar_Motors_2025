"""
save_utils.py

Utilities for saving preprocessed tomogram volumes, heatmaps,
and metadata to disk in a structured format.
"""

import os
import json
import numpy as np
import logging

import csv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def save_npy(array: np.ndarray, path: str):
    """
    Save a NumPy array to a .npy file.

    Parameters
    ----------
    array : np.ndarray
        The array to save.
    path : str
        File path where the array will be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)
    logger.info(f"Saved array to {path}")


def save_json(data: dict, path: str):
    """
    Save a dictionary as a JSON file.

    Parameters
    ----------
    data : dict
        The metadata to save.
    path : str
        File path where the JSON will be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved metadata to {path}")

def save_all_outputs(tomo_id: str, volume: np.ndarray, heatmap: np.ndarray, metadata: dict, output_dir: str):
    """
    Save volume, heatmap, and metadata to disk for a given tomogram.

    Parameters
    ----------
    tomo_id : str
        Unique identifier of the tomogram.
    volume : np.ndarray
        The 3D volume to save.
    heatmap : np.ndarray
        The 3D heatmap to save.
    metadata : dict
        Associated metadata to save as JSON.
    output_dir : str
        Root directory to save the outputs under.
    """
    out_dir = os.path.join(output_dir, tomo_id)

    volume_path = os.path.join(out_dir, "volume.npy")
    save_npy(volume, volume_path)
    logger.info(f"Volume saved ({os.path.getsize(volume_path) / 1e6:.2f} MB)")

    heatmap_path = os.path.join(out_dir, "heatmap.npy")
    save_npy(heatmap, heatmap_path)
    logger.info(f"Heatmap saved ({os.path.getsize(heatmap_path) / 1e6:.2f} MB)")

    metadata_path = os.path.join(out_dir, "metadata.json")
    save_json(metadata, metadata_path)
    logger.info(f"Metadata saved ({os.path.getsize(metadata_path) / 1e3:.2f} KB)")


# --- Logging preprocessing details to a central CSV log ---
def append_zoom_log_csv(tomo_id: str, original_spacing: float, target_spacing: float,
                        applied_zoom_z: float, resampled_shape: tuple,
                        skipped: bool, log_path: str):
    """
    Append preprocessing metadata for a tomogram to a central CSV log.

    Parameters
    ----------
    tomo_id : str
        Identifier for the tomogram.
    original_spacing : float
        Original voxel spacing.
    target_spacing : float
        Target voxel spacing.
    applied_zoom_z : float
        Actual zoom factor applied to the Z axis.
    resampled_shape : tuple
        Final shape of the volume after resampling.
    skipped : bool
        Whether the tomogram was skipped due to memory or other constraints.
    log_path : str
        Path to the output CSV log file.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    header = ["tomo_id", "original_spacing", "target_spacing", "applied_zoom_z", "resampled_shape", "skipped"]
    row = [tomo_id, original_spacing, target_spacing, applied_zoom_z, str(resampled_shape), skipped]

    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
