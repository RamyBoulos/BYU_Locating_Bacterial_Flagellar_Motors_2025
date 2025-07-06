"""
dataset_builder.py

Pipeline to preprocess a single tomogram into a normalized volume,
resampled heatmap, and saved dataset outputs for 3D U-Net training.
"""

import os
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import argparse
from preprocessing_3d.volume_io import load_volume_from_jpegs
from preprocessing_3d.label_processing import load_labels, filter_valid_labels, group_labels_by_tomo
from preprocessing_3d.normalization import normalize_intensity
from preprocessing_3d.resampling import resample_volume, resample_coordinates
from preprocessing_3d.heatmap_generator import generate_gaussian_heatmap
from preprocessing_3d.save_utils import save_all_outputs, append_zoom_log_csv
from src import config
from src.data_utils.loader import get_data_sources

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_tomogram(tomo_id: str, label_dict: dict, output_dir: str, log_path: str):
    """
    Run full preprocessing on a single tomogram.

    Parameters
    ----------
    tomo_id : str
        The tomogram ID to process.
    label_dict : dict
        Dictionary mapping tomo_id to list of (z, y, x) coordinates.
    output_dir : str
        Directory where the processed data should be saved.
    log_path : str
        Path to the log file for recording preprocessing info.
    """
    tomo_dir = os.path.join(config.SAMPLED_TOMO_DIR if config.USE_SAMPLED_TRAIN_DATASET
                            else config.FULL_DATA_TRAIN_DIR, tomo_id)

    logger.info(f"Processing {tomo_id} from {tomo_dir}")

    volume = load_volume_from_jpegs(tomo_dir, use_tqdm=True)
    spacing = label_dict[tomo_id]["voxel_spacing"]
    coords = label_dict[tomo_id]["coords"]

    norm_volume = normalize_intensity(volume, method="minmax")


    resampled_volume, zoom_z = resample_volume(norm_volume, original_spacing=spacing)
    resampled_coords = resample_coordinates(coords, original_spacing=spacing)

    MAX_VOXELS = 800_000_000  # ~3.2 GB for float32

    if resampled_volume.size > MAX_VOXELS:
        logger.warning(f"Skipping tomogram {tomo_id}: volume too large ({resampled_volume.shape}, {resampled_volume.size} voxels)")
        append_zoom_log_csv(
            tomo_id=tomo_id,
            original_spacing=spacing,
            target_spacing=config.TARGET_VOXEL_SPACING,
            applied_zoom_z=zoom_z,
            resampled_shape=resampled_volume.shape,
            skipped=True,
            log_path=log_path
        )
        logger.info(f"Logged skipped tomogram to: {log_path}")
        return

    heatmap = generate_gaussian_heatmap(
        resampled_volume.shape,
        resampled_coords,
        sigma=config.GAUSSIAN_SIGMA,
        patch_radius=config.GAUSSIAN_PATCH_RADIUS
    )

    metadata = {
        "tomo_id": tomo_id,
        "original_spacing": spacing,
        "target_spacing": config.TARGET_VOXEL_SPACING,
        "original_shape": volume.shape,
        "final_shape": resampled_volume.shape,
        "resampled_shape": list(resampled_volume.shape),
        "num_motors": len(coords),
        "applied_zoom_z": zoom_z,
        "zoom_capped": bool(zoom_z < (spacing / config.TARGET_VOXEL_SPACING)),
        "heatmap_dtype": str(heatmap.dtype),
    }

    save_all_outputs(tomo_id, resampled_volume, heatmap, metadata, output_dir)

    append_zoom_log_csv(
        tomo_id=tomo_id,
        original_spacing=spacing,
        target_spacing=config.TARGET_VOXEL_SPACING,
        applied_zoom_z=zoom_z,
        resampled_shape=resampled_volume.shape,
        skipped=False,
        log_path=log_path
    )
    logger.info(f"Logged preprocessing details to: {log_path}")


def preprocess_all(limit: int = None):
    """
    Run preprocessing on all tomograms listed in the labels file.
    Parameters
    ----------
    limit : int, optional
        Maximum number of tomograms to process (for testing). If None, process all.
    """
    label_path, tomo_root = get_data_sources()
    df = load_labels(csv_path=label_path)
    df_valid = filter_valid_labels(df)
    grouped = group_labels_by_tomo(df_valid)

    # Set up shared log path for this run
    log_dir = os.path.join(config.PROJECT_ROOT, "data", "preprocessed_data_3d")
    os.makedirs(log_dir, exist_ok=True)
    existing_logs = [f for f in os.listdir(log_dir) if re.match(r"preprocessing_log_\d+\.csv", f)]
    log_numbers = [int(re.findall(r"\d+", f)[0]) for f in existing_logs]
    next_log_number = max(log_numbers, default=0) + 1
    log_filename = f"preprocessing_log_{next_log_number}.csv"
    log_path = os.path.join(log_dir, log_filename)

    for i, tomo_id in enumerate(grouped):
        if limit is not None and i >= limit:
            break
        preprocess_tomogram(tomo_id, grouped, config.PREPROCESSED_DATASET_DIR, log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess tomograms for 3D U-Net training.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of tomograms to process (for testing).")
    args = parser.parse_args()

    preprocess_all(limit=args.limit)
