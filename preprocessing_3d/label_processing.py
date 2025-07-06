"""
label_processing.py

Module for loading, cleaning, and grouping flagellar motor labels
from the training CSV file.
"""

import os
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from src.data_utils.loader import get_data_sources

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_labels(csv_path: str = None) -> pd.DataFrame:
    """
    Load the training labels CSV as a DataFrame.

    Parameters
    ----------
    csv_path : str, optional
        Path to the CSV file containing training labels.
        If None, uses sampled or full labels based on centralized config.

    Returns
    -------
    pd.DataFrame
        Raw label data.
    """
    if csv_path is None:
        csv_path, _ = get_data_sources()

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Label file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded labels from: {csv_path} ({len(df)} rows)")
    return df


def filter_valid_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out rows with placeholder (-1.0) motor coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with motor labels.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with valid motor rows only.
    """
    filtered_df = df[(df["Motor axis 0"] >= 0) &
                     (df["Motor axis 1"] >= 0) &
                     (df["Motor axis 2"] >= 0)]
    logger.info(f"Filtered valid labels: {len(filtered_df)} out of {len(df)} rows")
    return filtered_df


def group_labels_by_tomo(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Group valid motor coordinates and voxel spacing by tomo_id.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with valid motor coordinates.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping tomo_id to a dict with 'coords' and 'voxel_spacing'.
    """
    grouped = {}
    for tomo_id, group in df.groupby("tomo_id"):
        coords = list(zip(group["Motor axis 0"].astype(int),
                          group["Motor axis 1"].astype(int),
                          group["Motor axis 2"].astype(int)))
        spacing = group["Voxel spacing"].iloc[0]
        grouped[tomo_id] = {
            "coords": coords,
            "voxel_spacing": spacing
        }
    logger.info(f"Grouped labels into {len(grouped)} tomogram entries")
    return grouped
