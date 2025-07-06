"""
volume_io.py

Module for loading 3D tomograms from .jpg slice directories into numpy arrays.
"""

import os
import numpy as np
from PIL import Image
from typing import Tuple
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_volume_from_jpegs(tomo_dir: str, sort: bool = True, use_tqdm: bool = False) -> np.ndarray:
    """
    Load a stack of 2D .jpg images from a directory into a 3D numpy array.

    Parameters
    ----------
    tomo_dir : str
        Path to the directory containing .jpg slices.
    sort : bool, optional
        Whether to sort filenames alphabetically before stacking. Default is True.
    use_tqdm : bool, optional
        Whether to display a progress bar while loading slices. Default is False.

    Returns
    -------
    np.ndarray
        3D volume of shape (Z, Y, X) with dtype=np.uint8.
    """
    if not os.path.isdir(tomo_dir):
        raise FileNotFoundError(f"Directory not found: {tomo_dir}")

    # List all .jpg files (excluding .jpeg for consistency with dataset)
    files = [f for f in os.listdir(tomo_dir) if f.lower().endswith(".jpg")]
    if not files:
        raise ValueError(f"No .jpg image slices found in directory: {tomo_dir}")

    if sort:
        files.sort()

    logger.info(f"Loading {len(files)} .jpg slices from directory: {tomo_dir}")
    slices = []
    iterator = tqdm(files, desc="Loading slices") if use_tqdm else files
    for fname in iterator:
        img_path = os.path.join(tomo_dir, fname)
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        slices.append(np.array(img, dtype=np.uint8))

    volume = np.stack(slices, axis=0)  # Shape: (Z, Y, X)
    logger.info(f"Loaded volume shape: {volume.shape}")
    return volume


def get_volume_shape(tomo_dir: str) -> Tuple[int, int, int]:
    """
    Get the shape of the 3D volume from .jpg stack dimensions.

    Parameters
    ----------
    tomo_dir : str
        Path to the directory containing .jpg slices.

    Returns
    -------
    Tuple[int, int, int]
        The shape of the volume as (Z, Y, X).
    """
    volume = load_volume_from_jpegs(tomo_dir)
    return volume.shape
