"""
resampling.py

Module to resample 3D tomograms and associated coordinates
to a uniform voxel spacing.
"""

import numpy as np
from scipy.ndimage import zoom
from typing import List, Tuple
import logging
from src import config

MAX_Z_DIM = 1200 # Maximum Z dimension for resampling
MAX_MEM_BYTES = 100 * (1024 ** 3)  # 100 GiB

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def resample_volume(volume: np.ndarray, original_spacing: float, target_spacing: float = config.TARGET_VOXEL_SPACING) -> Tuple[np.ndarray, float]:
    """
    Resample a 3D volume to match the target voxel spacing.

    Parameters
    ----------
    volume : np.ndarray
        The original 3D volume (Z, Y, X).
    original_spacing : float
        Original voxel spacing (angstroms per voxel).
    target_spacing : float
        Desired voxel spacing.

    Returns
    -------
    np.ndarray
        The resampled 3D volume.
    """
    if original_spacing == target_spacing:
        logger.info(f"Voxel spacing already matches target; skipping resampling. Volume shape: {volume.shape}")
        return volume, 1.0

    zoom_z = original_spacing / target_spacing
    max_zoom_z = 4.0
    if zoom_z > max_zoom_z:
        logger.warning(f"Capping Z-axis zoom factor from {zoom_z:.2f} to {max_zoom_z}")
        zoom_z = max_zoom_z

    # already defined globally
    target_z = int(volume.shape[0] * zoom_z)
    if target_z > MAX_Z_DIM:
        adjusted_zoom_z = MAX_Z_DIM / volume.shape[0]
        logger.warning(f"Downscaling Z zoom from {zoom_z:.2f} to {adjusted_zoom_z:.2f} to fit max Z={MAX_Z_DIM}")
        zoom_z = adjusted_zoom_z

    zoom_factors = [zoom_z, 1.0, 1.0]
    logger.info(f"Resampling only Z-axis with zoom factor {zoom_factors[0]:.4f}, Y and X remain unchanged.")
    target_shape = tuple(int(s * z) for s, z in zip(volume.shape, zoom_factors))

    estimated_bytes = np.prod(target_shape) * np.dtype(volume.dtype).itemsize
    if estimated_bytes > MAX_MEM_BYTES:
        raise MemoryError(f"Resampled volume too large: shape {target_shape}, size {estimated_bytes / (1024**3):.2f} GiB")

    resampled = zoom(volume, zoom=zoom_factors, order=1)  # Linear interpolation
    logger.info(f"Resampled volume shape: {resampled.shape}")
    return resampled.astype(volume.dtype), zoom_z


def resample_coordinates(
    coords: List[Tuple[int, int, int]],
    original_spacing: float,
    target_spacing: float = config.TARGET_VOXEL_SPACING
) -> List[Tuple[int, int, int]]:
    """
    Adjust motor coordinates to match the resampled voxel spacing.

    Parameters
    ----------
    coords : List[Tuple[int, int, int]]
        Original (z, y, x) coordinates.
    original_spacing : float
        Original voxel spacing.
    target_spacing : float
        Desired voxel spacing.

    Returns
    -------
    List[Tuple[int, int, int]]
        Adjusted coordinates for the new spacing.
    """
    if original_spacing == target_spacing:
        logger.info("Coordinate spacing matches target; no adjustment needed.")
        return coords

    scale = original_spacing / target_spacing
    logger.info(f"Rescaling coordinates by factor {scale:.4f}")
    return [tuple(int(round(c * scale)) for c in coord) for coord in coords]
