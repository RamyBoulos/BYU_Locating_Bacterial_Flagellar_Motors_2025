"""
heatmap_generator.py

Generate 3D heatmaps from annotated motor coordinates for training
with U-Net models using peak localization.
"""

import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_gaussian_heatmap(
    shape: Tuple[int, int, int],
    coordinates: List[Tuple[int, int, int]],
    sigma: float = 2.0,
    patch_radius: int = 6
) -> np.ndarray:
    """
    Generate a sparse 3D heatmap with localized Gaussian peaks around each coordinate.

    Parameters
    ----------
    shape : Tuple[int, int, int]
        Shape of the output heatmap (Z, Y, X).
    coordinates : List[Tuple[int, int, int]]
        List of (z, y, x) motor positions.
    sigma : float, optional
        Standard deviation of the Gaussian. Default is 2.0.
    patch_radius : int, optional
        Radius of the local patch window. Default is 6.

    Returns
    -------
    np.ndarray
        Sparse 3D heatmap with Gaussian blobs around each motor coordinate.
    """
    heatmap = np.zeros(shape, dtype=np.float32)

    if not coordinates:
        logger.info("No motor coordinates provided; returning empty heatmap.")
        return heatmap

    # Create a local Gaussian patch once
    size = 2 * patch_radius + 1
    zz, yy, xx = np.meshgrid(
        np.arange(size) - patch_radius,
        np.arange(size) - patch_radius,
        np.arange(size) - patch_radius,
        indexing='ij'
    )
    patch = np.exp(-(zz**2 + yy**2 + xx**2) / (2 * sigma ** 2)).astype(np.float32)

    for idx, (z, y, x) in enumerate(coordinates):
        if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
            logger.warning(f"Skipped coordinate {idx} (z={z}, y={y}, x={x}) - out of bounds.")
            continue

        z_start, y_start, x_start = max(z - patch_radius, 0), max(y - patch_radius, 0), max(x - patch_radius, 0)
        z_end, y_end, x_end = min(z + patch_radius + 1, shape[0]), min(y + patch_radius + 1, shape[1]), min(x + patch_radius + 1, shape[2])

        patch_z_start = patch_radius - (z - z_start)
        patch_y_start = patch_radius - (y - y_start)
        patch_x_start = patch_radius - (x - x_start)

        patch_z_end = patch_radius + (z_end - z)
        patch_y_end = patch_radius + (y_end - y)
        patch_x_end = patch_radius + (x_end - x)

        heatmap[z_start:z_end, y_start:y_end, x_start:x_end] = np.maximum(
            heatmap[z_start:z_end, y_start:y_end, x_start:x_end],
            patch[patch_z_start:patch_z_end, patch_y_start:patch_y_end, patch_x_start:patch_x_end]
        )

    logger.info(f"Generated heatmap with {len(coordinates)} peaks using sparse Gaussian patches.")
    return heatmap
