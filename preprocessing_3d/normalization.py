

"""
normalization.py

Module for intensity normalization of 3D tomogram volumes.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def normalize_intensity(volume: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize a 3D volume's intensity values.

    Parameters
    ----------
    volume : np.ndarray
        The input 3D volume to normalize.
    method : str, optional
        Normalization method: "minmax" (default) or "zscore".

    Returns
    -------
    np.ndarray
        The normalized 3D volume.
    """
    if method == "minmax":
        v_min, v_max = np.min(volume), np.max(volume)
        if v_max == v_min:
            logger.warning("Volume has constant intensity; returning zeros.")
            return np.zeros_like(volume, dtype=np.float32)
        normalized = (volume - v_min) / (v_max - v_min)
    elif method == "zscore":
        mean, std = np.mean(volume), np.std(volume)
        if std == 0:
            logger.warning("Volume has zero standard deviation; returning zeros.")
            return np.zeros_like(volume, dtype=np.float32)
        normalized = (volume - mean) / std
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    logger.info(f"Volume normalized using '{method}' method.")
    return normalized.astype(np.float32)
