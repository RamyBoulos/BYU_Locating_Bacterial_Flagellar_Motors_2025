import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter, label, center_of_mass
from typing import List, Tuple

__all__ = [
    "compute_mse",
    "detect_peaks",
    "average_peak_distance",
    "compute_fbeta_score",
    "load_dataset",
]

# ----------------------
# Regression loss metric
# ----------------------

def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Mean Squared Error (MSE) between predicted and target heatmaps.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted heatmap tensor of shape (N, 1, D, H, W).
    target : torch.Tensor
        Ground truth heatmap tensor of shape (N, 1, D, H, W).

    Returns
    -------
    float
        Mean squared error.
    """
    return F.mse_loss(pred, target).item()


# ------------------------
# Peak detection utilities
# ------------------------

def detect_peaks(volume: np.ndarray, threshold: float = 0.5, size: int = 3) -> List[Tuple[int, int, int]]:
    """
    Detect peaks in a 3D heatmap volume using maximum filter.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (Z, Y, X) to detect peaks in.
    threshold : float, optional
        Minimum value to consider as a peak.
    size : int, optional
        Neighborhood size for local maxima.

    Returns
    -------
    List[Tuple[int, int, int]]
        List of (z, y, x) coordinates for detected peaks.
    """
    if volume.size == 0:
        return []
    maxima = maximum_filter(volume, size=size) == volume
    mask = (volume > threshold) & maxima
    labeled, _ = label(mask)
    centers = center_of_mass(volume, labeled, range(1, labeled.max() + 1))
    return [tuple(map(int, c)) for c in centers]


def average_peak_distance(pred_peaks: List[Tuple[int, int, int]], target_peaks: List[Tuple[int, int, int]]) -> float:
    """
    Compute average Euclidean distance between predicted and target peaks.

    Parameters
    ----------
    pred_peaks : list of tuple
        List of predicted (z, y, x) coordinates.
    target_peaks : list of tuple
        List of ground truth (z, y, x) coordinates.

    Returns
    -------
    float
        Average distance between closest predicted-target peak pairs.
        Returns inf if either list is empty.
    """
    if not pred_peaks or not target_peaks:
        return float("inf")
    distances = []
    for gt in target_peaks:
        dists = [np.linalg.norm(np.array(gt) - np.array(pred)) for pred in pred_peaks]
        distances.append(min(dists))
    return float(np.mean(distances)) if distances else float("inf")


# -----------------------
# Precision/Recall metric
# -----------------------

def compute_fbeta_score(
    pred_peaks: List[Tuple[int, int, int]],
    target_peaks: List[Tuple[int, int, int]],
    tolerance: float = 3.0,
    beta: float = 1.0
) -> float:
    """
    Compute the F-beta score between predicted and ground truth peaks.

    Parameters
    ----------
    pred_peaks : list of tuple
        List of predicted (z, y, x) coordinates.
    target_peaks : list of tuple
        List of ground truth (z, y, x) coordinates.
    tolerance : float, optional
        Maximum distance between predicted and ground truth peak to be considered a match.
    beta : float, optional
        Weight of recall in the F-score. beta < 1 favors precision, beta > 1 favors recall.

    Returns
    -------
    float
        F-beta score.
    """
    matched = 0
    used: set[int] = set()
    for gt in target_peaks:
        for i, pred in enumerate(pred_peaks):
            if i in used:
                continue
            dist = np.linalg.norm(np.array(gt) - np.array(pred))
            if dist <= tolerance:
                matched += 1
                used.add(i)
                break

    tp = matched
    fp = len(pred_peaks) - matched
    fn = len(target_peaks) - matched

    if tp + fp + fn == 0:
        return 1.0

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)


# -----------------------
# Dataset loading utility
# -----------------------

import os
import pandas as pd
from .dataset import Tomo3DDataset

def load_dataset(dataset_dir: str, log_path: str) -> Tomo3DDataset:
    """
    Load the dataset using filtered tomogram IDs from a preprocessing log.

    Parameters
    ----------
    dataset_dir : str
        Root directory where preprocessed tomogram folders are stored.
    log_path : str
        Path to the latest preprocessing log CSV.

    Returns
    -------
    Tomo3DDataset
        Dataset instance filtered by tomograms that were successfully processed.
    """
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"No preprocessing log CSV found at: {log_path}")
    df = pd.read_csv(log_path)
    tomo_ids = df.loc[~df["skipped"], "tomo_id"].tolist()
    return Tomo3DDataset(root_dir=dataset_dir, tomo_ids=tomo_ids)