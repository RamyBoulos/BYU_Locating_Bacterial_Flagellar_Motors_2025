"""
Shared evaluation utilities for motor detection models (3D U-Net).

Mini-Index
----------
1) EvaluationConfig            : Dataclass holding evaluation hyperparameters.
2) extract_peaks               : Peak finding with thresholding + local maxima + NMS.
3) compute_detection_metrics   : Peak extraction + GT matching + metrics (P/R/F1/F2).
4) _log_sample_extremes        : Pretty-print best/worst samples by F2.
5) evaluate                    : Full validation loop, aggregation, threshold sweep.
6) compute_loss_components_stub: Thin wrapper to call training loss (avoid circular import).

Notes
-----
- Coordinates inside volumes are voxel indices [z, y, x].
- Physical distances are computed in Ångström (Å) using `voxel_size`.
- KD-Tree (scipy.spatial.cKDTree) is used for fast nearest-neighbor matching.
"""

from __future__ import annotations  # Ensure future annotations behavior (helps with type hints)

from dataclasses import dataclass  # For concise configuration containers
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence  # Typing aliases

import numpy as np  # Numerical arrays and operations
import torch  # PyTorch tensors and models
from scipy.ndimage import maximum_filter  # Local maxima / neighborhood operations
from scipy.spatial import cKDTree  # Fast KD-Tree for nearest-neighbor search in C
from tqdm import tqdm  # Progress bars for loops


# ---------------------------------------------------------------------
# 1) Configuration
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluationConfig:
    """Immutable container for evaluation hyperparameters.

    Parameters
    ----------
    dist_threshold_angstrom : float
        Maximum allowed distance (Å) between a prediction and a GT point
        to count as a match (True Positive).
    peak_min_distance_angstrom : float
        Minimum allowed spacing (Å) between selected peaks (NMS radius).
    peak_threshold : float
        Probability threshold for considering voxels as candidate peaks.
    threshold_sweep : Sequence[float]
        Optional list of thresholds for a sweep to pick the best one by F2.
    debug_stats_samples : int
        Number of early samples to print min/max/p99 debug stats for.
    sample_summary_count : int
        How many best/worst samples to keep in summaries.
    """
    dist_threshold_angstrom: float
    peak_min_distance_angstrom: float
    peak_threshold: float
    threshold_sweep: Sequence[float]
    debug_stats_samples: int
    sample_summary_count: int


# ---------------------------------------------------------------------
# 2) Peak Extraction
# ---------------------------------------------------------------------
def extract_peaks(
    prob_map: np.ndarray,
    threshold: float,
    voxel_size: np.ndarray | Sequence[float] | None = None,
    min_distance_angstrom: float | None = None,
    seed_window: int = 3,
) -> np.ndarray:
    """Return voxel coordinates of peaks using thresholding + local maxima + NMS.

    This function first thresholds the probability map, then finds local maxima
    within a cubic `seed_window`, then applies Non-Maximum Suppression (NMS):
    - If `voxel_size` and `min_distance_angstrom` are provided, NMS is done in
      physical units (Å). Otherwise a small fixed voxel neighborhood is used.

    Parameters
    ----------
    prob_map : np.ndarray
        3D probability map (values in [0, 1]).
    threshold : float
        Voxel probability threshold for candidate selection.
    voxel_size : array-like or None
        Per-axis voxel size (Å) as [vz, vy, vx]. If None, NMS uses voxel spacing.
    min_distance_angstrom : float or None
        Minimum separation in Å between selected peaks. Ignored if None.
    seed_window : int
        The local neighborhood edge length for maxima detection (≥1).

    Returns
    -------
    np.ndarray
        Array of shape (K, 3) with integer voxel coordinates [[z, y, x], ...].
        Returns an empty array if no peaks are found.
    """
    # Binary mask of "confident" voxels above threshold
    mask = prob_map > float(threshold)  # shape == prob_map.shape, dtype bool

    # Find local maxima within a seed_window neighborhood (if > 1)
    if seed_window and seed_window > 1:  # ensure positive window
        # Local maxima test: a voxel equals the local maximum AND is above threshold
        localmax = (maximum_filter(prob_map, size=seed_window) == prob_map) & mask
        cand_coords = np.argwhere(localmax)  # list of [z, y, x] candidates
    else:
        # If no local maxima window, take all thresholded voxels as candidates
        cand_coords = np.argwhere(mask)

    # Fast exit: no candidates
    if cand_coords.size == 0:
        return np.empty((0, 3), dtype=int)

    # Attach scores to candidates and sort descending by confidence
    candidates = [
        (float(prob_map[z, y, x]), int(z), int(y), int(x))
        for z, y, x in cand_coords
    ]  # list of tuples (score, z, y, x)
    candidates.sort(key=lambda t: t[0], reverse=True)  # highest score first

    selected: list[tuple[float, int, int, int]] = []  # will collect kept peaks

    # Choose NMS mode: physical spacing (Å) if voxel_size & min_distance given; else voxel-based
    if voxel_size is None or min_distance_angstrom is None:
        # Voxel-based NMS: keep peaks at least `min_vox` apart in Chebyshev distance
        min_vox = 3  # conservative small separation in voxels
        for cand in candidates:  # iterate from highest score downward
            _, z, y, x = cand
            keep = True
            for _, sz, sy, sx in selected:
                # Chebyshev distance < min_vox → too close, suppress
                if max(abs(z - sz), abs(y - sy), abs(x - sx)) < min_vox:
                    keep = False
                    break
            if keep:
                selected.append(cand)
    else:
        # Physical NMS: convert voxel deltas to Å using voxel_size
        v = np.asarray(voxel_size, dtype=np.float32).reshape(-1)[:3]  # [vz, vy, vx]
        v = np.maximum(v, 1e-6)  # avoid zeros
        min_d2 = float(min_distance_angstrom) ** 2  # compare squared distances

        for cand in candidates:
            _, z, y, x = cand
            keep = True
            for _, sz, sy, sx in selected:
                # Compute squared physical distance in Å^2
                dz = (z - sz) * v[0]
                dy = (y - sy) * v[1]
                dx = (x - sx) * v[2]
                if (dz * dz + dy * dy + dx * dx) < min_d2:
                    keep = False
                    break
            if keep:
                selected.append(cand)

    # Return only the coordinates (drop scores)
    return np.asarray([[z, y, x] for _, z, y, x in selected], dtype=int)


# ---------------------------------------------------------------------
# 3) Detection Metrics
# ---------------------------------------------------------------------
def compute_detection_metrics(
    pred_map: np.ndarray,
    gt_points: np.ndarray,
    voxel_size: np.ndarray,
    *,
    threshold: float,
    min_distance_angstrom: float,
    dist_threshold_angstrom: float,
) -> Dict[str, Any]:
    """Compute detection metrics by extracting peaks and matching to GT.

    Steps
    -----
    1) Extract predicted peaks from `pred_map` with `threshold` and NMS.
    2) Convert predicted + GT coordinates to physical space (Å) using `voxel_size`.
    3) Use cKDTree to find nearest GT for each prediction within `dist_threshold_angstrom`.
    4) Greedy unique matching (closest first) to compute TP/FP/FN.
    5) Derive precision/recall/F1/F2 and distance stats.

    Parameters
    ----------
    pred_map : np.ndarray
        3D probability map after model sigmoid (0..1), may be masked beforehand.
    gt_points : np.ndarray
        Ground-truth voxel coordinates, shape (G, 3) or compatible.
    voxel_size : np.ndarray
        Per-axis voxel size in Å, broadcastable to (3,).
    threshold : float
        Probability threshold for peak extraction.
    min_distance_angstrom : float
        NMS spacing for peaks in Å (passed to `extract_peaks`).
    dist_threshold_angstrom : float
        Max matching distance (Å) to count a prediction as TP.

    Returns
    -------
    Dict[str, Any]
        Dictionary with counts (tp/fp/fn), metrics (precision/recall/f1/f2),
        distances (avg/max), and samples of unmatched coords.
    """
    def _flatten_coords(points):
        """Normalize inputs to a float32 array of shape (N, 3)."""
        arr = np.asarray(points)
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)        # Single point → (1, 3)
        elif arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])  # Collapse higher dims
        return arr.astype(np.float32)

    voxel_size = np.asarray(voxel_size, dtype=np.float32)  # ensure float array

    # 1) Extract predicted peak coordinates (voxel indices)
    predicted_pts = extract_peaks(
        pred_map,
        threshold=threshold,
        voxel_size=voxel_size,
        min_distance_angstrom=min_distance_angstrom,
    )
    gt_points = _flatten_coords(gt_points)          # normalize GT
    predicted_pts = _flatten_coords(predicted_pts)  # normalize preds
    pred_coords_vox = (
        predicted_pts.round().astype(np.int32) if predicted_pts.size else np.empty((0, 3), dtype=np.int32)
    )

    num_pred = predicted_pts.shape[0]  # number of predicted peaks
    num_gt = gt_points.shape[0]        # number of ground-truth points

    # Handle trivial cases upfront
    if num_pred == 0 and num_gt == 0:
        # Perfect empty case: define metrics as perfect to avoid divide-by-zero
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "f2": 1.0,
            "num_pred": 0,
            "num_gt": 0,
            "avg_tp_distance_angstrom": None,
            "max_tp_distance_angstrom": None,
            "unmatched_gt": [],
            "unmatched_pred": [],
            "pred_coords_vox": [],
        }

    if num_gt == 0:
        # Any prediction is a false positive when GT is empty
        precision = 0.0 if num_pred > 0 else 1.0
        recall = 1.0
        return {
            "tp": 0,
            "fp": int(num_pred),
            "fn": 0,
            "precision": precision,
            "recall": recall,
            "f1": 0.0,
            "f2": 0.0,
            "num_pred": int(num_pred),
            "num_gt": 0,
            "avg_tp_distance_angstrom": None,
            "max_tp_distance_angstrom": None,
            "unmatched_gt": [],
            "unmatched_pred": predicted_pts.astype(int).tolist()[:5],  # sample
            "pred_coords_vox": pred_coords_vox.tolist(),
        }

    # 2) Convert coordinates to physical space (Å)
    gt_angstrom = gt_points * voxel_size       # (G, 3) in Å
    pred_angstrom = predicted_pts * voxel_size # (P, 3) in Å

    # 3) Build KD-Tree over GT for fast nearest-neighbor queries
    kdtree = cKDTree(gt_angstrom)
    distances, indices = kdtree.query(
        pred_angstrom,                                   # query all predictions
        distance_upper_bound=float(dist_threshold_angstrom),  # cap search radius
    )

    # Collect feasible matches (finite distance and valid index)
    candidate_pairs = []
    for pred_idx, (dist, gt_idx) in enumerate(zip(distances, indices)):
        if not np.isfinite(dist):
            continue  # no GT within the radius
        if gt_idx >= len(gt_angstrom):
            continue  # cKDTree returns len if not found; guard against it
        candidate_pairs.append((float(dist), pred_idx, int(gt_idx)))

    # Sort by distance so closer matches are resolved first (greedy unique matching)
    candidate_pairs.sort(key=lambda x: x[0])

    # 4) Greedy one-to-one assignment (each GT matched at most once)
    used_gt = set()     # GT indices already matched
    tp_indices = []     # prediction indices that became TPs
    tp_distances = []   # corresponding distances
    for dist, pred_idx, gt_idx in candidate_pairs:
        if gt_idx in used_gt:
            continue  # GT already matched by a closer prediction
        used_gt.add(gt_idx)
        tp_indices.append(pred_idx)
        tp_distances.append(dist)

    # Compute counts
    tp = len(tp_indices)
    fp = int(num_pred - tp)
    fn = int(num_gt - len(used_gt))

    # 5) Metrics (with small epsilon to guard divisions)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    beta = 2.0
    f2 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + 1e-8)

    # Distance summaries for TPs
    tp_distances_np = np.asarray(tp_distances, dtype=np.float32)
    avg_tp_distance = float(np.mean(tp_distances_np)) if tp > 0 else None
    max_tp_distance = float(np.max(tp_distances_np)) if tp > 0 else None

    # Provide small samples of unmatched coordinates for debugging
    unmatched_pred_indices = np.array(
        sorted(set(range(num_pred)) - set(tp_indices)), dtype=int
    )
    unmatched_gt_indices = np.array(sorted(set(range(num_gt)) - used_gt), dtype=int)

    unmatched_gt_coords = (
        gt_points[unmatched_gt_indices].astype(int).tolist()[:5]
        if unmatched_gt_indices.size > 0
        else []
    )
    unmatched_pred_coords = (
        predicted_pts[unmatched_pred_indices].astype(int).tolist()[:5]
        if unmatched_pred_indices.size > 0
        else []
    )

    # Final structured result dict
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "num_pred": int(num_pred),
        "num_gt": int(num_gt),
        "avg_tp_distance_angstrom": avg_tp_distance,
        "max_tp_distance_angstrom": max_tp_distance,
        "unmatched_gt": unmatched_gt_coords,
        "unmatched_pred": unmatched_pred_coords,
        "pred_coords_vox": pred_coords_vox.tolist(),
    }


# ---------------------------------------------------------------------
# 4) Logging helpers
# ---------------------------------------------------------------------
def _log_sample_extremes(
    best_samples: List[Dict[str, Any]],
    worst_samples: List[Dict[str, Any]],
    prefix: str = "[VAL]"
) -> None:
    """Pretty-print the best and worst validation samples by F2.

    Parameters
    ----------
    best_samples : list of dict
        Top samples with highest F2 (each dict is one entry from sample_metrics).
    worst_samples : list of dict
        Bottom samples with lowest F2.
    prefix : str
        Prefix label for printed lines (e.g., '[VAL]').
    """
    if not best_samples and not worst_samples:
        return  # nothing to print

    if worst_samples:
        print(f"{prefix} Worst samples by F2 (up to {len(worst_samples)}):")
        for sample in worst_samples:
            unmatched_gt = sample.get("unmatched_gt") or []
            unmatched_pred = sample.get("unmatched_pred") or []
            avg_dist = sample.get("avg_tp_distance_angstrom")
            max_dist = sample.get("max_tp_distance_angstrom")
            dist_str = "N/A"
            if avg_dist is not None and max_dist is not None:
                dist_str = f"avg={avg_dist:.1f}Å, max={max_dist:.1f}Å"

            print(
                f"  - {sample['tomo_id']}: F2={sample['f2']:.3f}, F1={sample['f1']:.3f}, "
                f"TP={sample['tp']}, FP={sample['fp']}, FN={sample['fn']}, "
                f"pred={sample['num_pred']}, gt={sample['num_gt']}, match_dist={dist_str}"
            )
            if unmatched_gt:
                print(f"      Missed GT (up to 5) voxel coords: {unmatched_gt}")
            if unmatched_pred:
                print(f"      Extra preds (up to 5) voxel coords: {unmatched_pred}")

    if best_samples:
        print(f"{prefix} Best samples by F2 (up to {len(best_samples)}):")
        for sample in best_samples:
            unmatched_gt = sample.get("unmatched_gt") or []
            unmatched_pred = sample.get("unmatched_pred") or []
            avg_dist = sample.get("avg_tp_distance_angstrom")
            max_dist = sample.get("max_tp_distance_angstrom")
            dist_str = "N/A"
            if avg_dist is not None and max_dist is not None:
                dist_str = f"avg={avg_dist:.1f}Å, max={max_dist:.1f}Å"

            print(
                f"  - {sample['tomo_id']}: F2={sample['f2']:.3f}, F1={sample['f1']:.3f}, "
                f"TP={sample['tp']}, FP={sample['fp']}, FN={sample['fn']}, "
                f"pred={sample['num_pred']}, gt={sample['num_gt']}, match_dist={dist_str}"
            )
            if unmatched_gt:
                print(f"      Missed GT (up to 5) voxel coords: {unmatched_gt}")
            if unmatched_pred:
                print(f"      Extra preds (up to 5) voxel coords: {unmatched_pred}")


# ---------------------------------------------------------------------
# 5) Full evaluation loop
# ---------------------------------------------------------------------
def evaluate(
    model: torch.nn.Module,
    val_loader: Iterable,
    device: torch.device,
    loss_params: Mapping[str, torch.Tensor | float],
    config: EvaluationConfig,
    *,
    desc: str = "[val]",
) -> Dict[str, Any]:
    """Run the validation loop and aggregate detection and voxel metrics.

    Workflow
    --------
    For each batch/sample in `val_loader`:
        1) Forward pass → logits → sigmoid → prob map.
        2) (Optional) Apply mask to ignore invalid regions.
        3) Peak detection + NMS + KD-Tree matching to compute per-sample metrics.
        4) Voxel-wise confusion counts for auxiliary analysis.
        5) Accumulate losses and metrics; perform threshold sweep if requested.
    Finally:
        - Compute dataset-level precision/recall/F1/F2.
        - Summarize best/worst samples and threshold sweep statistics.

    Parameters
    ----------
    model : torch.nn.Module
        Trained 3D U-Net (or compatible) producing logits.
    val_loader : Iterable
        Yields tuples: (volume, gt_heatmap, mask, tomo_id, gt_coords, voxel_size).
    device : torch.device
        Device to run inference on.
    loss_params : Mapping[str, Tensor or float]
        Dict with loss weights/params, passed to `compute_loss_components_stub`.
    config : EvaluationConfig
        Hyperparameters controlling thresholds, spacing, and logging.
    desc : str, optional
        Progress bar label.

    Returns
    -------
    Dict[str, Any]
        Comprehensive evaluation summary (see return block at end).
    """
    was_training = model.training        # remember original mode
    model.eval()                         # set eval mode (no dropout/bn updates)

    # Running totals and storages
    detection_totals = {"tp": 0, "fp": 0, "fn": 0}                    # peak-level
    voxel_totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}               # voxel-level
    f2_list: List[float] = []                                         # per-sample F2
    sample_metrics: List[Dict[str, Any]] = []                         # per-sample dicts
    loss_total = 0.0                                                  # sum of total loss
    num_batches = 0                                                   # for averaging
    loss_component_sums: MutableMapping[str, float] = {               # sum components
        "bce": 0.0, "dice": 0.0, "focal": 0.0
    }
    threshold_sweeps: Dict[float, List[float]] = {thr: [] for thr in config.threshold_sweep}  # thr→list(F2)

    pbar_val = tqdm(val_loader, desc=desc, leave=False)  # progress bar over val set
    with torch.no_grad():                                # disable grad for eval speed
        for volume, gt_heatmap, mask, tomo_id, gt_coords, voxel_size in pbar_val:
            # Move tensors to device
            volume = volume.to(device)
            gt_heatmap_tensor = gt_heatmap.to(device)
            mask_tensor = mask.to(device)

            # Forward pass: logits (same spatial shape) → prob via sigmoid
            logits = model(volume)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # (D, H, W)

            # Convert variable types to numpy arrays for downstream routines
            if isinstance(gt_coords, torch.Tensor):
                gt_coords_np = gt_coords.cpu().numpy()
            else:
                gt_coords_np = np.asarray(gt_coords)

            if isinstance(voxel_size, torch.Tensor):
                voxel_size_np = voxel_size.cpu().numpy()
            else:
                voxel_size_np = np.asarray(voxel_size)

            # Compute and accumulate loss components (delegated to training helper)
            losses = compute_loss_components_stub(
                logits,
                gt_heatmap_tensor,
                mask_tensor,
                loss_params,
            )
            loss_total += float(losses["total"])
            loss_component_sums["bce"] += float(losses["bce"])
            loss_component_sums["dice"] += float(losses["dice"])
            loss_component_sums["focal"] += float(losses["focal"])
            num_batches += 1

            # Extract simple tomogram name if it's a length-1 list/tuple
            tomo_name = (
                tomo_id[0]
                if isinstance(tomo_id, (list, tuple)) and len(tomo_id) == 1
                else tomo_id
            )

            # Prepare numpy arrays for binary and masked ops
            gt_heatmap_np = gt_heatmap_tensor.squeeze().cpu().numpy()  # GT heatmap
            mask_np = mask_tensor.squeeze().cpu().numpy()              # validity mask
            valid_mask = mask_np > 0.5                                 # boolean mask
            prob_masked = prob * mask_np                               # zero out invalid

            # Peak detection + matching metrics at configured peak threshold
            det_metrics = compute_detection_metrics(
                prob_masked,
                gt_coords_np,
                voxel_size_np,
                threshold=config.peak_threshold,
                min_distance_angstrom=config.peak_min_distance_angstrom,
                dist_threshold_angstrom=config.dist_threshold_angstrom,
            )

            # Accumulate detection counts and keep per-sample F2
            f2_list.append(det_metrics["f2"])
            detection_totals["tp"] += det_metrics["tp"]
            detection_totals["fp"] += det_metrics["fp"]
            detection_totals["fn"] += det_metrics["fn"]

            # Optional threshold sweep for F2 vs threshold analysis
            for thr in config.threshold_sweep:
                sweep_metrics = compute_detection_metrics(
                    prob_masked,
                    gt_coords_np,
                    voxel_size_np,
                    threshold=thr,
                    min_distance_angstrom=config.peak_min_distance_angstrom,
                    dist_threshold_angstrom=config.dist_threshold_angstrom,
                )
                threshold_sweeps[thr].append(sweep_metrics["f2"])

            # Early debug stats: intensity distributions of GT and predictions
            if len(sample_metrics) < config.debug_stats_samples:
                gt_min = float(gt_heatmap_np.min())
                gt_max = float(gt_heatmap_np.max())
                gt_p99 = float(np.quantile(gt_heatmap_np, 0.99)) if gt_heatmap_np.size else 0.0
                pred_min = float(prob_masked.min())
                pred_max = float(prob_masked.max())
                pred_p99 = float(np.quantile(prob_masked, 0.99)) if prob_masked.size else 0.0
                print(
                    f"{desc} Stats[{tomo_name}] GT(min/max/p99)=({gt_min:.3f}/{gt_max:.3f}/{gt_p99:.3f}) "
                    f"| Pred(min/max/p99)=({pred_min:.3f}/{pred_max:.3f}/{pred_p99:.3f})"
                )

            # Auxiliary voxel-wise confusion counts at the same peak threshold
            prob_binary = prob_masked > config.peak_threshold
            gt_binary = np.logical_and(gt_heatmap_np > config.peak_threshold, valid_mask)

            tp_vox = int(np.logical_and(prob_binary, gt_binary).sum())
            fp_vox = int(np.logical_and(prob_binary, np.logical_not(gt_binary)).sum())
            fn_vox = int(np.logical_and(np.logical_not(prob_binary), gt_binary).sum())
            total_vox = prob_binary.size
            tn_vox = total_vox - (tp_vox + fp_vox + fn_vox)

            voxel_totals["tp"] += tp_vox
            voxel_totals["fp"] += fp_vox
            voxel_totals["fn"] += fn_vox
            voxel_totals["tn"] += tn_vox

            # Record a rich per-sample summary for later analysis
            sample_metrics.append({
                "tomo_id": tomo_name,
                "f2": det_metrics["f2"],
                "f1": det_metrics["f1"],
                "precision": det_metrics["precision"],
                "recall": det_metrics["recall"],
                "tp": det_metrics["tp"],
                "fp": det_metrics["fp"],
                "fn": det_metrics["fn"],
                "num_pred": det_metrics["num_pred"],
                "num_gt": det_metrics["num_gt"],
                "avg_tp_distance_angstrom": det_metrics["avg_tp_distance_angstrom"],
                "max_tp_distance_angstrom": det_metrics["max_tp_distance_angstrom"],
                "unmatched_gt": det_metrics["unmatched_gt"],
                "unmatched_pred": det_metrics["unmatched_pred"],
                "tp_vox": tp_vox,
                "fp_vox": fp_vox,
                "fn_vox": fn_vox,
                "tn_vox": tn_vox,
                "valid_vox": int(valid_mask.sum()),
            })

            # Update the progress bar postfix with current sample's F2
            pbar_val.set_postfix(f2=f"{det_metrics['f2']:.3f}")

    # Dataset-level means for loss and F2
    mean_f2 = float(np.mean(f2_list)) if f2_list else 0.0
    mean_loss = loss_total / max(1, num_batches)
    mean_loss_components = {
        key: loss_component_sums[key] / max(1, num_batches)
        for key in loss_component_sums
    }

    # Aggregate detection totals to overall metrics
    total_tp = detection_totals["tp"]
    total_fp = detection_totals["fp"]
    total_fn = detection_totals["fn"]
    total_precision = total_tp / (total_tp + total_fp + 1e-8)
    total_recall = total_tp / (total_tp + total_fn + 1e-8)
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall + 1e-8)
    beta = 2.0
    total_f2 = (1 + beta ** 2) * (total_precision * total_recall) / (beta ** 2 * total_precision + total_recall + 1e-8)

    # Summarize threshold sweep (average F2 per threshold and best threshold)
    threshold_sweep_means: Dict[str, float] = {}
    best_threshold = None
    if config.threshold_sweep:
        sweep_strings = []
        for thr in config.threshold_sweep:
            values = threshold_sweeps.get(thr, [])
            mean_val = float(np.mean(values)) if values else 0.0
            threshold_sweep_means[f"{thr:.2f}"] = mean_val
            sweep_strings.append(f"thr={thr:.2f}:F2={mean_val:.3f}")
        if threshold_sweep_means:
            best_threshold = float(max(threshold_sweep_means.items(), key=lambda kv: kv[1])[0])
            sweep_strings.append(f"best={best_threshold:.2f}")
        print(f"{desc} threshold sweep -> " + ", ".join(sweep_strings))

    # Rank samples by F2 to create best/worst lists
    sorted_samples = sorted(sample_metrics, key=lambda x: x["f2"])
    worst_samples = sorted_samples[: config.sample_summary_count]
    best_samples = list(reversed(sorted_samples[-config.sample_summary_count:])) if sorted_samples else []

    # Final result object for downstream consumers (JSON-able)
    results = {
        "mean_f2": mean_f2,
        "f2_list": f2_list,
        "detection_totals": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": total_precision,
            "recall": total_recall,
            "f1": total_f1,
            "f2": total_f2,
        },
        "voxel_totals": voxel_totals,
        "val_loss": mean_loss,
        "val_loss_components": mean_loss_components,
        "sample_metrics": sample_metrics,
        "worst_samples": worst_samples,
        "best_samples": best_samples,
        "threshold_sweep_means": threshold_sweep_means,
        "best_threshold": best_threshold,
    }

    # Restore original training mode if needed
    if was_training:
        model.train()

    return results


# ---------------------------------------------------------------------
# 6) Loss wrapper (avoid circular imports)
# ---------------------------------------------------------------------
def compute_loss_components_stub(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    loss_params: Mapping[str, torch.Tensor | float],
) -> Dict[str, float]:
    """Compute loss components by delegating to the training module.

    This indirection avoids circular imports between evaluation and training
    modules. It calls `train_validate_motor_only.compute_loss_components`
    with the expected signature.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model outputs before sigmoid (shape matches targets).
    targets : torch.Tensor
        Ground-truth heatmap tensor.
    mask : torch.Tensor
        Validity mask tensor (same spatial shape).
    loss_params : Mapping[str, Tensor or float]
        Contains keys: 'pos_weight', 'focal_alpha', 'focal_gamma',
        'dice_weight', 'focal_weight'.

    Returns
    -------
    Dict[str, float]
        Scalar loss components: {'total', 'bce', 'dice', 'focal'}.
    """
    # Deferred import to avoid circular dependencies with the training code
    from train_validate_motor_only import compute_loss_components

    # Compute a dict of tensor losses given the model outputs and params
    losses = compute_loss_components(
        logits,
        targets,
        mask,
        loss_params["pos_weight"],
        loss_params["focal_alpha"],
        loss_params["focal_gamma"],
        loss_params["dice_weight"],
        loss_params["focal_weight"],
    )
    # Convert tensors to Python floats for logging/aggregation
    return {
        "total": float(losses["total"].item()),
        "bce": float(losses["bce"].item()),
        "dice": float(losses["dice"].item()),
        "focal": float(losses["focal"].item()),
    }
