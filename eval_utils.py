"""Shared evaluation utilities for motor detection models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
from tqdm import tqdm


@dataclass(frozen=True)
class EvaluationConfig:
    dist_threshold_angstrom: float
    peak_min_distance_angstrom: float
    peak_threshold: float
    threshold_sweep: Sequence[float]
    debug_stats_samples: int
    sample_summary_count: int


def extract_peaks(
    prob_map: np.ndarray,
    threshold: float,
    voxel_size: np.ndarray | Sequence[float] | None = None,
    min_distance_angstrom: float | None = None,
    seed_window: int = 3,
) -> np.ndarray:
    """Return voxel coordinates of peaks using confidence-aware NMS."""

    mask = prob_map > float(threshold)

    if seed_window and seed_window > 1:
        localmax = (maximum_filter(prob_map, size=seed_window) == prob_map) & mask
        cand_coords = np.argwhere(localmax)
    else:
        cand_coords = np.argwhere(mask)

    if cand_coords.size == 0:
        return np.empty((0, 3), dtype=int)

    candidates = [
        (float(prob_map[z, y, x]), int(z), int(y), int(x))
        for z, y, x in cand_coords
    ]
    candidates.sort(key=lambda t: t[0], reverse=True)

    selected: list[tuple[float, int, int, int]] = []

    if voxel_size is None or min_distance_angstrom is None:
        min_vox = 3
        for cand in candidates:
            _, z, y, x = cand
            keep = True
            for _, sz, sy, sx in selected:
                if max(abs(z - sz), abs(y - sy), abs(x - sx)) < min_vox:
                    keep = False
                    break
            if keep:
                selected.append(cand)
    else:
        v = np.asarray(voxel_size, dtype=np.float32).reshape(-1)[:3]
        v = np.maximum(v, 1e-6)
        min_d2 = float(min_distance_angstrom) ** 2

        for cand in candidates:
            _, z, y, x = cand
            keep = True
            for _, sz, sy, sx in selected:
                dz = (z - sz) * v[0]
                dy = (y - sy) * v[1]
                dx = (x - sx) * v[2]
                if (dz * dz + dy * dy + dx * dx) < min_d2:
                    keep = False
                    break
            if keep:
                selected.append(cand)

    return np.asarray([[z, y, x] for _, z, y, x in selected], dtype=int)


def compute_detection_metrics(
    pred_map: np.ndarray,
    gt_points: np.ndarray,
    voxel_size: np.ndarray,
    *,
    threshold: float,
    min_distance_angstrom: float,
    dist_threshold_angstrom: float,
) -> Dict[str, Any]:
    """Compute detection metrics (precision/recall/F-scores) between predicted peaks and GT."""

    def _flatten_coords(points):
        arr = np.asarray(points)
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        return arr.astype(np.float32)

    voxel_size = np.asarray(voxel_size, dtype=np.float32)

    predicted_pts = extract_peaks(
        pred_map,
        threshold=threshold,
        voxel_size=voxel_size,
        min_distance_angstrom=min_distance_angstrom,
    )
    gt_points = _flatten_coords(gt_points)
    predicted_pts = _flatten_coords(predicted_pts)

    num_pred = predicted_pts.shape[0]
    num_gt = gt_points.shape[0]

    if num_pred == 0 and num_gt == 0:
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
        }

    if num_gt == 0:
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
            "unmatched_pred": predicted_pts.astype(int).tolist()[:5],
        }

    gt_angstrom = gt_points * voxel_size
    pred_angstrom = predicted_pts * voxel_size

    kdtree = cKDTree(gt_angstrom)
    distances, indices = kdtree.query(
        pred_angstrom,
        distance_upper_bound=float(dist_threshold_angstrom),
    )

    candidate_pairs = []
    for pred_idx, (dist, gt_idx) in enumerate(zip(distances, indices)):
        if not np.isfinite(dist):
            continue
        if gt_idx >= len(gt_angstrom):
            continue
        candidate_pairs.append((float(dist), pred_idx, int(gt_idx)))

    candidate_pairs.sort(key=lambda x: x[0])

    used_gt = set()
    tp_indices = []
    tp_distances = []
    for dist, pred_idx, gt_idx in candidate_pairs:
        if gt_idx in used_gt:
            continue
        used_gt.add(gt_idx)
        tp_indices.append(pred_idx)
        tp_distances.append(dist)

    tp = len(tp_indices)
    fp = int(num_pred - tp)
    fn = int(num_gt - len(used_gt))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    beta = 2.0
    f2 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + 1e-8)

    tp_distances_np = np.asarray(tp_distances, dtype=np.float32)
    avg_tp_distance = float(np.mean(tp_distances_np)) if tp > 0 else None
    max_tp_distance = float(np.max(tp_distances_np)) if tp > 0 else None

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
    }


def _log_sample_extremes(best_samples: List[Dict[str, Any]], worst_samples: List[Dict[str, Any]], prefix: str = "[VAL]") -> None:
    """Pretty-print the best and worst validation samples."""
    if not best_samples and not worst_samples:
        return

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


def evaluate(
    model: torch.nn.Module,
    val_loader: Iterable,
    device: torch.device,
    loss_params: Mapping[str, torch.Tensor | float],
    config: EvaluationConfig,
    *,
    desc: str = "[val]",
) -> Dict[str, Any]:
    """Run the validation loop and aggregate detection/voxel metrics."""

    was_training = model.training
    model.eval()

    detection_totals = {"tp": 0, "fp": 0, "fn": 0}
    voxel_totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    f2_list: List[float] = []
    sample_metrics: List[Dict[str, Any]] = []
    loss_total = 0.0
    num_batches = 0
    loss_component_sums: MutableMapping[str, float] = {"bce": 0.0, "dice": 0.0, "focal": 0.0}
    threshold_sweeps: Dict[float, List[float]] = {thr: [] for thr in config.threshold_sweep}

    pbar_val = tqdm(val_loader, desc=desc, leave=False)
    with torch.no_grad():
        for volume, gt_heatmap, mask, tomo_id, gt_coords, voxel_size in pbar_val:
            volume = volume.to(device)
            gt_heatmap_tensor = gt_heatmap.to(device)
            mask_tensor = mask.to(device)
            logits = model(volume)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()

            if isinstance(gt_coords, torch.Tensor):
                gt_coords_np = gt_coords.cpu().numpy()
            else:
                gt_coords_np = np.asarray(gt_coords)

            if isinstance(voxel_size, torch.Tensor):
                voxel_size_np = voxel_size.cpu().numpy()
            else:
                voxel_size_np = np.asarray(voxel_size)

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

            tomo_name = (
                tomo_id[0]
                if isinstance(tomo_id, (list, tuple)) and len(tomo_id) == 1
                else tomo_id
            )

            gt_heatmap_np = gt_heatmap_tensor.squeeze().cpu().numpy()
            mask_np = mask_tensor.squeeze().cpu().numpy()
            valid_mask = mask_np > 0.5
            prob_masked = prob * mask_np

            det_metrics = compute_detection_metrics(
                prob_masked,
                gt_coords_np,
                voxel_size_np,
                threshold=config.peak_threshold,
                min_distance_angstrom=config.peak_min_distance_angstrom,
                dist_threshold_angstrom=config.dist_threshold_angstrom,
            )

            f2_list.append(det_metrics["f2"])
            detection_totals["tp"] += det_metrics["tp"]
            detection_totals["fp"] += det_metrics["fp"]
            detection_totals["fn"] += det_metrics["fn"]

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

            pbar_val.set_postfix(f2=f"{det_metrics['f2']:.3f}")

    mean_f2 = float(np.mean(f2_list)) if f2_list else 0.0
    mean_loss = loss_total / max(1, num_batches)
    mean_loss_components = {
        key: loss_component_sums[key] / max(1, num_batches)
        for key in loss_component_sums
    }

    total_tp = detection_totals["tp"]
    total_fp = detection_totals["fp"]
    total_fn = detection_totals["fn"]
    total_precision = total_tp / (total_tp + total_fp + 1e-8)
    total_recall = total_tp / (total_tp + total_fn + 1e-8)
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall + 1e-8)
    beta = 2.0
    total_f2 = (1 + beta ** 2) * (total_precision * total_recall) / (beta ** 2 * total_precision + total_recall + 1e-8)

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

    sorted_samples = sorted(sample_metrics, key=lambda x: x["f2"])
    worst_samples = sorted_samples[: config.sample_summary_count]
    best_samples = list(
        reversed(sorted_samples[-config.sample_summary_count:])
    ) if sorted_samples else []

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

    if was_training:
        model.train()

    return results


def compute_loss_components_stub(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    loss_params: Mapping[str, torch.Tensor | float],
) -> Dict[str, float]:
    """Wrapper calling training loss helper from the main module.

    We defer import to avoid circular dependencies with the training module.
    """

    from train_validate_motor_only import compute_loss_components

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
    return {
        "total": float(losses["total"].item()),
        "bce": float(losses["bce"].item()),
        "dice": float(losses["dice"].item()),
        "focal": float(losses["focal"].item()),
    }
