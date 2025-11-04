"""Evaluate multi-sigma motor checkpoints on test tomograms."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from eval_utils import compute_detection_metrics, extract_peaks
from model import UNet3D
from train_validate_motor_only import (
    DIST_THRESHOLD_ANGSTROM,
    PEAK_MIN_DISTANCE_ANGSTROM,
    TomoDataset,
)

# Sigma folders and their metrics logs.
SIGMA_METRIC_FILES: Mapping[str, Path] = {
    "sigma4": Path("/data/horse/ws/rabo074f-team_project/BYU2/checkpoints_dynamic_sigma4/metrics_debug.csv"),
    "sigma8": Path("/data/horse/ws/rabo074f-team_project/BYU2/checkpoints_dynamic_sigma8/metrics_debug.csv"),
    "sigma12": Path("/data/horse/ws/rabo074f-team_project/BYU2/checkpoints_dynamic_sigma12/metrics_debug.csv"),
    "sigma16": Path("/data/horse/ws/rabo074f-team_project/BYU2/checkpoints_dynamic_sigma16/metrics_debug.csv"),
}

THRESHOLDS: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.5)
TEST_LABELS_PATH = Path(
    "/data/horse/ws/rabo074f-team_project/byu-locating-bacterial-flagellar-motors-2025/labels_CSVs/test_labels.csv"
)
OUTPUT_ROOT = Path("outputs")
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation across sigma checkpoints")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to the first N tomograms (defaults to full set)",
    )
    return parser.parse_args()


def load_test_dataframe(csv_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"test labels file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "tomo_id" not in df.columns:
        raise ValueError("CSV must contain a 'tomo_id' column")

    numeric_cols = [
        "Motor axis 0",
        "Motor axis 1",
        "Motor axis 2",
        "Array shape (axis 0)",
        "Array shape (axis 1)",
        "Array shape (axis 2)",
        "Voxel spacing",
        "Number of motors",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Voxel spacing" in df.columns:
        df = df[df["Voxel spacing"].notnull() & (df["Voxel spacing"] > 0)].copy()

    tomo_ids = sorted(df["tomo_id"].unique())
    if not tomo_ids:
        raise ValueError("No tomograms found in test labels file")

    return df, tomo_ids


def find_best_epoch(metrics_csv: Path) -> int:
    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics_csv}")

    best_epoch = None
    best_score = -float("inf")
    with metrics_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epoch_str = row.get("epoch")
            score_str = row.get("mean_f2")
            if not epoch_str or not score_str:
                continue
            try:
                epoch = int(float(epoch_str))
                score = float(score_str)
            except ValueError:
                continue
            if score > best_score:
                best_score = score
                best_epoch = epoch
    if best_epoch is None:
        raise ValueError(f"Could not determine best epoch from {metrics_csv}")
    return best_epoch


def format_threshold_dir(threshold: float) -> str:
    text = f"{threshold:.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"threshold_{text}"


def to_display(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    amin = float(arr.min())
    amax = float(arr.max())
    if amax - amin < EPS:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - amin) / (amax - amin)


def save_slice_figure(
    volume_np: np.ndarray,
    gt_heatmap_np: np.ndarray,
    prob_np: np.ndarray,
    coords_np: np.ndarray,
    pred_pts: np.ndarray,
    metrics: Mapping[str, float],
    threshold: float,
    tomo_id: str,
    out_path: Path,
) -> None:
    idx = np.argmax(gt_heatmap_np.max(axis=(1, 2)))
    #base_mip = to_display(volume_np[idx])
    #gt_mip = to_display(gt_heatmap_np[idx])
    #pred_mip = to_display(prob_np[idx])
    base_slice = volume_np[idx]
    gt_slice = gt_heatmap_np[idx]
    prob_slice = prob_np[idx]
    slice_idx = int(idx)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(base_slice, cmap="gray")
    axes[0].set_title("Input slice")
    axes[0].axis("off")

    axes[1].imshow(base_slice, cmap="gray", alpha=.5)
    axes[1].imshow(gt_slice, cmap="viridis", alpha=0.8, clim=(0,1))
    if coords_np.size:
        coords_plot = coords_np.astype(np.float32, copy=False)
        slice_mask = np.abs(coords_plot[:, 0] - slice_idx) < 0.5
        coords_slice = coords_plot[slice_mask]
        if coords_slice.size:
            axes[1].scatter(
                coords_slice[:, 2],
                coords_slice[:, 1],
                s=40,
                facecolors="none",
                edgecolors="lime",
                linewidths=1.5,
            )
    axes[1].set_title("GT overlay")
    axes[1].axis("off")

    axes[2].imshow(base_slice, cmap="gray", alpha=.5)
    axes[2].imshow(prob_slice, cmap="hot", alpha=0.8, clim=(0,1))
    if pred_pts.size:
        pred_pts_plot = pred_pts.astype(np.float32, copy=False)
        pred_slice_mask = np.abs(pred_pts_plot[:, 0] - slice_idx) < 10
        pred_slice = pred_pts_plot[pred_slice_mask]
        if pred_slice.size:
            axes[2].scatter(
                pred_slice[:, 2],
                pred_slice[:, 1],
                s=40,
                facecolors="none",
                edgecolors="red",
                linewidths=1.0,
            )
    axes[2].set_title(f"Pred overlay (thr={threshold:.2f})")
    axes[2].axis("off")

    fig.suptitle(
        f"{tomo_id} | GT={metrics['num_gt']} TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} | "
        f"F1={metrics['f1']:.3f} F2={metrics['f2']:.3f}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate_sigma(
    sigma_name: str,
    checkpoint_path: Path,
    dataset: TomoDataset,
    thresholds: Iterable[float],
    output_root: Path,
    max_visuals: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, out_channels=1, dropout=0.1).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    sigma_root = output_root / sigma_name
    sigma_root.mkdir(parents=True, exist_ok=True)

    max_visuals = min(max_visuals, len(dataset))

    thresholds = list(thresholds)
    per_threshold_rows: Dict[float, List[Dict[str, float]]] = {thr: [] for thr in thresholds}
    per_threshold_totals: Dict[float, Dict[str, float]] = {
        thr: {"tp": 0, "fp": 0, "fn": 0, "num_pred": 0, "num_gt": 0} for thr in thresholds
    }
    saved_images: Dict[float, int] = {thr: 0 for thr in thresholds}

    for idx in range(len(dataset)):
        volume, gt_heatmap, mask, tomo_id, coords, eff_voxel_size = dataset[idx]
        tomo_name = str(tomo_id)

        volume_np = volume.squeeze(0).cpu().numpy()
        gt_heatmap_np = gt_heatmap.squeeze(0).cpu().numpy()
        mask_np = mask.squeeze(0).cpu().numpy()
        coords_np = np.asarray(coords)
        if coords_np.size == 0:
            coords_np = coords_np.reshape(0, 3)
        elif coords_np.ndim == 1:
            coords_np = coords_np.reshape(1, -1)
        voxel_size_np = np.asarray(eff_voxel_size)

        volume_batch = volume.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(volume_batch)
            prob_np = torch.sigmoid(logits).squeeze().cpu().numpy()

        prob_np *= mask_np

        for thr in thresholds:
            metrics = compute_detection_metrics(
                prob_np,
                coords_np,
                voxel_size_np,
                threshold=thr,
                min_distance_angstrom=PEAK_MIN_DISTANCE_ANGSTROM,
                dist_threshold_angstrom=DIST_THRESHOLD_ANGSTROM,
            )
            row = {
                "tomo_id": tomo_name,
                "tp": int(metrics["tp"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "f2": float(metrics["f2"]),
                "num_pred": int(metrics["num_pred"]),
                "num_gt": int(metrics["num_gt"]),
                "pred_coords_vox": json.dumps(metrics.get("pred_coords_vox", [])),
                "gt_coords_vox": json.dumps(coords_np.astype(int).tolist()),
            }
            per_threshold_rows[thr].append(row)

            totals = per_threshold_totals[thr]
            totals["tp"] += row["tp"]
            totals["fp"] += row["fp"]
            totals["fn"] += row["fn"]
            totals["num_pred"] += row["num_pred"]
            totals["num_gt"] += row["num_gt"]

            threshold_dir = sigma_root / format_threshold_dir(thr)
            images_dir = threshold_dir / "images"
            if saved_images[thr] < max_visuals:
                threshold_dir.mkdir(parents=True, exist_ok=True)
                images_dir.mkdir(parents=True, exist_ok=True)
                pred_pts = extract_peaks(
                    prob_np,
                    threshold=thr,
                    voxel_size=voxel_size_np,
                    min_distance_angstrom=PEAK_MIN_DISTANCE_ANGSTROM,
                )
                image_path = images_dir / f"{saved_images[thr]:03d}_{tomo_name}.png"
                save_slice_figure(
                    volume_np,
                    gt_heatmap_np,
                    prob_np,
                    coords_np,
                    pred_pts,
                    metrics,
                    thr,
                    tomo_name,
                    image_path,
                )
                saved_images[thr] += 1

    for thr in thresholds:
        threshold_dir = sigma_root / format_threshold_dir(thr)
        threshold_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = threshold_dir / "metrics.csv"
        rows = per_threshold_rows[thr]
        totals = per_threshold_totals[thr]

        tp = totals["tp"]
        fp = totals["fp"]
        fn = totals["fn"]
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        f1 = 2 * precision * recall / (precision + recall + EPS)
        beta = 2.0
        f2 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + EPS)
        total_row = {
            "tomo_id": "OVERALL",
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
            "num_pred": totals["num_pred"],
            "num_gt": totals["num_gt"],
            "pred_coords_vox": "",
            "gt_coords_vox": "",
        }

        headers = [
            "tomo_id",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "f2",
            "num_pred",
            "num_gt",
            "pred_coords_vox",
            "gt_coords_vox",
        ]
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            writer.writerow(total_row)


def main() -> None:
    args = parse_args()
    labels_df, all_tomo_ids = load_test_dataframe(TEST_LABELS_PATH)

    selected_tomo_ids = all_tomo_ids
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        selected_tomo_ids = selected_tomo_ids[: args.max_samples]

    dataset = TomoDataset(selected_tomo_ids, labels_df, augment=False)

    for sigma_name, metrics_csv in SIGMA_METRIC_FILES.items():
        best_epoch = find_best_epoch(metrics_csv)
        checkpoint_path = metrics_csv.parent / f"best_epoch{best_epoch}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if args.max_samples is None:
            max_visuals = 10
        else:
            max_visuals = args.max_samples

        evaluate_sigma(
            sigma_name,
            checkpoint_path,
            dataset,
            THRESHOLDS,
            OUTPUT_ROOT,
            max_visuals,
        )


if __name__ == "__main__":
    main()
