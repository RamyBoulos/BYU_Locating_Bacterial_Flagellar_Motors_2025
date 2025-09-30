"""Standalone validation script for motor-only checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from eval_utils import EvaluationConfig, _log_sample_extremes, evaluate
from model import UNet3D
from train_validate_motor_only import (
    BASE_DATA_DIR,
    CHECKPOINT_DIR,
    DEBUG_STATS_SAMPLES,
    DICE_WEIGHT,
    DIST_THRESHOLD_ANGSTROM,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    FOCAL_WEIGHT,
    PEAK_MIN_DISTANCE_ANGSTROM,
    PEAK_THRESHOLD,
    POS_WEIGHT,
    SAMPLE_SUMMARY_COUNT,
    THRESHOLD_SWEEP,
    METRICS_LOG_FIELDS,
    TomoDataset,
    append_metrics_log,
    load_validation_split,
)


def _serialize_record_fields(record: dict[str, object]) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for field in METRICS_LOG_FIELDS:
        value = record.get(field, "")
        if value is None:
            serialized[field] = ""
        elif isinstance(value, (dict, list, tuple)):
            serialized[field] = json.dumps(value, sort_keys=True)
        else:
            serialized[field] = value
    return serialized


def _ensure_eval_run_dir() -> Path:
    root = Path(CHECKPOINT_DIR).resolve().parent / "evals"
    root.mkdir(exist_ok=True)

    idx = 1
    while (root / f"eval{idx}").exists():
        idx += 1

    run_dir = root / f"eval{idx}"
    run_dir.mkdir()
    return run_dir


def _write_eval_outputs(run_dir: Path, record: dict[str, object], sample_metrics: list[dict[str, object]]) -> None:
    metrics_path = run_dir / "metrics.csv"
    row = _serialize_record_fields(record)
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_LOG_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    sample_path = run_dir / "sample_metrics.csv"
    if sample_metrics:
        headers = sorted(sample_metrics[0].keys())
        with open(sample_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for sample in sample_metrics:
                row = {}
                for key, value in sample.items():
                    if isinstance(value, (dict, list, tuple)):
                        row[key] = json.dumps(value, sort_keys=True)
                    else:
                        row[key] = value
                writer.writerow(row)
    else:
        sample_path.touch()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate motor-only model checkpoints")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file. Defaults to latest checkpoint in the dynamic directory.",
    )
    parser.add_argument(
        "--val-sample-count",
        type=int,
        default=None,
        help="Evaluate on the first N validation tomograms.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of workers for the DataLoader.",
    )
    return parser.parse_args()


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return checkpoint


def main() -> None:
    args = _parse_args()

    checkpoint_path = args.checkpoint or os.path.join(CHECKPOINT_DIR, "latest.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    available_tomos = {p.name for p in Path(BASE_DATA_DIR).glob("tomo_*")}
    val_labels_df, val_ids = load_validation_split(available_tomos)

    full_val_count = len(val_ids)
    if args.val_sample_count is not None:
        if args.val_sample_count <= 0:
            raise ValueError("val_sample_count must be a positive integer")
        if args.val_sample_count < full_val_count:
            print(f"[INFO] Restricting validation to {args.val_sample_count} of {full_val_count} tomograms")
        val_ids = val_ids[: args.val_sample_count]
        val_labels_df = val_labels_df[val_labels_df["tomo_id"].isin(val_ids)].copy()

    print(
        f"[INFO] Validation tomos: {len(val_ids)} / {full_val_count} | Label rows: {len(val_labels_df)}"
    )

    val_ds = TomoDataset(val_ids, val_labels_df, augment=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = UNet3D(in_channels=1, out_channels=1, dropout=0.1).to(device)
    _load_checkpoint(model, checkpoint_path, device)
    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")

    pos_weight_tensor = torch.tensor([POS_WEIGHT], device=device)
    loss_params = {
        "pos_weight": pos_weight_tensor,
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "dice_weight": DICE_WEIGHT,
        "focal_weight": FOCAL_WEIGHT,
    }

    eval_config = EvaluationConfig(
        dist_threshold_angstrom=DIST_THRESHOLD_ANGSTROM,
        peak_min_distance_angstrom=PEAK_MIN_DISTANCE_ANGSTROM,
        peak_threshold=PEAK_THRESHOLD,
        threshold_sweep=THRESHOLD_SWEEP,
        debug_stats_samples=DEBUG_STATS_SAMPLES,
        sample_summary_count=SAMPLE_SUMMARY_COUNT,
    )

    eval_stats = evaluate(
        model,
        val_loader,
        device,
        loss_params,
        eval_config,
        desc="[EVAL]",
    )

    detection = eval_stats["detection_totals"]
    voxel = eval_stats["voxel_totals"]
    mean_f2 = eval_stats["mean_f2"]
    val_loss = eval_stats["val_loss"]

    print(
        f"[EVAL] ValLoss={val_loss:.4f} | ValF2={mean_f2:.4f} | ValF1={detection['f1']:.4f} | "
        f"TP={detection['tp']} | FP={detection['fp']} | FN={detection['fn']} | "
        f"VoxTP={voxel['tp']} | VoxTN={voxel['tn']} | VoxFP={voxel['fp']} | VoxFN={voxel['fn']}"
    )

    _log_sample_extremes(
        eval_stats["best_samples"],
        eval_stats["worst_samples"],
        prefix="[EVAL]",
    )

    record = {
        "mode": "eval",
        "checkpoint": checkpoint_path,
        "val_loss": val_loss,
        "val_loss_components": eval_stats["val_loss_components"],
        "mean_f2": mean_f2,
        "detection_totals": detection,
        "voxel_totals": voxel,
        "worst_samples": eval_stats["worst_samples"],
        "best_samples": eval_stats["best_samples"],
        "threshold_sweep_means": eval_stats["threshold_sweep_means"],
        "best_threshold": eval_stats["best_threshold"],
    }

    append_metrics_log(record)

    run_dir = _ensure_eval_run_dir()
    _write_eval_outputs(run_dir, record, eval_stats["sample_metrics"])

    print(f"[EVAL] Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
