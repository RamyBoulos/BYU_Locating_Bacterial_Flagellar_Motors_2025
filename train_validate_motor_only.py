# train_validate_motor_only.py
import os
import argparse
import csv
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom, gaussian_filter
from skimage.transform import resize
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from config import BASE_DATA_DIR, BATCH_SIZE, NUM_EPOCHS
from eval_utils import EvaluationConfig, _log_sample_extremes, evaluate
from model import UNet3D  # supports dropout


# ----------------------------
# Checkpoint directory
# ----------------------------
CHECKPOINT_DIR = "/data/horse/ws/rabo074f-team_project/BYU2/checkpoints_dynamic_sigma12"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ----------------------------
# Constants
# ----------------------------
DIST_THRESHOLD_ANGSTROM = 1000.0
PEAK_MIN_DISTANCE_ANGSTROM = 300.0  # Approximate physical spacing between distinct motors
SIGMA_Z_VOXELS = 12  # Fixed sigma along Z; XY derived from voxel spacing
PEAK_THRESHOLD = 0.2  # Shared threshold for detection and voxel metrics changed from o.5 in the first 100 epochs and 0.1 from 101 to 150
THRESHOLD_SWEEP = (0.05, 0.1, 0.2, 0.3, 0.5)
DEBUG_STATS_SAMPLES = 3  # How many validation samples to log detailed stats
SAMPLE_SUMMARY_COUNT = 5  # Number of best/worst samples to record per epoch
METRICS_LOG_PATH = os.path.join(CHECKPOINT_DIR, "metrics_debug.csv")
METRICS_LOG_FIELDS = [
    "mode",
    "epoch",
    "checkpoint",
    "train_loss",
    "train_loss_components",
    "val_loss",
    "val_loss_components",
    "mean_f2",
    "detection_totals",
    "voxel_totals",
    "worst_samples",
    "best_samples",
    "threshold_sweep_means",
    "best_threshold",
]
POS_WEIGHT = 50.0  # Emphasize sparse positives in BCE term without overpowering negatives
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
DICE_WEIGHT = 0.5
FOCAL_WEIGHT = 0.25
LR_SCHEDULER_PATIENCE = 5 
LR_SCHEDULER_WARMUP_EPOCHS = 5
FINAL_GRID_SIZE = 128  # XY resolution after preprocessing pipeline
LABEL_SPLITS_ROOT = Path(BASE_DATA_DIR).resolve().parent / "labels_CSVs"
TRAIN_LABELS_CSV = LABEL_SPLITS_ROOT / "train_labels.csv"
VAL_LABELS_CSV_CANDIDATES = [LABEL_SPLITS_ROOT / "val_labels.csv"]


def _compute_anisotropic_sigma(voxel_size, sigma_z_vox):
    """Derive per-axis Gaussian sigma in voxels using constant physical blur."""
    if voxel_size is None:
        return float(sigma_z_vox)

    voxel_size = np.asarray(voxel_size, dtype=np.float32).reshape(-1)
    if voxel_size.size < 3:
        raise ValueError(f"voxel_size must contain at least 3 elements, got {voxel_size}")
    if voxel_size.size > 3:
        voxel_size = voxel_size[:3]
    sigma_z = float(sigma_z_vox)
    sigma_phys = sigma_z * float(voxel_size[0])
    eps = 1e-8
    sigma_y = sigma_phys / max(float(voxel_size[1]), eps)
    sigma_x = sigma_phys / max(float(voxel_size[2]), eps)

    # Early training benefits from broader XY support to avoid penalizing small misalignments
    sigma_y = max(sigma_y, 2.0)
    sigma_x = max(sigma_x, 2.0)

    return (sigma_z, sigma_y, sigma_x)


def save_checkpoint(obj, filename):
    torch.save(obj, os.path.join(CHECKPOINT_DIR, filename))


def append_metrics_log(record):
    serialized = {}
    for field in METRICS_LOG_FIELDS:
        value = record.get(field, "")
        if value is None:
            serialized[field] = ""
        elif isinstance(value, (dict, list, tuple)):
            serialized[field] = json.dumps(value, sort_keys=True)
        else:
            serialized[field] = value

    file_exists = os.path.exists(METRICS_LOG_PATH)
    needs_header = not file_exists or os.path.getsize(METRICS_LOG_PATH) == 0

    with open(METRICS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_LOG_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(serialized)


def load_motor_label_split(csv_paths, available_tomos):
    """Load a split CSV and keep only tomos with valid motor annotations."""
    if isinstance(csv_paths, (str, Path)):
        candidate_paths = [Path(csv_paths)]
    else:
        candidate_paths = [Path(p) for p in csv_paths]

    csv_path = None
    for path in candidate_paths:
        if path.exists():
            csv_path = path
            break

    if csv_path is None:
        csv_path = candidate_paths[0]
        available = []
        split_root = csv_path.parent
        if split_root.exists():
            available = sorted(p.name for p in split_root.glob("*.csv"))
        raise FileNotFoundError(
            f"Label split not found: {csv_path}. Available splits: {available}"
        )

    labels_df = pd.read_csv(csv_path)
    numeric_cols = [
        "Motor axis 0",
        "Motor axis 1",
        "Motor axis 2",
        "Array shape (axis 0)",
        "Array shape (axis 1)",
        "Array shape (axis 2)",
        "Voxel spacing",
    ]
    for col in numeric_cols:
        if col in labels_df.columns:
            labels_df[col] = pd.to_numeric(labels_df[col], errors="coerce")
    coord_cols = ["Motor axis 0", "Motor axis 1", "Motor axis 2"]
    valid_row_mask = ~(labels_df[coord_cols] == -1).any(axis=1)
    if "Voxel spacing" in labels_df.columns:
        valid_row_mask &= labels_df["Voxel spacing"].notnull() & (labels_df["Voxel spacing"] > 0)
    motor_labels_df = labels_df[valid_row_mask].copy()

    tomo_ids = [tomo_id for tomo_id in sorted(motor_labels_df["tomo_id"].unique()) if tomo_id in available_tomos]
    if len(tomo_ids) < motor_labels_df["tomo_id"].nunique():
        missing = set(motor_labels_df["tomo_id"].unique()) - set(tomo_ids)
        preview = sorted(missing)
        suffix = "..." if len(preview) > 3 else ""
        print(
            f"[WARN] Skipping {len(missing)} tomos listed in {csv_path.name} with no matching volume: {preview[:3]}{suffix}"
        )

    motor_labels_df = motor_labels_df[motor_labels_df["tomo_id"].isin(tomo_ids)].copy()

    return motor_labels_df, tomo_ids


def compute_loss_components(logits, targets, mask, pos_weight, focal_alpha, focal_gamma,
                            dice_weight, focal_weight, smooth=1e-6, eps=1e-6):
    mask = mask.to(logits.device, logits.dtype)
    valid_count = mask.sum().clamp_min(eps)
    pos_weight = pos_weight.to(logits.device, logits.dtype)

    bce_unreduced = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    bce_loss = (bce_unreduced * mask).sum() / valid_count

    probs = torch.sigmoid(logits)
    probs_masked = probs * mask
    targets_masked = targets * mask
    intersection = (probs_masked * targets_masked).sum()
    cardinality = probs_masked.sum() + targets_masked.sum()
    dice_loss = 1.0 - (2.0 * intersection + smooth) / (cardinality + smooth)

    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_factor = targets * focal_alpha + (1.0 - targets) * (1.0 - focal_alpha)
    focal_unreduced = alpha_factor * (1.0 - pt).pow(focal_gamma) * bce_unreduced
    focal_loss = (focal_unreduced * mask).sum() / valid_count

    total_loss = bce_loss + dice_weight * dice_loss + focal_weight * focal_loss

    return {
        "total": total_loss,
        "bce": bce_loss,
        "dice": dice_loss,
        "focal": focal_loss,
    }


# ----------------------------
# Preprocessing utilities
# ----------------------------
def percentile_normalize(volume, low=0.5, high=99.5):
    vmin, vmax = np.percentile(volume, [low, high])
    volume = np.clip(volume, vmin, vmax)
    volume = 2.0 * (volume - vmin) / (vmax - vmin + 1e-8) - 1.0
    return volume.astype(np.float32)


def downsample_xy(volume, factor=8):
    zoom_factors = (1.0, 1.0 / factor, 1.0 / factor)
    return zoom(volume, zoom_factors, order=1)


def generate_gaussian_heatmap(coords_np, shape, sigma=None, voxel_size=None, sigma_z=SIGMA_Z_VOXELS):
    if sigma is None:
        sigma = _compute_anisotropic_sigma(voxel_size, sigma_z)
    heatmap = np.zeros(shape, dtype=np.float32)
    if coords_np.size > 0:
        Z, Y, X = shape
        for z, y, x in coords_np:
            if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
                heatmap[int(z), int(y), int(x)] = 1.0
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    max_val = float(heatmap.max())
    if max_val > 0.0:
        heatmap /= max_val
    return np.clip(heatmap.astype(np.float32), 0, 1.0)


# ----------------------------
# Augmentation utilities
# ----------------------------
def random_flip_rotate(volume, heatmap, mask):
    if np.random.rand() < 0.5:
        volume = np.flip(volume, axis=1).copy()
        heatmap = np.flip(heatmap, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()
    if np.random.rand() < 0.5:
        volume = np.flip(volume, axis=2).copy()
        heatmap = np.flip(heatmap, axis=2).copy()
        mask = np.flip(mask, axis=2).copy()
    k = np.random.randint(0, 4)
    if k > 0:
        volume = np.rot90(volume, k, axes=(1, 2)).copy()
        heatmap = np.rot90(heatmap, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(1, 2)).copy()
    return volume, heatmap, mask


def random_intensity_jitter(volume):
    if np.random.rand() < 0.5:
        scale = 0.9 + 0.2 * np.random.rand()
        shift = 0.1 * (2 * np.random.rand() - 1)
        volume = volume * scale + shift
        volume = np.clip(volume, -1, 1)
    return volume


def random_z_shift(volume, heatmap, mask, max_shift=10):
    if np.random.rand() < 0.5:
        shift = np.random.randint(-max_shift, max_shift + 1)
        volume = np.roll(volume, shift, axis=0)
        heatmap = np.roll(heatmap, shift, axis=0)
        mask = np.roll(mask, shift, axis=0)
        if shift > 0:
            volume[:shift] = 0
            heatmap[:shift] = 0
            mask[:shift] = 0
        elif shift < 0:
            volume[shift:] = 0
            heatmap[shift:] = 0
            mask[shift:] = 0
    return volume, heatmap, mask


# ----------------------------
# Dataset
# ----------------------------
class TomoDataset(Dataset):
    def __init__(self, tomo_ids, labels_df, augment=False):
        self.tomo_ids = list(tomo_ids)
        self.labels_df = labels_df
        self.augment = augment

    def __len__(self):
        return len(self.tomo_ids)

    def __getitem__(self, idx):
        tomo_id = self.tomo_ids[idx]
        tomo_dir = os.path.join(BASE_DATA_DIR, tomo_id)

        # --- Load slices ---
        slice_paths = sorted(Path(tomo_dir).glob("slice_*.jpg"))
        if not slice_paths:
            raise RuntimeError(f"No slices found in {tomo_dir}")
        vol_slices = [np.array(Image.open(p).convert("L"), dtype=np.float32) for p in slice_paths]
        volume = np.stack(vol_slices, axis=0)

        # --- Normalize ---
        volume = percentile_normalize(volume, low=0.5, high=99.5)

        # --- Store shape before processing ---
        orig_z, orig_y, orig_x = volume.shape

        # --- Downsample XY ---
        volume = downsample_xy(volume, factor=8)
        mask = np.ones_like(volume, dtype=np.float32)
        ds_z, ds_y, ds_x = volume.shape

        # --- Pad or crop Z to target depth ---
        if ds_z >= FINAL_GRID_SIZE:
            z_start = (ds_z - FINAL_GRID_SIZE) // 2
            volume = volume[z_start:z_start+FINAL_GRID_SIZE, :, :]
            mask = mask[z_start:z_start+FINAL_GRID_SIZE, :, :]
            z_offset = -z_start
        else:
            pad_before = (FINAL_GRID_SIZE - ds_z) // 2
            pad_after = FINAL_GRID_SIZE - ds_z - pad_before
            volume = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode="constant")
            mask = np.pad(mask, ((pad_before, pad_after), (0, 0), (0, 0)), mode="constant")
            z_offset = pad_before
        ds_z = FINAL_GRID_SIZE

        # --- Resize XY to FINAL_GRID_SIZE x FINAL_GRID_SIZE ---
        volume = resize(volume, (ds_z, FINAL_GRID_SIZE, FINAL_GRID_SIZE), order=1,
                        preserve_range=True, anti_aliasing=True).astype(np.float32)
        mask = resize(mask, (ds_z, FINAL_GRID_SIZE, FINAL_GRID_SIZE), order=0,
                      preserve_range=True, anti_aliasing=False).astype(np.float32)
        mask = (mask > 0.5).astype(np.float32)

        # --- Get GT coords ---
        df_tomo = self.labels_df[self.labels_df["tomo_id"] == tomo_id]
        coords = df_tomo[["Motor axis 0", "Motor axis 1", "Motor axis 2"]].values if not df_tomo.empty else np.empty((0, 3), dtype=int)
        if coords.size > 0:
            coords = coords[~np.any(coords == -1, axis=1)]
        coords = coords.astype(np.float32)

        if df_tomo.empty:
            raise RuntimeError(f"No label metadata found for {tomo_id}")
        voxel_spacing_angstrom = float(df_tomo.iloc[0]["Voxel spacing"])
        if not np.isfinite(voxel_spacing_angstrom) or voxel_spacing_angstrom <= 0:
            raise ValueError(f"Invalid voxel spacing {voxel_spacing_angstrom} for {tomo_id}")
        voxel_spacing_angstrom = max(float(voxel_spacing_angstrom), 1e-6)

        # --- Transform coords step by step ---
        if coords.size > 0:
            # 1) Downsample XY
            coords[:, 1] /= 8.0
            coords[:, 2] /= 8.0

            # 2) Adjust Z for crop/pad
            coords[:, 0] += z_offset

            # 3) Resize XY to FINAL_GRID_SIZE
            coords[:, 1] *= (float(FINAL_GRID_SIZE) / ds_y)
            coords[:, 2] *= (float(FINAL_GRID_SIZE) / ds_x)

            # Clip
            coords[:, 0] = np.clip(coords[:, 0], 0, ds_z - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, FINAL_GRID_SIZE - 1)
            coords[:, 2] = np.clip(coords[:, 2], 0, FINAL_GRID_SIZE - 1)

        coords = np.rint(coords).astype(int)

        # --- Compute effective voxel size after transforms ---
        eff_voxel_size = np.array([
            voxel_spacing_angstrom,
            voxel_spacing_angstrom * (orig_y / float(FINAL_GRID_SIZE)),
            voxel_spacing_angstrom * (orig_x / float(FINAL_GRID_SIZE)),
        ], dtype=np.float32)

        # --- Heatmap ---
        heatmap = generate_gaussian_heatmap(coords, volume.shape, voxel_size=eff_voxel_size)

        # --- Augment ---
        if self.augment:
            volume, heatmap, mask = random_flip_rotate(volume, heatmap, mask)
            volume = random_intensity_jitter(volume)
            volume, heatmap, mask = random_z_shift(volume, heatmap, mask)

        # --- Torch ---
        volume = torch.from_numpy(volume).unsqueeze(0)
        heatmap = torch.from_numpy(heatmap).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return volume, heatmap, mask, tomo_id, coords, eff_voxel_size


# ----------------------------
# Validation utilities
# ----------------------------
def load_validation_split(available_tomos):
    val_candidate_paths = [Path(p) for p in VAL_LABELS_CSV_CANDIDATES]
    val_csv_path = None
    for path in val_candidate_paths:
        if path.exists():
            val_csv_path = path
            break
    if val_csv_path is None:
        sample_path = val_candidate_paths[0]
        available = []
        split_root = sample_path.parent
        if split_root.exists():
            available = sorted(p.name for p in split_root.glob("*.csv"))
        raise FileNotFoundError(
            f"Label split not found: {sample_path}. Available splits: {available}"
        )

    val_labels_df = pd.read_csv(val_csv_path)
    numeric_cols = [
        "Motor axis 0",
        "Motor axis 1",
        "Motor axis 2",
        "Array shape (axis 0)",
        "Array shape (axis 1)",
        "Array shape (axis 2)",
        "Voxel spacing",
    ]
    for col in numeric_cols:
        if col in val_labels_df.columns:
            val_labels_df[col] = pd.to_numeric(val_labels_df[col], errors="coerce")
    if "Voxel spacing" in val_labels_df.columns:
        val_labels_df = val_labels_df[
            val_labels_df["Voxel spacing"].notnull()
            & (val_labels_df["Voxel spacing"] > 0)
        ].copy()

    val_labels_df = val_labels_df[val_labels_df["tomo_id"].isin(available_tomos)].copy()
    val_ids = sorted(val_labels_df["tomo_id"].unique())
    if not val_ids:
        raise ValueError("No validation tomograms found in val_labels.csv")

    return val_labels_df, val_ids


def train(val_sample_count=None):
    available_tomos = {p.name for p in Path(BASE_DATA_DIR).glob("tomo_*")}

    train_labels_df, train_ids = load_motor_label_split(
        TRAIN_LABELS_CSV,
        available_tomos,
    )
    if not train_ids:
        raise ValueError("No motor-positive tomograms found in train_labels.csv")

    val_labels_df, val_ids = load_validation_split(available_tomos)

    full_val_count = len(val_ids)
    if val_sample_count is not None:
        if val_sample_count <= 0:
            raise ValueError("val_sample_count must be a positive integer")
        if val_sample_count < full_val_count:
            print(f"[INFO] Restricting validation to {val_sample_count} of {full_val_count} tomograms")
        val_ids = val_ids[:val_sample_count]
        val_labels_df = val_labels_df[val_labels_df["tomo_id"].isin(val_ids)].copy()

    print(
        f"[INFO] Training tomos (motors only): {len(train_ids)} | Motor annotations: {len(train_labels_df)}"
    )
    val_label_count = len(val_labels_df)
    if val_sample_count is None or val_sample_count >= full_val_count:
        print(
            f"[INFO] Validation tomos: {len(val_ids)} | Label rows: {val_label_count}"
        )
    else:
        print(
            f"[INFO] Validation tomos: {len(val_ids)} / {full_val_count} used | "
            f"Label rows: {val_label_count}"
        )

    val_ds = TomoDataset(val_ids, val_labels_df, augment=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    train_ds = TomoDataset(train_ids, train_labels_df, augment=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    model = UNet3D(in_channels=1, out_channels=1, dropout=0.1).to(device)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=LR_SCHEDULER_PATIENCE,
    )

    resume_path = os.path.join(CHECKPOINT_DIR, "resume.pt")
    start_epoch, best_f2 = 1, -1.0

    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_f2 = ckpt["best_f2"]
        print(f"[INFO] Resumed from epoch {ckpt['epoch']} with best F2={best_f2:.4f}")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # ---- Training ----
        model.train()
        running_loss = 0.0
        running_components = {"bce": 0.0, "dice": 0.0, "focal": 0.0}
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch:03d}][train]", leave=False)
        for volume, heatmap, mask, _, _, _ in pbar:
            volume = volume.to(device)
            heatmap = heatmap.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            logits = model(volume)
            losses = compute_loss_components(
                logits,
                heatmap,
                mask,
                loss_params["pos_weight"],
                loss_params["focal_alpha"],
                loss_params["focal_gamma"],
                loss_params["dice_weight"],
                loss_params["focal_weight"],
            )
            loss = losses["total"]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_components["bce"] += float(losses["bce"].item())
            running_components["dice"] += float(losses["dice"].item())
            running_components["focal"] += float(losses["focal"].item())
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                bce=f"{losses['bce'].item():.4f}",
                dice=f"{losses['dice'].item():.4f}",
                focal=f"{losses['focal'].item():.4f}",
            )

        train_loss = running_loss / max(1, len(train_loader))
        train_loss_components = {
            key: running_components[key] / max(1, len(train_loader))
            for key in running_components
        }

        # ---- Validation ----
        eval_stats = evaluate(
            model,
            val_loader,
            device,
            loss_params,
            eval_config,
            desc=f"[Epoch {epoch:03d}][val]",
        )
        mean_f2 = eval_stats["mean_f2"]
        detection = eval_stats["detection_totals"]
        voxel = eval_stats["voxel_totals"]
        val_loss = eval_stats["val_loss"]
        print(
            f"[Epoch {epoch:03d}] TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | ValF2={mean_f2:.4f} | "
            f"ValF1={detection['f1']:.4f} | TP={detection['tp']} | FP={detection['fp']} | FN={detection['fn']} | "
            f"VoxTP={voxel['tp']} | VoxTN={voxel['tn']} | VoxFP={voxel['fp']} | VoxFN={voxel['fn']}"
        )
        _log_sample_extremes(eval_stats["best_samples"], eval_stats["worst_samples"], prefix=f"[Epoch {epoch:03d}]")

        append_metrics_log({
            "mode": "train",
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_components": train_loss_components,
            "val_loss": val_loss,
            "val_loss_components": eval_stats["val_loss_components"],
            "mean_f2": mean_f2,
            "detection_totals": eval_stats["detection_totals"],
            "voxel_totals": eval_stats["voxel_totals"],
            "worst_samples": eval_stats["worst_samples"],
            "best_samples": eval_stats["best_samples"],
            "threshold_sweep_means": eval_stats["threshold_sweep_means"],
            "best_threshold": eval_stats["best_threshold"],
        })

        if epoch >= LR_SCHEDULER_WARMUP_EPOCHS:  # avoid early LR drops while detections emerge
            scheduler.step(mean_f2)

        # ---- Save best ----
        if mean_f2 > best_f2:
            best_f2 = mean_f2
            model_state = model.state_dict()
            best_filename = f"best_epoch{epoch}.pt"
            save_checkpoint(model_state, best_filename)
            save_checkpoint(model_state, "latest.pt")
            primary_ckpt_path = os.path.join(CHECKPOINT_DIR, best_filename)
            print(f"[INFO] New best model saved @ {primary_ckpt_path} (F2={best_f2:.4f})")

        # ---- Save resume checkpoint ----
        resume_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_f2": best_f2,
        }
        save_checkpoint(resume_payload, "resume.pt")

    print("[DONE] Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train motor-only model")
    parser.add_argument(
        "--val-sample-count",
        type=int,
        default=None,
        help="Limit validation to the first N tomograms in the split.",
    )
    args = parser.parse_args()

    train(
        val_sample_count=args.val_sample_count,
    )
