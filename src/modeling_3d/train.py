"""
train.py

Script to demonstrate loading and training a 3D U-Net for volumetric tomogram data
with weighted sampling and regression-based heatmap learning.
"""

import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torchio import SubjectsDataset
from torchio.data import GridSampler, Queue, SubjectsLoader, UniformSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import datetime
import torchio as tio

from src.modeling_3d.dataset import Tomo3DDataset
from src.modeling_3d.model import UNet3D
from src.modeling_3d.utils import load_dataset
from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes like batch_size.
    """
    parser = argparse.ArgumentParser(description="Train 3D U-Net on tomogram dataset.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument(
        "--final_activation",
        type=str,
        default="identity",
        choices=["identity", "sigmoid", "softmax", "tanh"],
        help="Final activation function applied to model output"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training: 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        default=[64, 128, 128],
        help="3D patch size (Z Y X) for patch-wise training"
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=None,
        help="Optional override for starting epoch if resuming from checkpoint"
    )
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of dataset to use for validation")
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating weights"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="Number of epochs with no improvement after which training will be stopped"
    )
    return parser.parse_args()

def build_model(final_activation="identity"):
    """
    Construct the UNet3D model with a specified output activation.

    Parameters
    ----------
    final_activation : str, optional
        The final activation function to apply ('identity', 'sigmoid', 'softmax', or 'tanh').

    Returns
    -------
    nn.Module
        A UNet3D instance ready for training.
    """
    model = UNet3D(in_channels=1, out_channels=1, final_activation=final_activation)
    logger.info(f"Model initialized with final_activation='{final_activation}'")
    return model

def train_one_epoch(model, dataloader, device, optimizer, criterion, accumulation_steps, epoch, writer, scaler):
    """
    Train the model for one epoch using mixed precision and gradient accumulation.

    Parameters
    ----------
    model : nn.Module
        The 3D U-Net model.
    dataloader : DataLoader
        DataLoader yielding batches of tomogram volumes and heatmaps.
    device : torch.device
        Target device (e.g., 'cuda' or 'cpu').
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    criterion : nn.Module
        Loss function used for training.
    epoch : int
        Current epoch number.
    writer : SummaryWriter
        TensorBoard writer for logging.
    scaler : GradScaler
        Gradient scaler for mixed precision training.
    """
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}")
    for step, batch in progress_bar:
        inputs = batch["volume"]["data"].to(device, dtype=torch.float32)
        targets = batch["heatmap"]["data"].to(device, dtype=torch.float32)

        if inputs.numel() == 0 or targets.numel() == 0:
            continue

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps  # Normalize loss

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        running_loss += loss.item() * accumulation_steps  # Re-scale loss for logging
        progress_bar.set_postfix(loss=loss.item() * accumulation_steps)

    avg_loss = running_loss / len(dataloader)
    logger.info(f"Train loss: {avg_loss:.4f}")
    logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    if torch.cuda.is_available() and device.type == 'cuda':
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    writer.add_scalar("Loss/train", avg_loss, epoch)
    return avg_loss

def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["volume"]["data"].to(device, dtype=torch.float32)
            targets = batch["heatmap"]["data"].to(device, dtype=torch.float32)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def save_checkpoint(path, epoch, model, optimizer, scaler=None):
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
    }, path)

def main():
    """
    Entry point for training loop.
    Loads dataset, builds model, and runs one training epoch.
    """
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    logger.info(f"Patch size: {tuple(args.patch_size)}")

    raw_dataset = load_dataset(config.PREPROCESSED_DATASET_DIR, config.PREPROCESSING_LOG_CSV)
    subjects = list(raw_dataset)  # raw_dataset is iterable over Subjects

    train_transform = tio.Compose([
        tio.RandomFlip(axes=('LR', 'AP', 'IS'), p=0.5),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=5, p=0.75),
        tio.RandomNoise(std=0.03, p=0.25),
    ])

    train_subjects, val_subjects = train_test_split(subjects, test_size=args.val_split, random_state=args.seed)
    train_dataset_raw = Tomo3DDataset(
        root_dir=config.PREPROCESSED_DATASET_DIR,
        transform=train_transform,
        tomo_ids=[s.tomo_id for s in train_subjects]
    )
    train_subjects = list(train_dataset_raw)
    train_dataset = SubjectsDataset(train_subjects)

    val_dataset_raw = Tomo3DDataset(
        root_dir=config.PREPROCESSED_DATASET_DIR,
        transform=None,
        tomo_ids=[s.tomo_id for s in val_subjects]
    )
    val_subjects = list(val_dataset_raw)
    val_dataset = SubjectsDataset(val_subjects)

    if len(train_dataset) == 0:
        raise ValueError("Filtered training dataset is empty. Check your preprocessing log.")
    if len(val_dataset) == 0:
        raise ValueError("Filtered validation dataset is empty. Check your preprocessing log.")

    queue = Queue(
        subjects_dataset=train_dataset,
        max_length=300,
        samples_per_volume=10,
        sampler=UniformSampler(patch_size=tuple(args.patch_size)),
        num_workers=0
    )

    dataloader = SubjectsLoader(
        queue,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=(args.device == "cuda")
    )

    val_queue = Queue(
        subjects_dataset=val_dataset,
        max_length=100,
        samples_per_volume=5,
        sampler=UniformSampler(patch_size=tuple(args.patch_size)),
        num_workers=0
    )
    val_loader = SubjectsLoader(
        val_queue,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=(args.device == "cuda")
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    model = build_model(final_activation=args.final_activation).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = torch.nn.MSELoss()

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/unet3d_{run_id}")

    start_epoch = 0
    checkpoint_path = args.resume_checkpoint
    if checkpoint_path is None:
        default_latest_path = os.path.join("checkpoints", "unet3d_latest.pth")
        if os.path.isfile(default_latest_path):
            checkpoint_path = default_latest_path
            logger.info(f"No checkpoint provided. Attempting to auto-resume from {checkpoint_path}")

    if checkpoint_path is not None:
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logger.info(f"Resumed training from checkpoint: {checkpoint_path}")
            if args.start_epoch is not None:
                start_epoch = args.start_epoch
                logger.info(f"Overriding start_epoch with user-provided value: {start_epoch}")
            else:
                start_epoch = checkpoint.get("epoch", start_epoch)
        else:
            logger.warning(f"Checkpoint file not found: {checkpoint_path}")

    logger.info(f"Starting training loop with activation '{args.final_activation}'...")

    os.makedirs("checkpoints", exist_ok=True)

    best_loss = float('inf')
    early_stop_patience = args.early_stop_patience
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        avg_loss = train_one_epoch(model, dataloader, device, optimizer, criterion, args.accumulation_steps, epoch, writer, scaler)
        val_loss = evaluate(model, val_loader, device, criterion)
        logger.info(f"Validation loss: {val_loss:.4f}")
        writer.add_scalar("Loss/val", val_loss, epoch)
        scheduler.step(val_loss)

        checkpoint_path = os.path.join("checkpoints", f"unet3d_epoch_{epoch + 1}_val{val_loss:.4f}.pth")
        save_checkpoint(checkpoint_path, epoch, model, optimizer, scaler)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        latest_checkpoint_path = os.path.join("checkpoints", "unet3d_latest.pth")
        save_checkpoint(latest_checkpoint_path, epoch, model, optimizer, scaler)
        logger.info(f"Updated latest checkpoint to {latest_checkpoint_path}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_checkpoint_path = os.path.join("checkpoints", "unet3d_best.pth")
            save_checkpoint(best_checkpoint_path, epoch, model, optimizer, scaler)
            logger.info(f"New best model saved to {best_checkpoint_path}")
            epochs_no_improve = 0  # Reset counter on improvement
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stop_patience:
                logger.info("Early stopping triggered.")
                break

    writer.close()

if __name__ == "__main__":
    main()