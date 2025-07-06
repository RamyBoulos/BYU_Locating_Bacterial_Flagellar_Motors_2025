"""
train.py

Script to demonstrate loading and training a 3D U-Net for volumetric tomogram data
with weighted sampling and regression-based heatmap learning.
"""

import torch
from torch.utils.data import WeightedRandomSampler
from torchio import SubjectsDataset
from torchio.data import GridSampler, Queue
from torchio.data import SubjectsLoader
from src.modeling_3d.dataset import Tomo3DDataset
from src.modeling_3d.model import UNet3D
from src.modeling_3d.utils import load_dataset
from src import config
import numpy as np
import pandas as pd
import os
import argparse
import logging

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

def train_one_epoch(model, dataloader, device, optimizer, criterion):
    """
    Train the model for one epoch on the provided DataLoader.

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
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Move inputs and targets to the target device (e.g., GPU or CPU)
        inputs = batch["volume"]["data"].to(device, dtype=torch.float32)
        targets = batch["heatmap"]["data"].to(device, dtype=torch.float32)

        optimizer.zero_grad()         # Reset gradients
        outputs = model(inputs)       # Forward pass
        loss = criterion(outputs, targets)  # Compute loss between prediction and target
        loss.backward()              # Backpropagation
        optimizer.step()             # Update model parameters
        torch.cuda.empty_cache()  # Helps reduce CUDA fragmentation

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    logger.info(f"Train loss: {avg_loss:.4f}")

def main():
    """
    Entry point for training loop.
    Loads dataset, builds model, and runs one training epoch.
    """
    args = parse_args()

    raw_dataset = load_dataset(config.PREPROCESSED_DATASET_DIR, config.PREPROCESSING_LOG_CSV)
    subjects = list(raw_dataset)  # raw_dataset is iterable over Subjects
    dataset = SubjectsDataset(subjects)

    if len(dataset) == 0:
        raise ValueError("Filtered dataset is empty. Check your preprocessing log.")

    sampler = GridSampler(subject=dataset[0], patch_size=tuple(args.patch_size))
    queue = Queue(
        subjects_dataset=dataset,
        sampler=sampler,
        max_length=300,
        samples_per_volume=10,
        num_workers=0
    )

    dataloader = SubjectsLoader(
        queue,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=(args.device == "cuda")
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = build_model(final_activation=args.final_activation).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    logger.info(f"Starting training loop with activation '{args.final_activation}'...")

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        train_one_epoch(model, dataloader, device, optimizer, criterion)

        checkpoint_path = os.path.join("checkpoints", f"unet3d_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()