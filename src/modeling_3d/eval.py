"""
eval.py

Evaluate a trained 3D U-Net model using peak localization metrics.
"""

import os
import torch
import numpy as np
import argparse
import logging
from tqdm import tqdm
import pandas as pd
import torchio as tio
from src import config
from src.modeling_3d.model import UNet3D
from src.modeling_3d.utils import detect_peaks, average_peak_distance, compute_fbeta_score, load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed evaluation arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate trained 3D U-Net.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for peak detection")
    parser.add_argument("--beta", type=float, default=1.0, help="F-beta score weighting")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tomograms to evaluate")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run evaluation")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 128, 128], help="Patch size (Z Y X)")
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """
    Load the model from a saved checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the model .pth file.
    device : torch.device
        Target device to load the model on.

    Returns
    -------
    torch.nn.Module
        Loaded model in evaluation mode.
    """
    model = UNet3D(in_channels=1, out_channels=1, final_activation="identity")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, subjects, device, threshold=0.5, beta=1.0, patch_size=(64, 128, 128)):
    """
    Evaluate the model on a dataset using localization metrics with patch-wise inference.

    Parameters
    ----------
    model : nn.Module
        The trained UNet3D model.
    subjects : list
        List of torchio.Subject items to evaluate.
    device : torch.device
        Device to run inference on.
    threshold : float
        Threshold for peak detection.
    beta : float
        F-beta weighting for peak localization score.
    patch_size : tuple
        Patch size for sliding window inference.
    """
    distances = []
    fbetas = []

    for subject in tqdm(subjects, desc="Evaluating"):
        sampler = tio.GridSampler(subject, patch_size=patch_size)
        aggregator = tio.GridAggregator(sampler, overlap_mode="average")

        loader = tio.SubjectsLoader(sampler, batch_size=1)

        with torch.no_grad():
            for patch in loader:
                input_tensor = patch["volume"]["data"].to(device, dtype=torch.float32)
                locations = patch[tio.LOCATION]
                pred = model(input_tensor)
                aggregator.add_batch(pred, locations)

        pred_volume = aggregator.get_output_tensor().squeeze().cpu().numpy()
        target_volume = subject["heatmap"]["data"].squeeze().numpy()

        pred_peaks = detect_peaks(pred_volume, threshold=threshold)
        target_peaks = detect_peaks(target_volume, threshold=threshold)

        dist = average_peak_distance(pred_peaks, target_peaks)
        fbeta_score = compute_fbeta_score(pred_peaks, target_peaks, beta=beta)

        distances.append(dist)
        fbetas.append(fbeta_score)

    logger.info(f"Evaluation complete on {len(subjects)} volumes.")
    logger.info(f"Average peak distance: {np.mean(distances):.2f}")
    if beta == 1.0:
        logger.info(f"Average F1 score: {np.mean(fbetas):.3f}")
    else:
        logger.info(f"Average F{beta:.1f}-score: {np.mean(fbetas):.3f}")


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(args.checkpoint, device)

    dataset = load_dataset(config.PREPROCESSED_DATASET_DIR, config.PREPROCESSING_LOG_CSV)
    if args.limit:
        dataset.tomo_ids = dataset.tomo_ids[:args.limit]
        dataset.samples = [s for s in dataset.samples if s['tomo_id'] in dataset.tomo_ids]

    subjects = list(dataset)
    evaluate_model(model, subjects, device, threshold=args.threshold, beta=args.beta, patch_size=tuple(args.patch_size))


if __name__ == "__main__":
    main()