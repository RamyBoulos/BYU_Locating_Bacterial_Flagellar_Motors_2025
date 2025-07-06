import torchio as tio
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json

class Tomo3DDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed 3D tomogram volumes and heatmaps.

    Each sample directory should contain:
        - volume.npy: 3D volume of shape (Z, Y, X)
        - heatmap.npy: 3D heatmap of same shape
        - metadata.json: metadata dictionary

    Parameters
    ----------
    root_dir : str
        Path to the root directory containing per-tomogram subdirectories.
    transform : callable, optional
        Optional transform to apply to each sample.

    Returns
    -------
    dict
        Dictionary with keys:
            - 'volume': torch.Tensor of shape (1, Z, Y, X)
            - 'heatmap': torch.Tensor of shape (1, Z, Y, X)
            - 'tomo_id': str
            - 'meta': dict
    """
    def __init__(self, root_dir: str, transform=None, tomo_ids=None):
        """
        Parameters
        ----------
        root_dir : str
            Path to the root directory containing per-tomogram subdirectories.
        transform : callable, optional
            Optional transform to apply to each sample.
        tomo_ids : list of str, optional
            Optional list of tomogram IDs to include. If None, all subdirectories in root_dir are used.
        """
        self.root_dir = root_dir
        import glob
        if tomo_ids is None:
            self.tomo_ids = sorted([
                os.path.basename(d)
                for d in glob.glob(os.path.join(root_dir, "tomo_*"))
                if os.path.isfile(os.path.join(d, "metadata.json"))
            ])
        else:
            self.tomo_ids = sorted(tomo_ids)
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of tomograms."""
        return len(self.tomo_ids)

    def __getitem__(self, idx: int) -> tio.Subject:
        """
        Load the volume, heatmap, and metadata for the given index, wrapped as a TorchIO Subject.

        Parameters
        ----------
        idx : int
            Index of the tomogram to load.

        Returns
        -------
        torchio.Subject
            A TorchIO Subject with volume and heatmap tensors.
        """
        tomo_id = self.tomo_ids[idx]
        path = os.path.join(self.root_dir, tomo_id)

        volume = np.load(os.path.join(path, "volume.npy"))  # shape (Z, Y, X)
        heatmap = np.load(os.path.join(path, "heatmap.npy"))  # shape (Z, Y, X)

        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        volume_tensor = torch.from_numpy(volume).unsqueeze(0).float()  # (1, Z, Y, X)
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).float()  # (1, Z, Y, X)

        subject = tio.Subject(
            volume=tio.ScalarImage(tensor=volume_tensor),
            heatmap=tio.ScalarImage(tensor=heatmap_tensor)
        )
        subject.tomo_id = tomo_id
        subject.meta = metadata
        subject.num_motors = metadata.get("num_motors", 0)

        return subject