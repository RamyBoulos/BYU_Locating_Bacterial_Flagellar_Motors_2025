import os
import numpy as np
import json
import tempfile
import pytest
from preprocessing_3d.dataset_builder import preprocess_tomogram
from preprocessing_3d.label_processing import group_labels_by_tomo, filter_valid_labels, load_labels
from src import config


@pytest.mark.slow
def test_preprocess_tomogram_creates_outputs():
    # Use one known tomo ID from your sampled dataset
    tomo_id = config.EXAMPLE_TOMO_ID

    # Load and prepare label dict
    df = load_labels(config.SAMPLED_LABELS_PATH)
    df_valid = filter_valid_labels(df)
    label_dict = group_labels_by_tomo(df_valid)

    with tempfile.TemporaryDirectory() as tmpdir:
        preprocess_tomogram(tomo_id, label_dict, output_dir=tmpdir)

        tomo_dir = os.path.join(tmpdir, tomo_id)
        volume_path = os.path.join(tomo_dir, "volume.npy")
        heatmap_path = os.path.join(tomo_dir, "heatmap.npy")
        metadata_path = os.path.join(tomo_dir, "metadata.json")

        assert os.path.exists(volume_path), "Volume file not created"
        assert os.path.exists(heatmap_path), "Heatmap file not created"
        assert os.path.exists(metadata_path), "Metadata file not created"

        volume = np.load(volume_path)
        heatmap = np.load(heatmap_path)
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert volume.shape == heatmap.shape, "Volume and heatmap shape mismatch"
        assert metadata["tomo_id"] == tomo_id
        assert "original_spacing" in metadata