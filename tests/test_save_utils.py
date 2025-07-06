import os
import json
import numpy as np
import tempfile
from preprocessing_3d.save_utils import save_npy, save_json, save_all_outputs


def test_save_npy_creates_file_and_data_matches():
    array = np.random.rand(5, 5, 5).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_volume.npy")
        save_npy(array, path)

        assert os.path.exists(path)

        loaded = np.load(path)
        assert np.allclose(array, loaded)
        assert loaded.dtype == np.float32


def test_save_json_creates_file_and_content_matches():
    metadata = {
        "tomo_id": "tomo_test",
        "shape": [5, 5, 5],
        "spacing": 15.6,
        "motors": 2
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "metadata.json")
        save_json(metadata, path)

        assert os.path.exists(path)

        with open(path, "r") as f:
            loaded = json.load(f)

        assert loaded == metadata


def test_save_all_outputs_creates_expected_files_and_data():
    volume = np.random.rand(4, 4, 4).astype(np.float32)
    heatmap = np.random.rand(4, 4, 4).astype(np.float32)
    metadata = {
        "tomo_id": "tomo_123",
        "shape": [4, 4, 4],
        "spacing": 15.6
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        save_all_outputs(
            tomo_id="tomo_123",
            volume=volume,
            heatmap=heatmap,
            metadata=metadata,
            output_dir=tmpdir
        )

        base_path = os.path.join(tmpdir, "tomo_123")
        vol_path = os.path.join(base_path, "volume.npy")
        heat_path = os.path.join(base_path, "heatmap.npy")
        meta_path = os.path.join(base_path, "metadata.json")

        assert os.path.exists(vol_path)
        assert os.path.exists(heat_path)
        assert os.path.exists(meta_path)

        assert np.allclose(np.load(vol_path), volume)
        assert np.allclose(np.load(heat_path), heatmap)

        with open(meta_path, "r") as f:
            loaded_meta = json.load(f)
        assert loaded_meta == metadata
