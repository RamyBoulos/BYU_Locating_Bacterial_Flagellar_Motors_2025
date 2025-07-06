import numpy as np
from preprocessing_3d.normalization import normalize_intensity


def test_minmax_normalization_basic():
    volume = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.uint8)
    norm = normalize_intensity(volume, method="minmax")

    assert np.isclose(norm.min(), 0.0)
    assert np.isclose(norm.max(), 1.0)
    assert norm.dtype == np.float32


def test_zscore_normalization_basic():
    volume = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    norm = normalize_intensity(volume, method="zscore")

    mean = norm.mean()
    std = norm.std()

    assert np.isclose(mean, 0.0, atol=1e-5)
    assert np.isclose(std, 1.0, atol=1e-5)
    assert norm.dtype == np.float32


def test_constant_volume_minmax():
    volume = np.full((4, 4, 4), fill_value=5, dtype=np.uint8)
    norm = normalize_intensity(volume, method="minmax")

    assert np.all(norm == 0.0)


def test_constant_volume_zscore():
    volume = np.full((4, 4, 4), fill_value=7, dtype=np.uint8)
    norm = normalize_intensity(volume, method="zscore")

    assert np.all(norm == 0.0)


def test_invalid_method_raises():
    volume = np.random.rand(4, 4, 4).astype(np.float32)
    try:
        normalize_intensity(volume, method="invalid")
    except ValueError as e:
        assert "Unsupported normalization method" in str(e)
    else:
        assert False, "Expected ValueError for invalid method"
