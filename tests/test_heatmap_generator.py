import numpy as np
from preprocessing_3d.heatmap_generator import generate_gaussian_heatmap


def test_generate_gaussian_heatmap_shape():
    shape = (10, 32, 32)
    coords = [(5, 16, 16)]
    heatmap = generate_gaussian_heatmap(shape, coords)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == shape
    assert heatmap.dtype == np.float32


def test_peak_value_close_to_one():
    shape = (10, 32, 32)
    coords = [(5, 16, 16)]
    heatmap = generate_gaussian_heatmap(shape, coords, sigma=1.5)

    peak = heatmap[5, 16, 16]
    assert peak > 0.99, f"Expected peak near 1.0, got {peak}"


def test_out_of_bounds_coord_handling():
    shape = (10, 32, 32)
    coords = [(50, 50, 50)]  # Clearly out of bounds
    heatmap = generate_gaussian_heatmap(shape, coords)

    assert np.all(heatmap == 0.0), "Heatmap should remain empty for out-of-bounds input"


def test_multiple_peaks():
    shape = (10, 32, 32)
    coords = [(5, 10, 10), (5, 20, 20)]
    heatmap = generate_gaussian_heatmap(shape, coords)

    assert heatmap[5, 10, 10] > 0.9
    assert heatmap[5, 20, 20] > 0.9
