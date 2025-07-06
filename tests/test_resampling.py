import numpy as np
from preprocessing_3d.resampling import resample_volume, resample_coordinates


def test_resample_volume_downscale():
    volume = np.random.rand(20, 40, 40)
    original_spacing = 10.0
    target_spacing = 20.0  # Zoom factor = 0.5

    result = resample_volume(volume, original_spacing, target_spacing)
    expected_shape = tuple(int(round(s * (original_spacing / target_spacing))) for s in volume.shape)

    assert result.shape == expected_shape
    assert result.dtype == volume.dtype


def test_resample_volume_no_op():
    volume = np.random.rand(10, 20, 30)
    result = resample_volume(volume, 15.6, 15.6)

    assert result.shape == volume.shape
    assert np.allclose(result, volume)


def test_resample_coordinates_downscale():
    coords = [(10, 20, 30), (5, 10, 15)]
    original_spacing = 20.0
    target_spacing = 10.0  # Zoom factor = 2.0

    expected = [(20, 40, 60), (10, 20, 30)]
    result = resample_coordinates(coords, original_spacing, target_spacing)

    assert result == expected


def test_resample_coordinates_no_op():
    coords = [(1, 2, 3), (4, 5, 6)]
    result = resample_coordinates(coords, 15.6, 15.6)

    assert result == coords
