import numpy as np
from preprocessing_3d.volume_io import load_volume_from_jpegs
from src import config


def test_load_volume_from_jpegs():
    tomo_dir = config.EXAMPLE_TOMO_DIR
    volume = load_volume_from_jpegs(tomo_dir)

    # Assert volume is a 3D numpy array
    assert isinstance(volume, np.ndarray), "Volume should be a numpy array"
    assert volume.ndim == 3, "Volume should be 3-dimensional (Z, Y, X)"

    # Assert dimensions are non-zero
    z, y, x = volume.shape
    assert z > 0 and y > 0 and x > 0, "Volume dimensions must be non-zero"
    assert volume.dtype == np.uint8, "Expected dtype uint8 for grayscale slices"
