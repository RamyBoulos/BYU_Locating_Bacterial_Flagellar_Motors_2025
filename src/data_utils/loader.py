

"""
loader.py

Utility functions for dataset path resolution and loading,
based on config flags.
"""

from src import config


def get_data_sources():
    """
    Determine which dataset (sampled or full) to use and return
    corresponding label CSV path and tomogram directory root.

    Returns
    -------
    Tuple[str, str]
        (label_csv_path, tomogram_root_dir)
    """
    if config.USE_SAMPLED_TRAIN_DATASET:
        return config.SAMPLED_LABELS_PATH, config.SAMPLED_TOMO_DIR
    return config.TRAIN_LABELS_PATH, config.FULL_DATA_TRAIN_DIR
