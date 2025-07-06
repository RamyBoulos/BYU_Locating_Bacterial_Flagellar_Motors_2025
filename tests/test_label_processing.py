import pandas as pd
from preprocessing_3d import label_processing


def test_load_labels():
    df = label_processing.load_labels()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_filter_valid_labels():
    # Create a test dataframe with valid and invalid rows
    test_data = pd.DataFrame({
        "tomo_id": ["tomo1", "tomo1", "tomo2"],
        "Motor axis 0": [10, -1, 20],
        "Motor axis 1": [50, 60, -1],
        "Motor axis 2": [70, 80, 90]
    })

    filtered = label_processing.filter_valid_labels(test_data)
    assert len(filtered) == 1
    assert filtered.iloc[0]["Motor axis 0"] == 10


def test_group_labels_by_tomo():
    test_data = pd.DataFrame({
        "tomo_id": ["tomo1", "tomo1", "tomo2"],
        "Motor axis 0": [10, 20, 30],
        "Motor axis 1": [100, 110, 120],
        "Motor axis 2": [200, 210, 220]
    })

    grouped = label_processing.group_labels_by_tomo(test_data)
    assert "tomo1" in grouped and "tomo2" in grouped
    assert grouped["tomo1"] == [(10, 100, 200), (20, 110, 210)]
    assert grouped["tomo2"] == [(30, 120, 220)]
