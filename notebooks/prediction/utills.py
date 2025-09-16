import os
import sys
from pathlib import Path
from PIL import Image
import pandas as pd

# Ensure repository root is on sys.path so `src` is importable
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src import config

def find_all_tomoId_attributes(tomo_id):
    labels_df = pd.read_csv(config.TRAIN_LABELS_PATH)
    filtered_df = labels_df[labels_df['tomo_id'] == tomo_id]
    
    if filtered_df.empty:
        return None # Return None if tomo_id not found

    print(filtered_df)
    return filtered_df

def find_tomoId_slice_attributes(tomo_id, slice_num):
    labels_df = pd.read_csv(config.TRAIN_LABELS_PATH)
    filtered_df = labels_df[(labels_df['tomo_id'] == tomo_id) & (labels_df['Motor axis 0'] == slice_num)]
    
    if filtered_df.empty:
        return None # Return None if tomo_id or slice_num not found

    print(filtered_df)
    return filtered_df

def find_width_height(tomo_id, slice_num):
    attributes = find_tomoId_slice_attributes(tomo_id, slice_num)

    if attributes is None or attributes.empty:
        print(f"No attributes found for tomo_id: {tomo_id}, slice_num: {slice_num}")
        return None, None
    
    width = attributes.iloc[0]['Array shape (axis 2)']
    height = attributes.iloc[0]['Array shape (axis 1)']
    print(f"Width: {width}, Height: {height}")
    return width, height





if __name__ == "__main__":
    print(find_all_tomoId_attributes("tomo_00e047"))
