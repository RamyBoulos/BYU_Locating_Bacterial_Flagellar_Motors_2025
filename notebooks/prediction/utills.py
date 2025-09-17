import os
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
from pathlib import Path
import glob


# Ensure repository root is on sys.path so `src` is importable
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src import config

#trainLabelDataFrame = pd.read_csv(config.TRAIN_LABELS_PATH)

from typing import Union

def find_all_predicted_tomoId_attributes(tomo_id:str) -> Union[pd.DataFrame, None]:
    labels_df = pd.read_csv(config.DENORMALIZED_RESULTS)
    filtered_df = labels_df[labels_df['tomo_id'] == tomo_id]
    
    if filtered_df.empty:
        return None 

    return filtered_df

def find_predicted_tomoId_slice_attributes(tomo_id, slice_num) -> Union[pd.DataFrame, None]:
    labels_df = pd.read_csv(config.DENORMALIZED_RESULTS)
    filtered_df = labels_df[(labels_df['tomo_id'] == tomo_id)]
    filtered_df = labels_df[(labels_df['Motor axis 0'] == slice_num)]

    if filtered_df.empty:
        return None # Return None if tomo_id or slice_num not found

    return filtered_df

def find_all_tomoId_attributes(tomo_id : str) -> Union[pd.DataFrame, None]:
    labels_df = pd.read_csv(config.TRAIN_LABELS_PATH)
    filtered_df = labels_df[labels_df['tomo_id'] == tomo_id]
    
    if filtered_df.empty:
        return None # Return None if tomo_id not found

    return filtered_df

def find_tomoId_slice_attributes(tomo_id, slice_num) -> Union[pd.DataFrame, None]:
    print(config.TRAIN_LABELS_PATH)
    labels_df = pd.read_csv(config.TRAIN_LABELS_PATH)
    filtered_df = labels_df[(labels_df['tomo_id'] == tomo_id)]
    filtered_df = labels_df[(labels_df['Motor axis 0'] == slice_num)]

    if filtered_df.empty:
        return None # Return None if tomo_id or slice_num not found

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


def parse_txt(file_path, f):
        file_name = os.path.basename(file_path)            # e.g. "image1.txt"
        file_id = os.path.splitext(file_name)[0]           # e.g. "image1"
        _,id,_,slice = file_id.split('_')
        slice = int(slice)
        tomoId = "tomo_" + str(id)
        _,confidence,x_center,y_center,width,height = f.readline().strip().split()
        return tomoId,slice,confidence,x_center,y_center,width,height

def cheap_nms():
    return

def rescale_letterbox():
    rows = []

    print(f"{config.RTDETR_RESULT_LABELS}/*.txt")
    txt_files = sorted(glob.glob(f"{config.RTDETR_RESULT_LABELS}/*.txt"))
    for file_path in txt_files:
        print(f"\n--- Reading: {file_path} ---")
        with open(file_path, "r", encoding="utf-8") as f:
            #print(parse_txt(file_path,f))
            tomoId,slice,confidence,x_center,y_center,width,height = parse_txt(file_path,f)
            Width, Height = find_width_height(tomoId, slice)

            if Width is None or Height is None:
                print(f"Skipping {tomoId}, {slice} due to missing dimensions.")
                continue

            print(f"Denormalized: {tomoId}, {slice}, {confidence}, {x_center}, {y_center}, {width}, {height}")
            # denomarize
            x_center = float(x_center) * Width
            y_center = float(y_center) * Height
            width = float(width) * Width
            height = float(height) * Height

            #what about voxel spacing ?
            print(f"Denormalized: {tomoId}, {slice}, {confidence}, {x_center}, {y_center}, {width}, {height}")

            rows.append({
                'tomo_id': tomoId,
                'slice': slice,
                'confidence': float(confidence),
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
   

    df = pd.DataFrame(rows)
    df.to_csv(config.DENORMALIZED_RESULTS, index=False)






if __name__ == "__main__":
    print(find_all_tomoId_attributes("tomo_00e047"))
