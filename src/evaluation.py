import os
import sys

import torch
from ultralytics import RTDETR
import pandas as pd
import glob
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from src.utils.utills import (cheap_NMS)
from src.utils.matching import points_matching_dataset

# Add project root to system path to allow src module imports
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root set to: {project_root}")

from src import config
from src.types.types import ModelType


print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

COLS = ["Motor axis 0", "Motor axis 1", "Motor axis 2"]

def create_points_list(df_gt, df_pred, COLS=["Motor axis 0", "Motor axis 1", "Motor axis 2"]):
    all_tomos = sorted(pd.concat([df_gt, df_pred])["tomo_id"].unique())
    p_gt = [df_gt[df_gt["tomo_id"] == tomo][COLS].to_numpy() for tomo in all_tomos]
    p_pred = [df_pred[df_pred["tomo_id"] == tomo][COLS].to_numpy() for tomo in all_tomos]
    return p_gt, p_pred

def get_coordinates(model_type : ModelType):
    return ""
   

if __name__ == "__main__":
    #cheap_NMS(type=ModelType.RTDETR)
    cheap_NMS(type=ModelType.YOLO)
    #get_coordinates(model_type=ModelType.RTDETR)