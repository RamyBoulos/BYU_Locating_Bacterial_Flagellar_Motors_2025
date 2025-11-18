from ultralytics import RTDETR, YOLO
import os
import sys
import glob
import pandas as pd
from pathlib import Path

# Add project root to system path to allow src module imports
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root set to: {project_root}")

from src import config 

def get_test_files_path():
   
    test_images =  config.YOLO_TEST_FORMAT_DIR
    test_files = glob.glob(os.path.join(test_images, "*.jpg"))
    print(f"Found {len(test_files)} test files.")
    return test_files

def get_tomo_id_and_slice_id_from_image_path(image_path: str):
    stem = Path(image_path).stem
    parts = stem.split("_slice_")
    tomo_id = parts[0]     # "tomo_675583"
    slice_id = int(parts[1])    # "0187" # -> tomo_675583_slice_0187
    return tomo_id, slice_id


from typing import Union

rows = []


def inference_on_image(image_path: str, model: Union[RTDETR, YOLO]):
    results = model(image_path, stream=True)
    
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        tomo_id, slice_id = get_tomo_id_and_slice_id_from_image_path(result.path)
        for box in boxes:
            x_center, y_center, box_width, box_height = box.xywh[0].tolist()
            row = (tomo_id, slice_id, float(box.conf), x_center, y_center, box_width, box_height)
            rows.append(row)
            print(row)
            print(f"Processed {image_path}, found {len(boxes)} boxes.")
            print("Rows length so far:", len(rows))
            print("--------------------------------------------------")


def inference_on_images(image_paths: list[str], model: Union[RTDETR, YOLO]):
    for image_path in image_paths:
        inference_on_image(image_path, model)
    

rtdetr_weights_path = "/data/horse/ws/kein254g-team_project/rtdetr_training_result/exp2/motor_rtdetr_x_102410/weights/best.pt"
rtdetr_results_path = "/home/kein254g/BYU_Locating_Bacterial_Flagellar_Motors_2025/data/raw/rtdetr_coordinates.csv"

yolo_weights_path = "/data/horse/ws/kein254g-team_project/yolo_training_result/exp2/motor_yolo_10245/weights/best.pt"
yolo_results_path = "/home/kein254g/BYU_Locating_Bacterial_Flagellar_Motors_2025/data/raw/yolo_coordinates.csv"

if __name__ == "__main__":
    files = get_test_files_path()
    #inference_on_images(files, RTDETR(config.RTDETR_BEST_WEIGHTS))
    #df = pd.DataFrame(rows, columns=["tomo_id", "Motor axis 0", "confidence", "Motor axis 2", "Motor axis 1", "box_width", "box_height"])
    #df.to_csv(config.DENORMALIZED_RESULTS_RTDETR, index=False)

    inference_on_images(files, YOLO(config.YOLO_BEST_WEIGHTS))
    df_yolo = pd.DataFrame(rows, columns=["tomo_id", "Motor axis 0", "confidence", "Motor axis 2", "Motor axis 1", "box_width", "box_height"])
    df_yolo.to_csv(config.DENORMALIZED_RESULTS_YOLO, index=False)

#df = pd.DataFrame(rows, columns=["tomo_id", "Motor axis 0", "confidence", "Motor axis 2", "Motor axis 1", "box_width", "box_height"])
#df.to_csv("rtdetr_coordinates.csv", index=False)