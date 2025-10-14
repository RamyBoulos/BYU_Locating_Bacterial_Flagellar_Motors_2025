from src.types.types import DatasetType
import os
import numpy as np
import sys


from src.utils.utills import (
    get_csv_data_frame,
    generate_filename,
    get_labels_directory,
    get_negative_tomo_id_slices,
    get_positive_tomo_id_slices,
    find_tomo_id_slice_attributes,
    copy_files
)

# Add project root to system path to allow src module imports
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root set to: {project_root}")

from src import config 


def generate_labels_txt(type: DatasetType):
    df = get_csv_data_frame(type)
    labels_dir = get_labels_directory(type)

    os.makedirs(labels_dir, exist_ok=True)

    # get all tomo_ids in the dataframe
    tomo_ids = df["tomo_id"].unique()
    print(f"Found {len(tomo_ids)} unique tomo_ids in the {type.value} dataset.")

    for tomo_id in tomo_ids:
        # generate_filename(tomo_id, slice_length)
        negative_tomo_id_slices = get_negative_tomo_id_slices(tomo_id)
        positive_tomo_id_slices = get_positive_tomo_id_slices(tomo_id)

        # 3:1 sampling 1 positive 3 negative
        sampled_negative_slices = np.random.choice(
                negative_tomo_id_slices,
                size=len(positive_tomo_id_slices) * 1,
                replace=False,
            )
        
        negative_tomo_id_slices = sampled_negative_slices.tolist()

        file_names_negative = [generate_filename(tomo_id, i) for i in negative_tomo_id_slices]
        file_names_negative = [f.replace(".jpg", ".txt") for f in file_names_negative]
        file_paths_negative = [os.path.join(labels_dir, f) for f in file_names_negative]
        print(f"Generating negative labels for tomo_id: {tomo_id} with {len(file_names_negative)} slices.")

        file_names_positive = [generate_filename(tomo_id, i) for i in positive_tomo_id_slices]
        file_names_positive = [f.replace(".jpg", ".txt") for f in file_names_positive]
        file_paths_positive = [os.path.join(labels_dir, f) for f in file_names_positive]
        print(f"Generating positive labels for tomo_id: {tomo_id} with {len(file_names_positive)} slices.")

        len(positive_tomo_id_slices)
        len(negative_tomo_id_slices)

        copy_files(positive_tomo_id_slices, negative_tomo_id_slices, tomo_id, type)

        # now create empty txt files for each file_paths
        for file_path in file_paths_negative:
            with open(file_path, "w") as f:
                pass  # create an empty file
        print(f"Finished generating empty labels for tomo_id: {tomo_id}")

        for slice_idx, file_path in zip(positive_tomo_id_slices, file_paths_positive):
            attributes = find_tomo_id_slice_attributes(tomo_id, slice_idx)
            if attributes is None or attributes.empty:
                print(
                    f"No attributes found for tomo_id: {tomo_id}, slice_idx: {slice_idx}. Skipping..."
                )
                continue
            image_width = attributes.iloc[0]["Array shape (axis 2)"]
            image_height = attributes.iloc[0]["Array shape (axis 1)"]
            x_center = attributes.iloc[0]["Motor axis 2"]
            y_center = attributes.iloc[0]["Motor axis 1"]

            x_center = x_center / image_width
            y_center = y_center / image_height

            # fix width and height normalization later , should i change it correspoinding to different image width and height
            box_width =  config.BOX_LENGTH / image_width
            box_height = config.BOX_LENGTH / image_height

            with open(file_path, "w") as f:
                # f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

        print(f"Finished generating positive labels for tomo_id: {tomo_id}")


if __name__ == "__main__":
    generate_labels_txt(DatasetType.TRAIN)
    generate_labels_txt(DatasetType.VALID)
    generate_labels_txt(DatasetType.TEST)
