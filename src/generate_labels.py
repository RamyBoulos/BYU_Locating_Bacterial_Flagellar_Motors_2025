from src.utils.utills  import (
    get_csv_data_frame,
    get_labels_directory,
    find_z_axis_length,
    generate_filename,
    get_positive_tomo_id_slices,
    find_tomo_id_slice_attributes
)

from src.types.types import DatasetType

import os


def generate_labels_txt(type : DatasetType):
    df = get_csv_data_frame(type)
    labels_dir = get_labels_directory(type)
    os.makedirs(labels_dir, exist_ok=True)

    # get all tomo_ids in the dataframe
    tomo_ids = df['tomo_id'].unique()
    print(f"Found {len(tomo_ids)} unique tomo_ids in the {type.value} dataset.")

    for tomo_id in tomo_ids:
        slice_length = find_z_axis_length(tomo_id)
        # generate_filename(tomo_id, slice_length)
        file_names = [generate_filename(tomo_id, i) for i in range(slice_length)]
        file_names = [f.replace('.jpg', '.txt') for f in file_names]
        file_paths = [os.path.join(labels_dir, f) for f in file_names]
        print(f"Generating labels for tomo_id: {tomo_id} with {len(file_names)} slices.")

        # now create empty txt files for each file_paths
        for file_path in file_paths:
            with open(file_path, 'w') as f:
                pass  # create an empty file
        print(f"Finished generating empty labels for tomo_id: {tomo_id}")
        positive_tomo_id_slices = get_positive_tomo_id_slices(tomo_id)
        for slice_idx in positive_tomo_id_slices:
            file_name = generate_filename(tomo_id, slice_idx).replace('.jpg', '.txt')
            file_path = os.path.join(labels_dir, file_name)
            attributes = find_tomo_id_slice_attributes(tomo_id, slice_idx)
            if attributes is None or attributes.empty:
                print(f"No attributes found for tomo_id: {tomo_id}, slice_idx: {slice_idx}. Skipping...")
                continue
            width = attributes.iloc[0]["Array shape (axis 2)"]
            height = attributes.iloc[0]["Array shape (axis 1)"]
            x_center = attributes.iloc[0]["Motor axis 2"]
            y_center = attributes.iloc[0]["Motor axis 1"]
            # convert to yolo format
            x_center /= width
            y_center /= height
            # fix width and height normalization later
            BOX_LENGTH = 20  # pixels
            width = BOX_LENGTH / width
            height = BOX_LENGTH / height

            with open(file_path, 'w') as f:
                f.write(f"0 {x_center} {y_center} {width} {height}\n")
        print(f"Finished generating positive labels for tomo_id: {tomo_id}")

if __name__ == "__main__":
    generate_labels_txt(DatasetType.VALID)
    generate_labels_txt(DatasetType.TEST)
    generate_labels_txt(DatasetType.TRAIN)