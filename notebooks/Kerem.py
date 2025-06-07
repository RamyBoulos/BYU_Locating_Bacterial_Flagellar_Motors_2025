import os
import pandas as pd
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

# Input paths
data_root = '/Users/ramy/Desktop/team_project/BYU_Locating_Bacterial_Flagellar_Motors_2025/sampled_train'
labels_csv = '/Users/ramy/Desktop/team_project/BYU_Locating_Bacterial_Flagellar_Motors_2025/data/sampled/sampled_train_labels.csv'

# Output base paths
output_base = '/Users/ramy/Desktop/team_project/BYU_Locating_Bacterial_Flagellar_Motors_2025/data/yolo'
output_images_train = os.path.join(output_base, 'images/train')
output_images_val = os.path.join(output_base, 'images/val')
output_labels_train = os.path.join(output_base, 'labels/train')
output_labels_val = os.path.join(output_base, 'labels/val')

# Create output directories
for path in [output_images_train, output_images_val, output_labels_train, output_labels_val]:
    os.makedirs(path, exist_ok=True)

# Load labels CSV
labels_df = pd.read_csv(labels_csv)

# Build a dict of positive samples: image_path -> (x, y, width, height, img_width, img_height)
positive_samples = {}

for idx, row in labels_df.iterrows():
    if row['Motor axis 0'] == -1:  # -1 means no motor (negative sample)
        continue
    tomo_id = row['tomo_id']
    z = int(row['Motor axis 0'])  # slice number
    y = row['Motor axis 1']
    x = row['Motor axis 2']
    img_width = row['Array shape (axis 2)']
    img_height = row['Array shape (axis 1)']
    img_name = f"slice_{z:04d}.jpg"
    image_path = os.path.join(data_root, tomo_id, img_name)
    positive_samples[image_path] = (x, y, img_width, img_height)

# Gather all image paths
all_images = []
for root, _, files in os.walk(data_root):
    for file in files:
        if file.endswith('.jpg'):
            full_path = os.path.join(root, file)
            all_images.append(full_path)

# Shuffle and split
random.shuffle(all_images)
split_ratio = 0.8  # 80% train, 20% val
split_index = int(len(all_images) * split_ratio)
train_images = all_images[:split_index]
val_images = all_images[split_index:]

def process_images(image_list, img_output_dir, lbl_output_dir):
    for img_path in image_list:
        # Prepare destination paths
        tomo_id = os.path.basename(os.path.dirname(img_path))
        z = int(img_path.split('_')[-1].split('.')[0])
        img_filename = f"{tomo_id}_slice_{z:04d}.jpg"
        label_filename = f"{tomo_id}_slice_{z:04d}.txt"

        dest_img_path = os.path.join(img_output_dir, img_filename)
        dest_lbl_path = os.path.join(lbl_output_dir, label_filename)

        # Copy image
        shutil.copyfile(img_path, dest_img_path)

        if img_path in positive_samples:
            # Create label with bounding box
            x, y, img_width, img_height = positive_samples[img_path]
            bbox_width = 10
            bbox_height = 10
            center_x = x / img_width
            center_y = y / img_height
            norm_width = bbox_width / img_width
            norm_height = bbox_height / img_height

            with open(dest_lbl_path, 'w') as f:
                f.write(f"0 {center_x} {center_y} {norm_width} {norm_height}\n")
        else:
            # Negative sample: Create empty label file
            open(dest_lbl_path, 'w').close()

# Process train and val
process_images(train_images, output_images_train, output_labels_train)
process_images(val_images, output_images_val, output_labels_val)

print(f"Dataset created with {len(train_images)} training and {len(val_images)} validation images, including negative samples!")