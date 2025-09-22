from pathlib import Path
import shutil
from src.types.types import DatasetType


from src.utils.utills import (
    find_all_tomo_ids,
    generate_absolute_paths__for_tomo_id,
    target_file_paths,
)


def move_files_to_yolo_format(type: DatasetType = DatasetType.TRAIN):
    """Move files to YOLO format directory structure."""
    tomos = find_all_tomo_ids(type=type)
    count = 0

    for tomo_id in tomos:
        src_files = generate_absolute_paths__for_tomo_id(tomo_id)
        dst_files = target_file_paths(tomo_id, type=type)
        for src, dst in zip(src_files, dst_files):
            dst_path = Path(dst)

            if dst_path.exists():
                print(f"Skipping {dst} as it already exists.")
                continue

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            print(f"{src} -> {dst}")
        count += 1
        print(f"Processed {count}/{len(tomos)}: last tomo_id {tomo_id}")


if __name__ == "__main__":
    move_files_to_yolo_format(type=DatasetType.VALID)
    move_files_to_yolo_format(type=DatasetType.TEST)
