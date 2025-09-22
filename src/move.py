from pathlib import Path
import shutil

from src.utils.utills import (
    find_all_train_tomo_ids,
    generate_absolute_paths__for_tomo_id,
    target_file_paths,
)


def move_files_to_yolo_format():
    """Move files to YOLO format directory structure."""
    # This function is a placeholder and needs to be implemented based on specific requirements.
    tomos = find_all_train_tomo_ids()
    count = 0

    for tomo_id in tomos:
        src_files = generate_absolute_paths__for_tomo_id(tomo_id)
        dst_files = target_file_paths(tomo_id)
        for src, dst in zip(src_files, dst_files):
            dst_path = Path(dst)

            if dst_path.exists():
                print(f"Skipping {dst} as it already exists.")
                continue

            dst_path.parent.mkdir(parents=True, exist_ok=True)  # klasörü yoksa oluştur
            shutil.copy(src, dst)  # taşıma istiyorsan copy yerine move yaz
            print(f"{src} -> {dst}")
        count += 1
        print(f"Processed {count}/{len(tomos)}: {tomo_id}")
