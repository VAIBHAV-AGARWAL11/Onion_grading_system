import os
import shutil
from pathlib import Path

def merge_datasets():
    # Source datasets
    ds1_path = Path("d:/AgriNovaa/veg1-hcqsf-2.v4-demo_model.yolov11")
    ds2_path = Path("d:/AgriNovaa/Onion detection.v1i.yolov11")
    
    # Destination
    out_path = Path("d:/AgriNovaa/merged_defect_dataset")
    
    # Create output directories
    for split in ['train', 'valid', 'test']:
        (out_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (out_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    def process_dataset(ds_path, prefix, class_map=None):
        for split in ['train', 'valid', 'test']:
            img_dir = ds_path / split / 'images'
            lbl_dir = ds_path / split / 'labels'
            
            if not img_dir.exists():
                continue
                
            for img_file in img_dir.glob('*.jpg'):
                # Copy image
                new_img_name = f"{prefix}_{img_file.name}"
                shutil.copy(img_file, out_path / split / 'images' / new_img_name)
                
                # Copy and potentially modify label
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                new_lbl_name = f"{prefix}_{img_file.stem}.txt"
                new_lbl_path = out_path / split / 'labels' / new_lbl_name
                
                if lbl_file.exists():
                    with open(lbl_file, 'r') as f:
                        lines = f.readlines()
                    
                    with open(new_lbl_path, 'w') as f:
                        for line in lines:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            class_id = int(parts[0])
                            
                            # Remap class if mapping is provided
                            if class_map and class_id in class_map:
                                class_id = class_map[class_id]
                                
                            f.write(f"{class_id} {' '.join(parts[1:])}\n")

    print("Merging Dataset 1 (veg1-hcqsf-2)...")
    # Dataset 1 has correct mapping: 0=double_split, 1=onion, 2=rotten, 3=sprout
    process_dataset(ds1_path, "ds1", class_map=None)
    
    print("Merging Dataset 2 (Onion detection)...")
    # Dataset 2 mapping: 0=rotten->2, 1=sprout->3
    ds2_mapping = {0: 2, 1: 3}
    process_dataset(ds2_path, "ds2", class_map=ds2_mapping)
    
    print("Merge complete! Output saved to merged_defect_dataset/")

if __name__ == "__main__":
    merge_datasets()
