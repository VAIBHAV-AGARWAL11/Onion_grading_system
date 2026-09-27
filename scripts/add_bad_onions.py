import os
import shutil
from pathlib import Path

def merge_bad_onions():
    # Source dataset (new rotten/bad onions)
    src_path = Path("d:/AgriNovaa/bad onion.v1-bad-onion.yolov11")
    
    # Destination dataset (the main defect dataset)
    dest_path = Path("d:/AgriNovaa/merged_defect_dataset")
    
    # Class mapping: bad-onion (0) -> rotten (2)
    NEW_CLASS_ID = "2"
    PREFIX = "bad_onion_"
    
    for split in ['train', 'valid', 'test']:
        src_img_dir = src_path / split / 'images'
        src_lbl_dir = src_path / split / 'labels'
        
        dest_img_dir = dest_path / split / 'images'
        dest_lbl_dir = dest_path / split / 'labels'
        
        if not src_img_dir.exists():
            continue
            
        print(f"Processing {split} set...")
        count = 0
        
        for img_file in src_img_dir.glob('*.jpg'):
            # Corresponding label file
            lbl_file = src_lbl_dir / (img_file.stem + '.txt')
            
            if not lbl_file.exists():
                continue
                
            new_img_name = f"{PREFIX}{img_file.name}"
            new_lbl_name = f"{PREFIX}{lbl_file.name}"
            
            # Copy image
            shutil.copy(img_file, dest_img_dir / new_img_name)
            
            # Process label file
            with open(lbl_file, 'r') as f:
                lines = f.readlines()
                
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    # The class ID is the first element. We change it to '2' (rotten)
                    if parts[0] == "0":
                        parts[0] = NEW_CLASS_ID
                    new_lines.append(" ".join(parts) + "\n")
                    
            # Write new label file
            with open(dest_lbl_dir / new_lbl_name, 'w') as f:
                f.writelines(new_lines)
                
            count += 1
            
        print(f"Added {count} images to {split}.")
        
    print("Merge complete! The merged_defect_dataset now contains the new rotten/black-spot onions.")

if __name__ == "__main__":
    merge_bad_onions()
