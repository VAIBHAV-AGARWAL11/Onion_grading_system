import argparse
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Validate YOLO11s for Onion Grading")
    parser.add_argument('--weights', type=str, default="models/best.pt", help="Path to best.pt")
    parser.add_argument('--data', type=str, default="d:/AgriNovaa/onion_grading.yaml", help="Path to yaml config")
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        print(f"Error: Weights file {args.weights} not found. Please train the model first.")
        return
        
    print(f"Loading model from {args.weights}...")
    model = YOLO(args.weights)
    
    print("\nStarting validation on the TEST dataset split...")
    # Validate on test set
    metrics = model.val(
        data=args.data,
        split='test',
        device=0,
        project="runs/onion_grading",
        name="val_results"
    )
    
    print("\n--- Validation Results ---")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    
    # metrics.class_result returns a list of results per class if needed
    print(f"\nDetailed metrics are saved in: runs/onion_grading/val_results/")
    print(f"Confusion matrix is available in: runs/onion_grading/val_results/confusion_matrix.png")

if __name__ == "__main__":
    main()
