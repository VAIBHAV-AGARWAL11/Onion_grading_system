import argparse
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Run YOLO11s Inference on an image")
    parser.add_argument('--weights', type=str, default="models/best.pt", help="Path to best.pt")
    parser.add_argument('--source', type=str, required=True, help="Path to image or directory of images")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        print(f"Error: Weights file {args.weights} not found. Please train the model first.")
        return
        
    print(f"Loading model from {args.weights}...")
    model = YOLO(args.weights)
    
    print(f"\nRunning inference on {args.source}...")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        device=0,
        save=True,
        project="runs/predict",
        name="inference_results",
        exist_ok=True
    )
    
    print("\n--- Predictions ---")
    for idx, r in enumerate(results):
        print(f"\nImage {idx+1}:")
        boxes = r.boxes
        if len(boxes) == 0:
            print("No onions detected.")
        else:
            print(f"Detected {len(boxes)} onions:")
            for box in boxes:
                # Class name
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id]
                # Confidence
                conf = box.conf[0].item() * 100
                # Bounding Box coordinates
                xyxy = box.xyxy[0].tolist()
                xyxy_rounded = [round(c, 1) for c in xyxy]
                print(f"- Grade: {cls_name} | Confidence: {conf:.1f}% | BBox: {xyxy_rounded}")
                
    print("\nAnnotated images saved to: runs/predict/inference_results/")

if __name__ == "__main__":
    main()
