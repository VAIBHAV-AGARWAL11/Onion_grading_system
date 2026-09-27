import argparse
import sys
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11s for Onion Grading")
    parser.add_argument('--smoke-test', action='store_true', help="Run a quick 2-epoch smoke test")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch', type=int, default=16, help="Batch size. Use -1 for AutoBatch")
    args = parser.parse_args()

    # Phase 2: Check Environment and GPU constraints
    print("--- GPU ENVIRONMENT CHECK ---")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Please fix the environment before training.")
        print("This script is configured to use the local NVIDIA GPU exclusively.")
        sys.exit(1)
    
    device_id = 0
    gpu_name = torch.cuda.get_device_name(device_id)
    vram_bytes = torch.cuda.get_device_properties(device_id).total_memory
    vram_gb = round(vram_bytes / (1024 ** 3), 2)
    
    print(f"Device: NVIDIA GPU")
    print(f"CUDA: Available")
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {vram_gb} GB")
    print("-----------------------------\n")

    yaml_path = "d:/AgriNovaa/onion_grading.yaml"
    model_name = "yolo11s.pt"
    
    epochs = 2 if args.smoke_test else args.epochs
    run_name = "yolo11s_baseline_smoke" if args.smoke_test else "yolo11s_baseline"
    
    print(f"Loading pre-trained model: {model_name}")
    model = YOLO(model_name)

    print(f"\nStarting {'SMOKE TEST' if args.smoke_test else 'FULL TRAINING'} for {epochs} epochs...")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=args.batch,
        device=0,
        workers=2, # Safe value for Windows 
        patience=20,
        project="runs/onion_grading",
        name=run_name,
        pretrained=True,
        cache=False,
        plots=True,
        exist_ok=True
    )
    
    print("\nTraining completed successfully!")
    print(f"Results and weights are saved in: runs/onion_grading/{run_name}/")
    print(f"Best model path: runs/onion_grading/{run_name}/weights/best.pt")

if __name__ == "__main__":
    main()
