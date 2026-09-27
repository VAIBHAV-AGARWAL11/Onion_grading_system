from ultralytics import YOLO

def main():
    print("Starting 5-epoch fine-tuning for Defect Model...")
    model = YOLO("d:/AgriNovaa/models/onion_defect_best_v2.pt")
    # Fine-tune for 5 epochs
    results = model.train(
        data="d:/AgriNovaa/merged_defect.yaml",
        epochs=5,
        imgsz=360,
        project="d:/AgriNovaa/runs/detect",
        name="onion_defect_finetune",
        exist_ok=True,
        # using the current weights, no need to reset
    )
    print("Defect model training complete!")

if __name__ == "__main__":
    main()
