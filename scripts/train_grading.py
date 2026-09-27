from ultralytics import YOLO

def main():
    print("Starting 5-epoch fine-tuning for Grading Model...")
    model = YOLO("d:/AgriNovaa/models/onion_grading_best.pt")
    # Fine-tune for 5 epochs
    results = model.train(
        data="d:/AgriNovaa/onion_grading.yaml",
        epochs=5,
        imgsz=360,
        project="d:/AgriNovaa/runs/detect",
        name="onion_grading_finetune",
        exist_ok=True,
    )
    print("Grading model training complete!")

if __name__ == "__main__":
    main()
