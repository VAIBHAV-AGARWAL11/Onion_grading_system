# AgriNovaa Onion Grading AI

This project trains and runs an object detection model to grade onions using YOLO11s.

## Objective
Automatically detect and classify onions into four distinct grades:
1. **Class 1** (0)
2. **Class 2** (1)
3. **Extra Class** (2)
4. **Reject** (3)

## Dataset
The model is trained strictly on the local dataset: `Onion Grading.v7i.yolov11`.
- **Train:** 2380 images
- **Validation:** 291 images
- **Test:** 292 images

## Environment & Hardware
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (~6 GB VRAM)
- **CUDA:** 13.1 (Driver 592.82)
- **Framework:** PyTorch (with CUDA 12.1) & Ultralytics YOLOv11

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Verify GPU status:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Workflow

### 1. Training
To train the model on your local GPU for 100 epochs:
```bash
python scripts/train.py
```
> The best model weights will be saved to `runs/onion_grading/yolo11s_baseline/weights/best.pt`.

### 2. Validation
To evaluate the model on the test dataset and generate metrics (mAP, Precision, Recall):
```bash
python scripts/validate.py
```

### 3. Inference
To test the model on a single image or folder of images:
```bash
python scripts/predict.py --source "Onion Grading.v7i.yolov11/test/images"
```
Predictions will be saved in `runs/predict/inference_results/`.

### 4. Interactive Interface (Streamlit)
To run a user-friendly web interface for drag-and-drop testing:
```bash
streamlit run app.py
```

## Directory Structure
- `Onion Grading.v7i.yolov11/` : Original Dataset
- `onion_grading.yaml` : Absolute path configuration for YOLO
- `scripts/` : Python scripts for train, validate, and predict
- `runs/` : Ultralytics training logs, weights, and plots
- `models/` : Stores the final copied `best.pt` model
- `app.py` : Streamlit application
