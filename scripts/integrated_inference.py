import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def calculate_intersection_ratio(small_box, large_box):
    """Calculate what fraction of the small_box is inside the large_box."""
    x1 = max(small_box[0], large_box[0])
    y1 = max(small_box[1], large_box[1])
    x2 = min(small_box[2], large_box[2])
    y2 = min(small_box[3], large_box[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    small_area = (small_box[2] - small_box[0]) * (small_box[3] - small_box[1])
    
    if small_area == 0:
        return 0.0
    return inter_area / float(small_area)

class OnionInspector:
    def __init__(self, defect_model_path="models/onion_defect_best.pt", grading_model_path="models/onion_grading_best.pt"):
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Inference will run on CPU.")
            self.device = 'cpu'
        else:
            self.device = 0
            
        print("Loading Defect Model...")
        self.defect_model = YOLO(defect_model_path)
        print("Loading Grading Model...")
        self.grading_model = YOLO(grading_model_path)
        
    def _parse_results(self, results_obj):
        """Parse YOLO results object into a list of dicts."""
        detections = []
        if len(results_obj) == 0:
            return detections
        
        result = results_obj[0]
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = result.names[cls_id]
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            detections.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": xyxy
            })
        return detections

    def inspect(self, source, conf_threshold=0.25):
        if isinstance(source, Image.Image):
            img_bgr = cv2.cvtColor(np.array(source.convert('RGB')), cv2.COLOR_RGB2BGR)
            img_disp = img_bgr.copy()
        elif isinstance(source, np.ndarray):
            img_bgr = source.copy()
            img_disp = img_bgr.copy()
        else:
            img_bgr = cv2.imread(source)
            if img_bgr is None:
                raise ValueError(f"Could not read image from {source}")
            img_disp = img_bgr.copy()

        # Inference
        # 1. Defect Inference (Detect all onions and defects)
        defect_preds = self.defect_model.predict(source=img_bgr, conf=conf_threshold, device=self.device, verbose=False)
        defect_detections = self._parse_results(defect_preds)
        
        anchors = []
        isolated_defects = []
        
        for d in defect_detections:
            if d['class'] == 'onion':
                anchors.append({
                    "object": d,
                    "grade": None,
                    "defects": []
                })
            else:
                isolated_defects.append(d)
                
        # 2. Grading Inference via Cropping
        img_h, img_w = img_bgr.shape[:2]
        
        for anchor in anchors:
            # Get bounding box for the onion
            x1, y1, x2, y2 = [int(v) for v in anchor['object']['bbox']]
            
            # Ensure within bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)
            
            # Crop the onion
            crop = img_bgr[y1:y2, x1:x2]
            
            # Skip invalid crops
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
                
            # Run grading model on the cropped individual onion
            grade_preds = self.grading_model.predict(source=crop, conf=0.05, device=self.device, verbose=False)
            grade_dets = self._parse_results(grade_preds)
            
            if grade_dets:
                # Pick the highest confidence grade
                best_grade = max(grade_dets, key=lambda x: x['confidence'])
                
                # Shift the local crop bbox back to global image coordinates
                gx1, gy1, gx2, gy2 = best_grade['bbox']
                best_grade['bbox'] = [gx1 + x1, gy1 + y1, gx2 + x1, gy2 + y1]
                
                anchor['grade'] = best_grade

        # 3. Match isolated defects to anchors
        for defect in isolated_defects:
            best_overlap = 0
            best_anchor_idx = -1
            
            for i, anchor in enumerate(anchors):
                if anchor['object']:
                    box = anchor['object']['bbox']
                elif anchor['grade']:
                    box = anchor['grade']['bbox']
                else:
                    continue
                    
                overlap = calculate_intersection_ratio(defect['bbox'], box)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_anchor_idx = i
            
            if best_overlap >= 0.5:
                anchors[best_anchor_idx]['defects'].append(defect)
            else:
                # Case 4: Unassociated defect
                anchors.append({
                    "object": None,
                    "grade": None,
                    "defects": [defect]
                })

        final_results = []
        
        for idx, anchor in enumerate(anchors):
            onion_id = idx + 1
            
            res_dict = {
                "onion_id": onion_id,
                "grade": anchor['grade'],
                "object": anchor['object'],
                "defects": anchor['defects']
            }
            final_results.append(res_dict)
            
            labels = [f"Onion #{onion_id}"]
            if anchor['grade']:
                labels.append(anchor['grade']['class'])
            else:
                labels.append("Grade: Not detected")
                
            if len(anchor['defects']) > 0:
                for d in anchor['defects']:
                    labels.append(d['class'])
            else:
                if anchor['object'] or anchor['grade']:
                    labels.append("No defect")
                else:
                    labels.append("Unassociated defect")
            
            if anchor['object']:
                main_box = anchor['object']['bbox']
            elif anchor['grade']:
                main_box = anchor['grade']['bbox']
            else:
                main_box = anchor['defects'][0]['bbox']
                
            x1, y1, x2, y2 = map(int, main_box)
            
            color = (0, 255, 0)
            if anchor['grade'] and anchor['grade']['class'] == 'Reject':
                color = (0, 0, 255)
            elif len(anchor['defects']) > 0:
                color = (0, 0, 255)
                
            cv2.rectangle(img_disp, (x1, y1), (x2, y2), color, 2)
            
            y_offset = y1 - 10
            for label in reversed(labels):
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img_disp, (x1, y_offset - text_h - 4), (x1 + text_w, y_offset + 4), color, -1)
                cv2.putText(img_disp, label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                y_offset -= (text_h + 8)

        out_dir = "runs/inference"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "integrated_result.jpg")
        cv2.imwrite(out_path, img_disp)
        
        return final_results, img_disp, out_path

    def print_report(self, final_results):
        print("========================================")
        print("ONION INSPECTION RESULT")
        print("========================================")
        print(f"Total onions detected: {len(final_results)}")
        print()
        
        for res in final_results:
            print("-" * 40)
            print(f"ONION #{res['onion_id']}")
            print("-" * 40)
            
            if res['grade']:
                print(f"Grade: {res['grade']['class']}")
                print(f"Grade Confidence: {res['grade']['confidence']*100:.0f}%")
            else:
                print("Grade: Not detected")
                
            if res['object']:
                print(f"Object: {res['object']['class']} (Conf: {res['object']['confidence']*100:.0f}%)")
            else:
                print("Object: Not detected")
                
            print("\nDefects:")
            if len(res['defects']) == 0:
                print("None")
            else:
                for d in res['defects']:
                    print(f"{d['class']} — {d['confidence']*100:.0f}%")
            print()
