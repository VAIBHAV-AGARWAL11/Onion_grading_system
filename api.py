from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import cv2
import base64
import time
import numpy as np
from PIL import Image

# Ensure scripts directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from integrated_inference import OnionInspector

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

print("Initializing YOLO Models...")
# Initialize the inspector globally so models are loaded once
inspector = OnionInspector(
    defect_model_path="models/onion_defect_best_v2.pt",
    grading_model_path="models/onion_grading_best.pt"
)

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        # Read the image file
        image = Image.open(file).convert('RGB')
        
        # Run inference
        conf_threshold = float(request.form.get('conf_threshold', 0.25))
        
        start_time = time.time()
        results, img_disp, out_path = inspector.inspect(source=image, conf_threshold=conf_threshold)
        inference_time_ms = round((time.time() - start_time) * 1000, 2)
        
        # Encode the annotated image to base64
        _, buffer = cv2.imencode('.jpg', img_disp)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Format the results
        return jsonify({
            "success": True,
            "onions": results,
            "annotated_image_base64": img_base64,
            "metadata": {
                "model": "YOLO11s",
                "task": "Multi-Model Defect & Grading",
                "classes": "Grading: Class 1, Class 2, Extra Class, Reject | Defects: onion, double_split, rotten, sprout",
                "inference_time_ms": inference_time_ms,
                "image_size": f"{image.width}x{image.height}"
            }
        })
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
