import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from integrated_inference import OnionInspector

images = [
    r"d:\AgriNovaa\veg1-hcqsf-2.v4-demo_model.yolov11\test\images\400205_jpg.rf.6e14dcf3ea78796e827819cd4f321bec.jpg",
    r"d:\AgriNovaa\veg1-hcqsf-2.v4-demo_model.yolov11\test\images\20230121_090646_jpg.rf.0e2fba5695f1138ce4728c5548d8e6ab.jpg",
    r"d:\AgriNovaa\veg1-hcqsf-2.v4-demo_model.yolov11\test\images\download-5-_jpg.rf.3eca4c4946178a9bcdd3572880dd5cbb.jpg",
    r"d:\AgriNovaa\veg1-hcqsf-2.v4-demo_model.yolov11\test\images\images_jpg.rf.0b4182dc35ad48a19f8f70afc2cc42f4.jpg",
    r"d:\AgriNovaa\veg1-hcqsf-2.v4-demo_model.yolov11\test\images\images-10-_jpg.rf.4289a6b7a22bacd528f4fa95deb60348.jpg",
    r"d:\AgriNovaa\veg1-hcqsf-2.v4-demo_model.yolov11\test\images\WhatsApp-Image-2023-03-18-at-12-50-30-PM-5-_jpeg.rf.fffc4a0d31fabd3f7adc71cb14378c1c.jpg",
    r"d:\AgriNovaa\veg1-hcqsf-2.v4-demo_model.yolov11\test\images\WhatsApp-Image-2023-03-18-at-12-50-31-PM-1-_jpeg.rf.484118f36d80f3d4a217e455d45b9b08.jpg"
]

print("Initializing models...")
inspector = OnionInspector()

for img in images:
    filename = os.path.basename(img)
    print(f"\n--- Analyzing {filename} ---")
    results, _, _ = inspector.inspect(source=img, conf_threshold=0.25)
    if not results:
        print("No onions or defects detected.")
    
    for res in results:
        grade = res['grade']['class'] if res['grade'] else "Unknown Grade"
        defects = [d['class'] for d in res['defects']]
        defect_str = ", ".join(defects) if defects else "No defects"
        print(f"Onion #{res['onion_id']}: {grade} | Defects: {defect_str}")

