import cv2
from scripts.integrated_inference import OnionInspector

inspector = OnionInspector('models/onion_defect_best_v2.pt', 'models/onion_grading_best.pt')
# We don't have their exact whatsapp image, let's just test any image or print the logic
print("Inspector loaded successfully.")
