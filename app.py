import streamlit as st
import cv2
import sys
import os
from PIL import Image
import numpy as np

# Ensure scripts directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from integrated_inference import OnionInspector

st.set_page_config(page_title="AgriNovaa Onion Inspection", layout="wide")

st.title("🧅 Onion Quality Inspection")
st.write("Integrated system combining Defect Detection and Grading using IoU matching.")

@st.cache_resource
def load_inspector():
    # This loads both YOLO models and keeps them in memory
    return OnionInspector(
        defect_model_path="models/onion_defect_best.pt",
        grading_model_path="models/onion_grading_best.pt"
    )

inspector = load_inspector()

st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.25, 0.05)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    st.write("### Processing...")
    
    # Run inference directly with the PIL image (inspector handles conversion)
    results, img_disp, _ = inspector.inspect(source=image, conf_threshold=conf_threshold)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Detection Results")
        # OpenCV output is BGR, Streamlit needs RGB
        img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, use_container_width=True)
        
    with col2:
        st.subheader("Inspection Report")
        st.markdown(f"**Total onions:** {len(results)}")
        
        for res in results:
            with st.expander(f"Onion #{res['onion_id']}", expanded=True):
                # Grade
                if res['grade']:
                    st.markdown(f"**Grade:** {res['grade']['class']} ({res['grade']['confidence']*100:.0f}%)")
                else:
                    st.markdown("**Grade:** Not detected")
                    
                # Defect
                st.markdown("**Defects:**")
                if len(res['defects']) == 0:
                    st.write("None")
                else:
                    for d in res['defects']:
                        st.write(f"- {d['class']} ({d['confidence']*100:.0f}%)")
