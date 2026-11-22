import streamlit as st
import cv2
import sys
import os
import time
from PIL import Image
import numpy as np

# Ensure scripts directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from integrated_inference import OnionInspector

st.set_page_config(
    page_title="OnionVision AI | Smart India Hackathon",
    page_icon="🧅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling to match OnionVision AI SIH Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #4a152d 0%, #6d1f40 50%, #83274f 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(109, 31, 64, 0.25);
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(8px);
        color: #fdf2f6;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        margin-bottom: 0.9rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        color: #ffffff !important;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #fad2e1;
        max-width: 800px;
        line-height: 1.5;
        margin: 0;
    }

    /* Cards & Containers */
    .assessment-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Badges & Tags */
    .verdict-tag {
        display: inline-block;
        padding: 0.45rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }
    .tag-pass {
        background: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }
    .tag-warn {
        background: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
    }
    .tag-fail {
        background: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }

    /* Parameter Bars */
    .param-group {
        margin-bottom: 1rem;
        padding: 0.75rem;
        background: #f8fafc;
        border-radius: 8px;
        border: 1px solid #f1f5f9;
    }
    .param-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.35rem;
    }
    .param-bar-bg {
        width: 100%;
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
    }
    .param-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.4s ease;
    }
    .param-desc {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-pill">🧅 Smart India Hackathon (SIH) &bull; Computer Vision &amp; Agriculture</div>
    <div class="hero-title">OnionVision AI &mdash; Quality Assessment System</div>
    <p class="hero-subtitle">
        Automated non-destructive onion grading and defect identification using dual YOLOv11 models. 
        Detects sprouting, basal fungal rot, skin integrity, and predicts mandi storage life in real-time.
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_inspector():
    return OnionInspector(
        defect_model_path="models/onion_defect_best.pt",
        grading_model_path="models/onion_grading_best.pt"
    )

with st.spinner("Initializing YOLO11 Neural Networks..."):
    inspector = load_inspector()

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/onion.png", width=64)
    st.title("Settings & Presets")
    
    st.markdown("### ⚙️ Inspection Parameters")
    conf_threshold = st.slider("Detection Confidence Threshold", 0.05, 1.00, 0.25, 0.05)
    
    st.markdown("---")
    st.markdown("### 🧪 Quick Sample Presets")
    preset_choice = st.radio(
        "Choose an image source:",
        ("Upload My Own Image", "Grade A (Export Quality)", "Sprouted Defect", "Basal Rot / Mold Defect"),
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    **Project Scope:**
    * **Model Architecture:** YOLO11s Dual Pipeline
    * **Task:** Defect Segmentation & APMC Mandi Grading
    * **SIH Problem:** Post-Harvest Storage Loss Mitigation
    """)

# Load image based on choice
image_to_process = None
sample_label = None

if preset_choice == "Upload My Own Image":
    uploaded_file = st.file_uploader("Upload Onion Image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file).convert('RGB')
        sample_label = uploaded_file.name
else:
    sample_paths = {
        "Grade A (Export Quality)": "frontendsih/frontendsih/assets/sample-grade-a.jpg",
        "Sprouted Defect": "frontendsih/frontendsih/assets/sample-sprouted.jpg",
        "Basal Rot / Mold Defect": "frontendsih/frontendsih/assets/sample-mold.jpg"
    }
    path = sample_paths.get(preset_choice)
    if path and os.path.exists(path):
        image_to_process = Image.open(path).convert('RGB')
        sample_label = preset_choice
    else:
        st.error(f"Sample asset not found at {path}")

# Process and Display
if image_to_process is not None:
    start_time = time.time()
    with st.spinner("Running YOLO11 Dual-Model Inference & IoU Matching..."):
        results, img_disp, _ = inspector.inspect(source=image_to_process, conf_threshold=conf_threshold)
    inference_time = round((time.time() - start_time) * 1000)

    col_img, col_metrics = st.columns([1.1, 1], gap="large")

    with col_img:
        st.markdown(f"### 📷 Visual Detection Results")
        img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, use_container_width=True)
        
        st.caption(f"⚡ Inferred {len(results)} onion bulb(s) in **{inference_time} ms** using YOLO11s (CPU)")

    with col_metrics:
        st.markdown("### 📊 Commercial Quality Assessment")

        if len(results) == 0:
            st.warning("⚠️ No onion detected with current confidence threshold. Try lowering the threshold in the sidebar.")
        else:
            # Aggregate batch statistics
            grade_hierarchy = {'Reject': 4, 'Class 2': 3, 'Class 1': 2, 'Extra Class': 1}
            worst_grade = 'Extra Class'
            highest_sev = 0
            worst_conf = 0
            
            has_sprout = False
            has_rotten = False
            has_double_split = False
            all_defects = set()

            for res in results:
                g_class = res['grade']['class'] if res['grade'] else 'Class 2'
                g_conf = res['grade']['confidence'] if res['grade'] else 0.5
                sev = grade_hierarchy.get(g_class, 2)
                
                if sev > highest_sev:
                    highest_sev = sev
                    worst_grade = g_class
                    worst_conf = g_conf
                    
                for d in res['defects']:
                    d_cls = d['class']
                    all_defects.add(d_cls)
                    if 'sprout' in d_cls.lower():
                        has_sprout = True
                    if 'rotten' in d_cls.lower() or 'mold' in d_cls.lower():
                        has_rotten = True
                    if 'double' in d_cls.lower() or 'split' in d_cls.lower():
                        has_double_split = True

            # Determine APMC Mandi Verdict
            if worst_grade == 'Reject' or has_rotten:
                verdict_text = "Rejected / Quarantine Required"
                verdict_class = "tag-fail"
                verdict_desc = "Bulb exhibits severe pathology (fungal decay / rot) or unmarketable deformation. Immediate culling advised to prevent batch contagion."
                shelf_life = "0 - 3 Days (Immediate Cull)"
                shelf_fill = "#ef4444"
                shelf_pct = 10
            elif worst_grade == 'Class 2' or has_sprout or has_double_split:
                verdict_text = "Substandard / Commercial Culinary Use"
                verdict_class = "tag-warn"
                verdict_desc = "Minor physiological defects or early shoot emergence. Not suitable for long-term storage; expedite distribution for food processing."
                shelf_life = "7 - 14 Days (Short Storage)"
                shelf_fill = "#f59e0b"
                shelf_pct = 35
            else:
                verdict_text = "Passes APMC Mandi Benchmark (Grade A)"
                verdict_class = "tag-pass"
                verdict_desc = "Firm, dry scales with tightly closed apical necks. Approved for premium export packaging and ventilated cold storage."
                shelf_life = "60 - 75 Days (Optimal Storage)"
                shelf_fill = "#10b981"
                shelf_pct = 90

            # Mandi Verdict Card
            st.markdown(f"""
            <div class="assessment-card">
                <div class="verdict-tag {verdict_class}">⚖️ {verdict_text}</div>
                <div style="font-size:1.4rem; font-weight:800; color:#0f172a; margin-bottom:0.2rem;">
                    Batch Grade: <span style="color:#6d1f40;">{worst_grade}</span> 
                    <span style="font-size:0.95rem; color:#64748b; font-weight:500;">({worst_conf*100:.1f}% confidence)</span>
                </div>
                <div style="font-size:0.88rem; color:#475569; line-height:1.4; margin-top:0.4rem;">
                    {verdict_desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Physiological Quality Parameters Card
            st.markdown("""
            <div class="assessment-card">
                <div class="card-title">🔬 Physiological Parameter Breakdown</div>
            """, unsafe_allow_html=True)

            # 1. Skin scale integrity
            skin_val = "Double Split Deformity" if has_double_split else "Intact Protective Scales"
            skin_pct = 40 if has_double_split else 95
            skin_fill = "#f59e0b" if has_double_split else "#10b981"
            st.markdown(f"""
            <div class="param-group">
                <div class="param-header">
                    <span>🧅 Outer Skin Integrity</span>
                    <span style="color:{skin_fill}">{skin_val}</span>
                </div>
                <div class="param-bar-bg"><div class="param-bar-fill" style="width:{skin_pct}%; background:{skin_fill};"></div></div>
                <div class="param-desc">Evaluates tunic uniformity and structural bulb morphology.</div>
            </div>
            """, unsafe_allow_html=True)

            # 2. Sprout emergence
            sprout_val = "Active Shoot Emergence" if has_sprout else "None Detected (Closed Neck)"
            sprout_pct = 85 if has_sprout else 5
            sprout_fill = "#f59e0b" if has_sprout else "#10b981"
            st.markdown(f"""
            <div class="param-group">
                <div class="param-header">
                    <span>🌱 Sprout Emergence Risk</span>
                    <span style="color:{sprout_fill}">{sprout_val}</span>
                </div>
                <div class="param-bar-bg"><div class="param-bar-fill" style="width:{sprout_pct}%; background:{sprout_fill};"></div></div>
                <div class="param-desc">Detects dormant or active vegetative shoot emergence at the apical neck.</div>
            </div>
            """, unsafe_allow_html=True)

            # 3. Mold / Basal Rot
            mold_val = "Fungal Rot Detected" if has_rotten else "Clean / No Fungal Defect"
            mold_pct = 95 if has_rotten else 0
            mold_fill = "#ef4444" if has_rotten else "#10b981"
            st.markdown(f"""
            <div class="param-group">
                <div class="param-header">
                    <span>🍄 Basal Rot &amp; Fungal Mold</span>
                    <span style="color:{mold_fill}">{mold_val}</span>
                </div>
                <div class="param-bar-bg"><div class="param-bar-fill" style="width:{mold_pct}%; background:{mold_fill};"></div></div>
                <div class="param-desc">Detects Aspergillus niger, bacterial soft rot, and basal plate decay.</div>
            </div>
            """, unsafe_allow_html=True)

            # 4. Storage Shelf Life
            st.markdown(f"""
            <div class="param-group">
                <div class="param-header">
                    <span>⏳ Estimated Storage Shelf-Life</span>
                    <span style="color:{shelf_fill}">{shelf_life}</span>
                </div>
                <div class="param-bar-bg"><div class="param-bar-fill" style="width:{shelf_pct}%; background:{shelf_fill};"></div></div>
                <div class="param-desc">Empirical rule-based storage longevity projection for APMC mandi lots.</div>
            </div>
            </div>
            """, unsafe_allow_html=True)

            # Multi-onion individual inspector
            st.markdown("#### 📋 Individual Bulb Inspection Ledger")
            for res in results:
                g_str = f"{res['grade']['class']} ({res['grade']['confidence']*100:.0f}%)" if res['grade'] else "Not Classified"
                d_str = ", ".join([f"{d['class']} ({d['confidence']*100:.0f}%)" for d in res['defects']]) if res['defects'] else "No defects"
                
                with st.expander(f"Bulb #{res['onion_id']} &bull; Grade: {g_str}", expanded=False):
                    st.write(f"**Identified Defects:** {d_str}")
                    if res['object']:
                        st.write(f"**Object Detection Confidence:** {res['object']['confidence']*100:.1f}%")
else:
    st.info("👆 Please select a sample preset from the sidebar or upload an image to begin assessment.")
