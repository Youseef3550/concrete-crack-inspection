"""
Streamlit Live Interactive Demo: Concrete Crack Inspection System.

Interactive features:
- Image selector (Dev set, Held-out Test set, Worst Failure Cases, or User Upload).
- Dynamic pipeline switcher (Baselines, Pipeline A, Pipeline B, Pipeline C).
- Interactive parameter sliders for real-time DIP tuning.
- Intermediate image inspector displaying all processing stages.
- Quantitative metrics, Structural Health Verdict (PASS/MONITOR/FAIL), and Lossless RLE Compression ratio.
"""

import os
import sys
import glob
import numpy as np
import cv2
import streamlit as st

# Setup paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data import load_image, load_mask
from src.metrics import (
    iou,
    dice_coefficient,
    crack_area_percentage,
    run_length_encode,
    compute_compression_ratio
)
from src.pipeline import (
    run_trivial_baseline,
    run_simple_baseline,
    run_pipeline_a,
    run_pipeline_b,
    run_pipeline_c
)

st.set_page_config(
    page_title="Concrete Crack Inspection DIP System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Concrete Crack Inspection & Structural Health Assessment")
st.markdown(
    "**Alamein International University (AIU) - CSE281 Digital Image Processing** | "
    "*100% Classical DIP Pipeline (No Machine Learning / No Neural Networks)*"
)

# Sidebar: Controls
st.sidebar.header("⚙️ Configuration & Controls")

# 1. Dataset Image Selection
images_dir = os.path.join(BASE_DIR, "data", "raw", "images")
masks_dir = os.path.join(BASE_DIR, "data", "raw", "masks")
available_images = sorted(glob.glob(os.path.join(images_dir, "*.*")))

sample_choice = st.sidebar.selectbox(
    "Select Image Source",
    ["Dataset Sample", "Upload Custom Image"]
)

raw_img = None
gt_mask = None
selected_name = "custom"

if sample_choice == "Dataset Sample" and available_images:
    image_names = [os.path.basename(p) for p in available_images]
    selected_name = st.sidebar.selectbox("Choose Image File", image_names, index=0)
    img_path = os.path.join(images_dir, selected_name)
    raw_img = load_image(img_path, as_gray=True)

    mask_path = os.path.join(masks_dir, f"{os.path.splitext(selected_name)[0]}.png")
    if os.path.exists(mask_path):
        gt_mask = load_mask(mask_path)
elif sample_choice == "Upload Custom Image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload concrete surface image",
        type=["jpg", "png", "jpeg", "bmp", "webp"]
    )
    if uploaded_file is not None:
        try:
            from PIL import Image
            pil_img = Image.open(uploaded_file)
            # Auto-resize large camera photos to max 640px for fast sub-second DIP execution
            max_dim = 640
            if max(pil_img.size) > max_dim:
                scale = max_dim / max(pil_img.size)
                new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
                pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

            raw_img = np.array(pil_img.convert("L"), dtype=np.uint8)
            selected_name = uploaded_file.name
            st.sidebar.success(f"Loaded: {selected_name} ({raw_img.shape[1]}x{raw_img.shape[0]} px)")
        except Exception as e:
            st.sidebar.error(f"Failed to decode uploaded image: {e}")

if raw_img is None:
    st.info("Please select or upload an image to begin.")
    st.stop()

# 2. Pipeline Selection
pipeline_choice = st.sidebar.selectbox(
    "Select Image Processing Pipeline",
    [
        "Pipeline C (Final Advanced Production Pipeline)",
        "Pipeline B (Frequency Restoration + Adaptive Thresh)",
        "Pipeline A (Spatial Enhancement + Otsu)",
        "Simple Baseline (Otsu + 3x3 Opening)",
        "Trivial Baseline (Global Thresh 127)"
    ]
)

# 3. Interactive Parameters
st.sidebar.subheader("🎛️ Interactive DIP Parameters")
gamma = st.sidebar.slider("Gamma Correction (Power-law)", 0.3, 2.0, 0.85, 0.05)
d0 = st.sidebar.slider("Butterworth Cutoff D0 (px)", 10.0, 100.0, 45.0, 5.0)
block_size = st.sidebar.slider("Adaptive Threshold Block Size", 7, 35, 17, step=2)
c_thresh = st.sidebar.slider("Adaptive Constant C", 1.0, 10.0, 4.5, 0.5)
closing_ksize = st.sidebar.slider("Morphological Closing Kernel Size", 3, 11, 5, step=2)
min_obj_area = st.sidebar.slider("Small Object Filter Area (Pore Removal)", 10, 150, 40, 5)

config = {
    "gamma": gamma,
    "d0": d0,
    "order": 2,
    "block_size": block_size,
    "c": c_thresh,
    "closing_ksize": closing_ksize,
    "min_object_area": min_obj_area,
    "safe_thresh_pct": 0.20,
    "critical_thresh_pct": 1.20
}

# Run selected pipeline
if "Pipeline C" in pipeline_choice:
    result = run_pipeline_c(raw_img, config=config)
elif "Pipeline B" in pipeline_choice:
    result = run_pipeline_b(raw_img, config=config)
elif "Pipeline A" in pipeline_choice:
    result = run_pipeline_a(raw_img, config=config)
elif "Simple Baseline" in pipeline_choice:
    result = run_simple_baseline(raw_img)
else:
    result = run_trivial_baseline(raw_img)

pred_mask = result["pred_mask"]
stages = result["intermediate_images"]
exec_time_ms = result["total_time_ms"]

# Calculate quantitative metrics
pred_area_pct = crack_area_percentage(pred_mask)
rle_runs = run_length_encode(pred_mask)
comp_metrics = compute_compression_ratio(pred_mask, rle_runs)

calc_iou = iou(gt_mask, pred_mask) if gt_mask is not None else None
calc_dice = dice_coefficient(gt_mask, pred_mask) if gt_mask is not None else None

# Key Metric Cards
st.subheader("📊 Inspection Dashboard & Telemetry")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Execution Time", f"{exec_time_ms:.1f} ms")

with col2:
    st.metric("Cracked Surface Area", f"{pred_area_pct:.3f} %")

with col3:
    if calc_iou is not None:
        st.metric("Test IoU vs Ground Truth", f"{calc_iou:.4f}")
    else:
        st.metric("Test IoU", "N/A (No Mask)")

with col4:
    if "structural_health" in result:
        verdict = result["structural_health"]["decision"]
        color = "red" if verdict == "FAIL" else ("orange" if verdict == "MONITOR" else "green")
        st.markdown(f"**Structural Verdict:**  \n### :{color}[{verdict}]")
    else:
        st.metric("Structural Verdict", "N/A (Baseline)")

with col5:
    st.metric(
        "RLE Compression Ratio",
        f"{comp_metrics['compression_ratio']:.2f}x",
        f"-{comp_metrics['space_savings_percent']:.1f}% space"
    )

st.divider()

# Tabbed Visualizer
tab1, tab2, tab3 = st.tabs([
    "🖼️ Intermediate Stage Breakdown",
    "🎯 Ground Truth & Error Analysis",
    "⚡ Stage Timings & Structural Report"
])

with tab1:
    st.markdown("### Step-by-Step Classical Pipeline Progression")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(raw_img, caption="1. Raw Input Grayscale", use_column_width=True)
    with c2:
        if "enhanced" in stages:
            st.image(stages["enhanced"], caption="2. Spatial Contrast Enhanced (Gamma)", use_column_width=True)
        else:
            st.info("Stage not in selected pipeline")
    with c3:
        if "spectrum" in stages:
            st.image(stages["spectrum"], caption="3. 2D Centered Fourier Log Spectrum", use_column_width=True)
        else:
            st.info("Stage not in selected pipeline")

    c4, c5, c6 = st.columns(3)
    with c4:
        if "raw_segmented" in stages:
            st.image(stages["raw_segmented"], caption="4. Adaptive Gaussian Thresholded", use_column_width=True)
        elif "binary" in stages:
            st.image(stages["binary"], caption="4. Thresholded Binary Mask", use_column_width=True)
        else:
            st.image(pred_mask, caption="4. Pred Mask", use_column_width=True)
    with c5:
        st.image(pred_mask, caption="5. Final Morphologically Cleaned Mask", use_column_width=True)
    with c6:
        if "overlay" in result:
            st.image(result["overlay"], caption="6. Overlay: Detected Cracks in Red", use_column_width=True)
        else:
            overlay_default = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
            overlay_default[pred_mask > 0] = [255, 0, 0]
            st.image(overlay_default, caption="6. Color Overlay", use_column_width=True)

with tab2:
    if gt_mask is not None:
        st.markdown("### Ground Truth vs Segmentation Error Map")
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            st.image(gt_mask, caption="Ground Truth Mask", use_column_width=True)
        with ec2:
            st.image(pred_mask, caption="Pipeline Predicted Mask", use_column_width=True)
        with ec3:
            # Color-coded error map:
            # Green = True Positive (TP), Red = False Positive (FP), Blue = False Negative (FN)
            tp = (gt_mask > 0) & (pred_mask > 0)
            fp = (gt_mask == 0) & (pred_mask > 0)
            fn = (gt_mask > 0) & (pred_mask == 0)

            error_map = np.zeros((*raw_img.shape, 3), dtype=np.uint8)
            error_map[tp] = [0, 255, 0]    # Green: Correct Detection
            error_map[fp] = [255, 0, 0]    # Red: False Positive (Noise)
            error_map[fn] = [0, 100, 255]  # Orange/Blue: Missed Crack

            st.image(error_map, caption="Error Map: Green=TP, Red=FP, Orange=FN", use_column_width=True)
            st.markdown(
                f"- **IoU:** `{calc_iou:.4f}` | **Dice:** `{calc_dice:.4f}`  \n"
                f"- **True Positive Pixels:** `{tp.sum():,}` | **False Positives:** `{fp.sum():,}` | **False Negatives:** `{fn.sum():,}`"
            )
    else:
        st.info("Ground truth mask not available for custom uploaded image.")

with tab3:
    st.markdown("### Processing Latency & Civil Engineering Assessment")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("#### Execution Latency Breakdown (ms)")
        timings = result.get("stage_timings_ms", {})
        for stage, t in timings.items():
            st.write(f"- **{stage}**: `{t:.2f} ms`")
        st.write(f"**Total Pipeline Time**: `{exec_time_ms:.2f} ms`")

    with tc2:
        if "structural_health" in result:
            health = result["structural_health"]
            st.markdown("#### Civil Engineering Assessment")
            st.write(f"- **Condition Status:** `{health['severity']}`")
            st.write(f"- **Major Crack Segments:** `{health['num_crack_components']}`")
            st.write(f"- **Maximum Crack Length:** `{health['max_crack_length_px']} px`")
            st.write(f"- **Action Required:** `{health['recommended_action']}`")
