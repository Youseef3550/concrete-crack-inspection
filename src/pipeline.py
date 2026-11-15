"""
Pipelines Architecture and Execution Engine.

Implements:
1. Trivial Baseline: Fixed global threshold at 127.
2. Simple Baseline: Otsu thresholding + 3x3 morphological opening.
3. Pipeline A: Spatial Enhancement (Gamma + Contrast Stretching) + Otsu Thresholding.
4. Pipeline B: Frequency/Restoration (Butterworth Lowpass) + Adaptive Gaussian Thresholding.
5. Pipeline C: Full Advanced Pipeline with Spatial Enhancement, Frequency Filtering,
   Adaptive Segmentation, Morphological Bridging & Hole Filling, Area Filtering,
   and Structural Damage Assessment Decision Rule.
"""

import time
from typing import Dict, Any, Optional
import numpy as np
import cv2

from src.enhancement import power_law_transform, contrast_stretching
from src.frequency import butterworth_filter_numpy, apply_frequency_filter
from src.segmentation import (
    global_threshold,
    otsu_threshold_numpy,
    adaptive_threshold_gaussian_numpy,
    sobel_magnitude_cv2
)
from src.morphology import (
    opening_numpy,
    closing_numpy,
    hole_filling_numpy,
    remove_small_objects_numpy,
    get_structuring_element,
    measure_and_classify_damage
)


# =========================================================================
# Baselines
# =========================================================================

def run_trivial_baseline(image: np.ndarray) -> Dict[str, Any]:
    """
    Trivial Baseline:
    Applies fixed global threshold of 127 on raw grayscale input.
    """
    t0 = time.perf_counter()
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Dark crack pixels (< 127) become foreground 255
    pred_mask = global_threshold(gray, thresh=127, invert=True)
    elapsed = time.perf_counter() - t0

    return {
        "pipeline_name": "Trivial Baseline (Global Thresh 127)",
        "pred_mask": pred_mask,
        "total_time_ms": elapsed * 1000.0,
        "stage_timings_ms": {"threshold": elapsed * 1000.0},
        "intermediate_images": {
            "input": gray,
            "threshold": pred_mask
        }
    }


def run_simple_baseline(image: np.ndarray) -> Dict[str, Any]:
    """
    Simple Baseline:
    Otsu thresholding followed by one 3x3 opening.
    """
    t0 = time.perf_counter()
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    t_thresh = time.perf_counter()
    _, raw_binary = otsu_threshold_numpy(gray, invert=True)
    time_thresh = time.perf_counter() - t_thresh

    t_morph = time.perf_counter()
    se = get_structuring_element("rect", ksize=3)
    opened = opening_numpy(raw_binary, se)
    time_morph = time.perf_counter() - t_morph

    total_time = time.perf_counter() - t0

    return {
        "pipeline_name": "Simple Baseline (Otsu + 3x3 Opening)",
        "pred_mask": opened,
        "total_time_ms": total_time * 1000.0,
        "stage_timings_ms": {
            "otsu_threshold": time_thresh * 1000.0,
            "opening_3x3": time_morph * 1000.0
        },
        "intermediate_images": {
            "input": gray,
            "otsu_binary": raw_binary,
            "opened": opened
        }
    }


# =========================================================================
# Pipelines A, B, C
# =========================================================================

def run_pipeline_a(image: np.ndarray, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Pipeline A:
    1. Spatial Contrast Enhancement (Gamma Correction + Contrast Stretching).
    2. Global Otsu Thresholding.
    """
    if config is None:
        config = {"gamma": 0.8, "r_min": 20, "r_max": 230}

    t0 = time.perf_counter()
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Stage 1: Spatial Enhancement
    t_enh = time.perf_counter()
    gamma_img = power_law_transform(gray, gamma=config.get("gamma", 0.8))
    enhanced = contrast_stretching(
        gamma_img, r_min=config.get("r_min", 20), r_max=config.get("r_max", 230)
    )
    time_enh = time.perf_counter() - t_enh

    # Stage 2: Segmentation
    t_seg = time.perf_counter()
    _, binary = otsu_threshold_numpy(enhanced, invert=True)
    time_seg = time.perf_counter() - t_seg

    total_time = time.perf_counter() - t0

    return {
        "pipeline_name": "Pipeline A (Spatial Enh + Otsu)",
        "pred_mask": binary,
        "total_time_ms": total_time * 1000.0,
        "stage_timings_ms": {
            "enhancement": time_enh * 1000.0,
            "segmentation": time_seg * 1000.0
        },
        "intermediate_images": {
            "input": gray,
            "enhanced": enhanced,
            "binary": binary
        }
    }


def run_pipeline_b(image: np.ndarray, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Pipeline B:
    1. Spatial Enhancement.
    2. Frequency-domain Butterworth Low-Pass filter (smooth granular texture).
    3. Local Adaptive Gaussian Thresholding.
    """
    if config is None:
        config = {"gamma": 0.8, "d0": 40.0, "order": 2, "block_size": 17, "c": 4.0}

    t0 = time.perf_counter()
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Stage 1: Spatial Enhancement
    t_enh = time.perf_counter()
    enhanced = power_law_transform(gray, gamma=config.get("gamma", 0.8))
    time_enh = time.perf_counter() - t_enh

    # Stage 2: Frequency Restoration
    t_freq = time.perf_counter()
    H_bw = butterworth_filter_numpy(
        gray.shape,
        d0=config.get("d0", 40.0),
        order=config.get("order", 2),
        filter_type="low"
    )
    restored, spectrum = apply_frequency_filter(enhanced, H_bw)
    time_freq = time.perf_counter() - t_freq

    # Stage 3: Adaptive Segmentation
    t_seg = time.perf_counter()
    binary = adaptive_threshold_gaussian_numpy(
        restored,
        block_size=config.get("block_size", 17),
        c=config.get("c", 4.0),
        invert=True
    )
    time_seg = time.perf_counter() - t_seg

    total_time = time.perf_counter() - t0

    return {
        "pipeline_name": "Pipeline B (Frequency Filter + Adaptive Thresh)",
        "pred_mask": binary,
        "total_time_ms": total_time * 1000.0,
        "stage_timings_ms": {
            "enhancement": time_enh * 1000.0,
            "frequency_filter": time_freq * 1000.0,
            "adaptive_thresh": time_seg * 1000.0
        },
        "intermediate_images": {
            "input": gray,
            "enhanced": enhanced,
            "spectrum": spectrum,
            "restored": restored,
            "binary": binary
        }
    }


def run_pipeline_c(image: np.ndarray, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Pipeline C (Final Advanced Production Pipeline):
    1. Preprocessing & Spatial Contrast Enhancement (Gamma 0.85 + Percentile Contrast Stretch).
    2. Frequency / Noise Restoration (Butterworth low-pass or Bilateral filter).
    3. Adaptive Gaussian Segmentation guided by Sobel Edge Magnitude.
    4. Morphological Refinement:
       - Directional/Elliptical Closing (bridges narrow hairline breaks in crack traces).
       - Morphological Hole Filling.
       - Small-Object Removal (Area Opening: eliminates pore noise < min_size).
    5. Structural Health Assessment & Decision Rule:
       - Connected component analysis, crack area percentage, max length,
         and safety verdict (PASS / MONITOR / FAIL).
    """
    if config is None:
        config = {
            "gamma": 0.80,
            "d0": 40.0,
            "order": 2,
            "block_size": 17,
            "c": 4.0,
            "closing_ksize": 3,
            "min_object_area": 20,
            "safe_thresh_pct": 0.20,
            "critical_thresh_pct": 1.20
        }

    t0 = time.perf_counter()
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Stage 1: Spatial Enhancement
    t1 = time.perf_counter()
    enhanced = power_law_transform(gray, gamma=config.get("gamma", 0.80))
    time_enh = (time.perf_counter() - t1) * 1000.0

    # Stage 2: Noise Restoration (Butterworth Lowpass)
    t2 = time.perf_counter()
    H_bw = butterworth_filter_numpy(
        gray.shape,
        d0=config.get("d0", 40.0),
        order=config.get("order", 2),
        filter_type="low"
    )
    restored, spectrum = apply_frequency_filter(enhanced, H_bw)
    time_rest = (time.perf_counter() - t2) * 1000.0

    # Stage 3: Segmentation (Local Adaptive Gaussian)
    t3 = time.perf_counter()
    adaptive_mask = adaptive_threshold_gaussian_numpy(
        restored,
        block_size=config.get("block_size", 17),
        c=config.get("c", 4.0),
        invert=True
    )
    segmented_raw = adaptive_mask
    time_seg = (time.perf_counter() - t3) * 1000.0

    # Stage 4: Morphological Refinement
    t4 = time.perf_counter()
    # 4a. Connect broken crack paths with closing
    se_close = get_structuring_element("ellipse", ksize=config.get("closing_ksize", 3))
    bridged = cv2.morphologyEx(segmented_raw, cv2.MORPH_CLOSE, se_close)

    # 4b. Remove isolated small pores / aggregate noise
    cleaned = remove_small_objects_numpy(
        bridged, min_size=config.get("min_object_area", 20)
    )

    final_mask = cleaned
    time_morph = (time.perf_counter() - t4) * 1000.0

    # Stage 5: Structural Measurement & Decision Rule
    t5 = time.perf_counter()
    health_assessment = measure_and_classify_damage(
        final_mask,
        safe_thresh_pct=config.get("safe_thresh_pct", 0.20),
        critical_thresh_pct=config.get("critical_thresh_pct", 1.20)
    )
    time_meas = (time.perf_counter() - t5) * 1000.0

    total_time = (time.perf_counter() - t0) * 1000.0

    # Overlay visualization for engineers
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    overlay[final_mask > 0] = [255, 0, 0]  # Red cracks on grayscale concrete

    return {
        "pipeline_name": "Pipeline C (Advanced Morphological Assessment)",
        "pred_mask": final_mask,
        "overlay": overlay,
        "structural_health": health_assessment,
        "total_time_ms": total_time,
        "stage_timings_ms": {
            "spatial_enhancement": time_enh,
            "restoration_frequency": time_rest,
            "segmentation": time_seg,
            "morphology_refinement": time_morph,
            "measurement_decision": time_meas
        },
        "intermediate_images": {
            "input": gray,
            "enhanced": enhanced,
            "spectrum": spectrum,
            "restored": restored,
            "raw_segmented": segmented_raw,
            "bridged": bridged,
            "cleaned": cleaned,
            "final_mask": final_mask,
            "overlay": overlay
        }
    }
