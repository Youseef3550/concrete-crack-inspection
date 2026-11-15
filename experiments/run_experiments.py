"""
Experiment Tracking, Systematic Parameter Tuning, and Ablation Studies.

Performs:
1. Systematic One-Factor-at-a-Time (OFAT) Tuning:
   - Smoothing kernel size (3, 5, 7)
   - Cutoff frequency D0 (25.0, 45.0, 65.0)
   - Threshold methods (Global 127, Otsu, Adaptive Gaussian)
   - Structuring-element size (3, 5, 7) and shape ('rect', 'ellipse')
2. Five Controlled Ablation Studies:
   - Ablation 1: With vs Without Spatial Contrast Enhancement
   - Ablation 2: Spatial (Gaussian) vs Frequency-Domain (Butterworth) Smoothing
   - Ablation 3: Mean vs Median Filter under Salt-and-Pepper Noise
   - Ablation 4: Global vs Otsu vs Adaptive Gaussian Thresholding
   - Ablation 5: With vs Without Morphological Refinement
3. Logs every single run into experiments/experiment_log.csv.
"""

import os
import sys
import datetime
import time
from typing import Dict, List
import numpy as np
import pandas as pd
import cv2

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data import load_dataset, train_test_split, add_salt_and_pepper_noise
from src.metrics import iou, psnr, mse
from src.enhancement import power_law_transform, contrast_stretching, gaussian_filter_numpy, mean_filter_numpy
from src.frequency import butterworth_filter_numpy, apply_frequency_filter
from src.restoration import arithmetic_mean_filter, median_filter_numpy
from src.segmentation import (
    global_threshold,
    otsu_threshold_numpy,
    adaptive_threshold_gaussian_numpy,
    sobel_magnitude_cv2
)
from src.morphology import (
    get_structuring_element,
    remove_small_objects_numpy,
    hole_filling_numpy
)

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
EXP_DIR = os.path.join(BASE_DIR, "experiments")
LOG_PATH = os.path.join(EXP_DIR, "experiment_log.csv")


class ExperimentTracker:
    def __init__(self, log_path: str = LOG_PATH):
        self.log_path = log_path
        self.records = []
        self.run_counter = 1

    def log_run(
        self,
        author: str,
        pipeline: str,
        stage: str,
        method: str,
        params: dict,
        metric_name: str,
        metric_val: float,
        processing_time_ms: float,
        seed: int = 42
    ):
        record = {
            "run_id": f"RUN_{self.run_counter:04d}",
            "date": datetime.date.today().isoformat(),
            "author": author,
            "pipeline": pipeline,
            "stage": stage,
            "method": method,
            "parameters": str(params),
            "random_seed": seed,
            "processing_time_ms": round(processing_time_ms, 2),
            "metric_name": metric_name,
            "metric_value": round(metric_val, 4)
        }
        self.records.append(record)
        self.run_counter += 1

    def save(self):
        df = pd.DataFrame(self.records)
        df.to_csv(self.log_path, index=False)
        print(f"Logged {len(df)} experimental runs to {self.log_path}")
        return df


def run_tuning_and_ablations():
    os.makedirs(EXP_DIR, exist_ok=True)
    tracker = ExperimentTracker()

    print("Loading development dataset for tuning and ablations...")
    dataset = load_dataset(os.path.join(DATA_DIR, "images"), os.path.join(DATA_DIR, "masks"))
    dev_set, _ = train_test_split(dataset, split_ratio=0.70, random_seed=42)
    print(f"Loaded {len(dev_set)} development images.")

    # =========================================================================
    # Part 1: Systematic One-Factor-at-a-Time (OFAT) Parameter Tuning
    # =========================================================================
    print("\n--- Part 1: Systematic Parameter Tuning ---")

    # 1. Tuning Cutoff Frequency D0 (25.0, 45.0, 65.0)
    print("Tuning Butterworth Cutoff Frequency D0...")
    for d0 in [25.0, 45.0, 65.0]:
        t0 = time.perf_counter()
        ious = []
        for s in dev_set:
            img = s["image"]
            H = butterworth_filter_numpy(img.shape, d0=d0, order=2, filter_type="low")
            restored, _ = apply_frequency_filter(img, H)
            bin_mask = adaptive_threshold_gaussian_numpy(restored, block_size=17, c=4.5, invert=True)
            ious.append(iou(s["mask"], bin_mask))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(dev_set)
        tracker.log_run(
            author="Tech Lead",
            pipeline="Pipeline B",
            stage="Frequency Restoration",
            method="Butterworth Lowpass",
            params={"D0": d0, "order": 2},
            metric_name="Dev Mean IoU",
            metric_val=float(np.mean(ious)),
            processing_time_ms=elapsed_ms
        )

    # 2. Tuning Smoothing Kernel Size (3, 5, 7)
    print("Tuning Spatial Smoothing Kernel Size...")
    for ksize in [3, 5, 7]:
        t0 = time.perf_counter()
        ious = []
        for s in dev_set:
            img = s["image"]
            smoothed = gaussian_filter_numpy(img, kernel_size=ksize, sigma=1.0)
            bin_mask = adaptive_threshold_gaussian_numpy(smoothed, block_size=17, c=4.5, invert=True)
            ious.append(iou(s["mask"], bin_mask))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(dev_set)
        tracker.log_run(
            author="Enhancement Engineer",
            pipeline="Pipeline B",
            stage="Spatial Restoration",
            method="Gaussian Smoothing",
            params={"kernel_size": ksize, "sigma": 1.0},
            metric_name="Dev Mean IoU",
            metric_val=float(np.mean(ious)),
            processing_time_ms=elapsed_ms
        )

    # 3. Tuning Structuring-Element Size (3, 5, 7) and Shape ('rect', 'ellipse')
    print("Tuning Structuring Element Shape & Size...")
    for shape in ["rect", "ellipse"]:
        for ksize in [3, 5, 7]:
            t0 = time.perf_counter()
            ious = []
            se = get_structuring_element(shape, ksize)
            for s in dev_set:
                img = s["image"]
                H = butterworth_filter_numpy(img.shape, d0=45.0, order=2, filter_type="low")
                restored, _ = apply_frequency_filter(img, H)
                raw_bin = adaptive_threshold_gaussian_numpy(restored, block_size=17, c=4.5, invert=True)
                bridged = cv2.morphologyEx(raw_bin, cv2.MORPH_CLOSE, se)
                cleaned = remove_small_objects_numpy(bridged, min_size=40)
                ious.append(iou(s["mask"], cleaned))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(dev_set)
            tracker.log_run(
                author="Morphology Engineer",
                pipeline="Pipeline C",
                stage="Morphology Refinement",
                method="Morphological Closing",
                params={"shape": shape, "ksize": ksize},
                metric_name="Dev Mean IoU",
                metric_val=float(np.mean(ious)),
                processing_time_ms=elapsed_ms
            )

    # =========================================================================
    # Part 2: Controlled Ablation Studies
    # =========================================================================
    print("\n--- Part 2: Running 5 Controlled Ablation Studies ---")

    # Ablation 1: With vs Without Spatial Contrast Enhancement
    print("Ablation 1: Spatial Contrast Enhancement Impact...")
    for with_enh in [False, True]:
        t0 = time.perf_counter()
        ious = []
        for s in dev_set:
            img = s["image"]
            if with_enh:
                gamma_corr = power_law_transform(img, gamma=0.85)
                proc = contrast_stretching(gamma_corr, r_min=20, r_max=230)
            else:
                proc = img
            H = butterworth_filter_numpy(proc.shape, d0=45.0, order=2, filter_type="low")
            restored, _ = apply_frequency_filter(proc, H)
            raw_bin = adaptive_threshold_gaussian_numpy(restored, block_size=17, c=4.5, invert=True)
            cleaned = remove_small_objects_numpy(raw_bin, min_size=40)
            ious.append(iou(s["mask"], cleaned))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(dev_set)
        tracker.log_run(
            author="Enhancement Engineer",
            pipeline="Ablation 1",
            stage="Spatial Enhancement",
            method=f"With Enhancement = {with_enh}",
            params={"with_enhancement": with_enh, "gamma": 0.85 if with_enh else 1.0},
            metric_name="Dev Mean IoU",
            metric_val=float(np.mean(ious)),
            processing_time_ms=elapsed_ms
        )

    # Ablation 2: Spatial (Gaussian) vs Frequency-Domain (Butterworth) Smoothing
    print("Ablation 2: Spatial vs Frequency Domain Smoothing...")
    for mode in ["Spatial_Gaussian", "Frequency_Butterworth"]:
        t0 = time.perf_counter()
        ious = []
        for s in dev_set:
            img = s["image"]
            if mode == "Spatial_Gaussian":
                restored = gaussian_filter_numpy(img, kernel_size=5, sigma=1.0)
            else:
                H = butterworth_filter_numpy(img.shape, d0=45.0, order=2, filter_type="low")
                restored, _ = apply_frequency_filter(img, H)
            raw_bin = adaptive_threshold_gaussian_numpy(restored, block_size=17, c=4.5, invert=True)
            cleaned = remove_small_objects_numpy(raw_bin, min_size=40)
            ious.append(iou(s["mask"], cleaned))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(dev_set)
        tracker.log_run(
            author="Restoration Engineer",
            pipeline="Ablation 2",
            stage="Smoothing Domain",
            method=mode,
            params={"mode": mode},
            metric_name="Dev Mean IoU",
            metric_val=float(np.mean(ious)),
            processing_time_ms=elapsed_ms
        )

    # Ablation 3: Mean vs Median Filter under Salt-and-Pepper Noise
    print("Ablation 3: Mean vs Median under Salt-and-Pepper Noise...")
    for filt_name, filt_fn in [("Arithmetic_Mean", arithmetic_mean_filter), ("Median_Filter", median_filter_numpy)]:
        t0 = time.perf_counter()
        psnrs = []
        for s in dev_set[:30]:  # subset for speed
            clean = s["image"]
            noisy = add_salt_and_pepper_noise(clean, salt_prob=0.03, pepper_prob=0.03, seed=42)
            filtered = filt_fn(noisy, ksize=3)
            psnrs.append(psnr(clean, filtered))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / 30
        tracker.log_run(
            author="Restoration Engineer",
            pipeline="Ablation 3",
            stage="Noise Restoration",
            method=filt_name,
            params={"filter": filt_name, "noise": "Salt-and-Pepper 3%"},
            metric_name="PSNR (dB)",
            metric_val=float(np.mean(psnrs)),
            processing_time_ms=elapsed_ms
        )

    # Ablation 4: Global vs Otsu vs Adaptive Gaussian Thresholding
    print("Ablation 4: Thresholding Method Comparison...")
    for t_method in ["Global_127", "Otsu", "Adaptive_Gaussian"]:
        t0 = time.perf_counter()
        ious = []
        for s in dev_set:
            img = s["image"]
            if t_method == "Global_127":
                bin_mask = global_threshold(img, thresh=127, invert=True)
            elif t_method == "Otsu":
                _, bin_mask = otsu_threshold_numpy(img, invert=True)
            else:
                bin_mask = adaptive_threshold_gaussian_numpy(img, block_size=17, c=4.5, invert=True)
            ious.append(iou(s["mask"], bin_mask))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(dev_set)
        tracker.log_run(
            author="Segmentation Engineer",
            pipeline="Ablation 4",
            stage="Thresholding Method",
            method=t_method,
            params={"threshold_method": t_method},
            metric_name="Dev Mean IoU",
            metric_val=float(np.mean(ious)),
            processing_time_ms=elapsed_ms
        )

    # Ablation 5: With vs Without Morphological Refinement
    print("Ablation 5: Morphological Refinement Impact...")
    for with_morph in [False, True]:
        t0 = time.perf_counter()
        ious = []
        for s in dev_set:
            img = s["image"]
            H = butterworth_filter_numpy(img.shape, d0=45.0, order=2, filter_type="low")
            restored, _ = apply_frequency_filter(img, H)
            raw_bin = adaptive_threshold_gaussian_numpy(restored, block_size=17, c=4.5, invert=True)
            if with_morph:
                se = get_structuring_element("ellipse", 5)
                bridged = cv2.morphologyEx(raw_bin, cv2.MORPH_CLOSE, se)
                cleaned = remove_small_objects_numpy(bridged, min_size=40)
                final_m = hole_filling_numpy(cleaned, max_iter=200)
            else:
                final_m = raw_bin
            ious.append(iou(s["mask"], final_m))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(dev_set)
        tracker.log_run(
            author="Morphology Engineer",
            pipeline="Ablation 5",
            stage="Morphology Refinement",
            method=f"With Morphology = {with_morph}",
            params={"with_morphology": with_morph},
            metric_name="Dev Mean IoU",
            metric_val=float(np.mean(ious)),
            processing_time_ms=elapsed_ms
        )

    tracker.save()
    print("All tuning and ablation experiments completed successfully!")


if __name__ == "__main__":
    run_tuning_and_ablations()
