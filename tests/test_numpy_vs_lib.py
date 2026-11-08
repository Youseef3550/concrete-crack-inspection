"""
PyTest Test Suite: NumPy From-Scratch Implementations vs OpenCV/Scikit-Image & Hand-Worked Examples.

Verifies:
1. Metrics (MSE, PSNR, IoU, Dice) against exact hand calculations.
2. Lossless Run-Length Encoding (RLE) round-trip compression.
3. 2D Convolution in NumPy vs scipy/cv2.
4. Histogram Equalization in NumPy vs cv2.equalizeHist.
5. Median Filter in NumPy vs cv2.medianBlur.
6. Alpha-Trimmed Mean Filter hand calculation.
7. Butterworth Filter transfer function hand calculation.
8. Sobel Gradient Magnitude in NumPy vs cv2.Sobel.
9. Otsu Threshold in pure NumPy vs cv2.threshold.
10. Morphological Erosion and Dilation in NumPy vs cv2.erode / cv2.dilate (np.array_equal).
11. Morphological Hole Filling in NumPy from scratch.
"""

import sys
import os
import pytest
import numpy as np
import cv2

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.metrics import (
    mse,
    psnr,
    iou,
    dice_coefficient,
    run_length_encode,
    run_length_decode,
    verify_hand_worked_metrics
)
from src.enhancement import (
    convolve2d_numpy,
    histogram_equalization_numpy,
    histogram_equalization_cv2,
    power_law_transform,
    verify_hand_worked_enhancement
)
from src.restoration import (
    median_filter_numpy,
    median_filter_cv2,
    alpha_trimmed_mean_filter_numpy,
    verify_hand_worked_restoration
)
from src.frequency import (
    butterworth_filter_numpy,
    verify_hand_worked_frequency
)
from src.segmentation import (
    sobel_magnitude_numpy,
    sobel_magnitude_cv2,
    otsu_threshold_numpy,
    otsu_threshold_cv2,
    verify_hand_worked_segmentation
)
from src.morphology import (
    erosion_numpy,
    dilation_numpy,
    hole_filling_numpy,
    verify_hand_worked_morphology
)


def test_hand_worked_metrics():
    """Validates metrics on 3x3 hand-computed patch."""
    assert verify_hand_worked_metrics() is True


def test_rle_lossless_roundtrip():
    """Validates lossless binary mask compression and exact decompression."""
    rng = np.random.RandomState(42)
    original_mask = (rng.rand(100, 100) > 0.85).astype(np.uint8) * 255
    encoded = run_length_encode(original_mask)
    decoded = run_length_decode(encoded, original_mask.shape)
    assert np.array_equal(original_mask, decoded * 255)


def test_hand_worked_enhancement():
    """Validates 2D convolution and point operations against hand calculations."""
    assert verify_hand_worked_enhancement() is True


def test_histogram_equalization_numpy_vs_cv2():
    """
    Validates NumPy histogram equalization against cv2.equalizeHist.
    Max absolute pixel difference must be <= 2 across 256 gray levels.
    """
    rng = np.random.RandomState(10)
    test_img = rng.randint(40, 210, size=(128, 128), dtype=np.uint8)

    eq_np, _ = histogram_equalization_numpy(test_img)
    eq_cv = cv2.equalizeHist(test_img)

    max_diff = np.max(np.abs(eq_np.astype(int) - eq_cv.astype(int)))
    assert max_diff <= 2, f"Histogram equalization discrepancy too high: {max_diff}"


def test_median_filter_numpy_vs_cv2():
    """
    Validates NumPy median filter against OpenCV cv2.medianBlur on interior pixels.
    """
    rng = np.random.RandomState(25)
    test_img = rng.randint(0, 256, size=(40, 40), dtype=np.uint8)

    med_np = median_filter_numpy(test_img, ksize=3, border_mode="reflect")
    med_cv = cv2.medianBlur(test_img, 3)

    # Check center region (ignoring boundary padding differences)
    np_interior = med_np[2:-2, 2:-2]
    cv_interior = med_cv[2:-2, 2:-2]
    assert np.array_equal(np_interior, cv_interior), "Interior median filter outputs do not match!"


def test_hand_worked_restoration():
    """Validates order-statistic restoration filters against hand calculations."""
    assert verify_hand_worked_restoration() is True


def test_hand_worked_frequency_transfer_function():
    """Validates Butterworth filter transfer function at center and cutoff D0."""
    assert verify_hand_worked_frequency() is True


def test_sobel_magnitude_numpy_vs_cv2():
    """
    Validates NumPy Sobel gradient magnitude against cv2.Sobel.
    """
    rng = np.random.RandomState(99)
    test_img = rng.randint(10, 240, size=(50, 50), dtype=np.uint8)

    mag_np, _, _ = sobel_magnitude_numpy(test_img)
    mag_cv = sobel_magnitude_cv2(test_img)

    # Allow tiny rounding difference of +/- 1
    max_diff = np.max(np.abs(mag_np.astype(int) - mag_cv.astype(int)))
    assert max_diff <= 1, f"Sobel magnitude difference exceeded tolerance: {max_diff}"


def test_otsu_threshold_numpy_vs_cv2():
    """
    Validates NumPy Otsu threshold value against cv2.threshold.
    """
    rng = np.random.RandomState(7)
    # Bimodal distribution
    cluster1 = rng.normal(60, 15, (60, 60))
    cluster2 = rng.normal(180, 15, (60, 60))
    bimodal = np.clip(np.concatenate([cluster1, cluster2], axis=1), 0, 255).astype(np.uint8)

    thresh_np, bin_np = otsu_threshold_numpy(bimodal)
    thresh_cv, bin_cv = otsu_threshold_cv2(bimodal)

    assert thresh_np == int(thresh_cv), f"Otsu threshold mismatch: {thresh_np} vs {thresh_cv}"
    assert np.array_equal(bin_np, bin_cv), "Otsu binary mask mismatch!"


def test_morphology_erosion_dilation_numpy_vs_cv2():
    """
    Validates NumPy erosion and dilation against cv2.erode and cv2.dilate using np.array_equal.
    """
    rng = np.random.RandomState(123)
    binary_img = (rng.rand(30, 30) > 0.5).astype(np.uint8) * 255
    se = np.ones((3, 3), dtype=np.uint8)

    # Test Erosion
    er_np = erosion_numpy(binary_img, se)
    er_cv = cv2.erode(binary_img, se, borderType=cv2.BORDER_CONSTANT)
    assert np.array_equal(er_np, er_cv), "NumPy erosion does not match cv2.erode!"

    # Test Dilation
    dil_np = dilation_numpy(binary_img, se)
    dil_cv = cv2.dilate(binary_img, se, borderType=cv2.BORDER_CONSTANT)
    assert np.array_equal(dil_np, dil_cv), "NumPy dilation does not match cv2.dilate!"


def test_hand_worked_morphology_and_hole_filling():
    """Validates morphology and hole filling on small 5x5 patches."""
    assert verify_hand_worked_morphology() is True


if __name__ == "__main__":
    pytest.main(["-v", __file__])
