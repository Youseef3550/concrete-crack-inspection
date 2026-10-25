"""
Evaluation Metrics Module for Digital Image Processing Project.

Implements MSE, PSNR, IoU (Jaccard Index), Dice Coefficient,
Precision, Recall, Crack Area Percentage, Count Error,
and Lossless Run-Length Encoding (RLE) Compression with Compression Ratio.
Includes hand-worked small array mathematical verifications.
"""

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Mean Squared Error (MSE) between two 2D or 3D image arrays.
    Formula: MSE = (1 / (M * N)) * sum((y_true - y_pred) ** 2)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float(np.mean((y_true - y_pred) ** 2))


def psnr(y_true: np.ndarray, y_pred: np.ndarray, max_val: float = 255.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Formula: PSNR = 10 * log10(max_val^2 / MSE)
    """
    err = mse(y_true, y_pred)
    if err == 0:
        return float("inf")
    return float(10.0 * np.log10((max_val ** 2) / err))


def iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU / Jaccard Index) for binary masks.
    Formula: IoU = |A ∩ B| / |A ∪ B|
    """
    mask_true = np.asarray(y_true > 0, dtype=bool)
    mask_pred = np.asarray(y_pred > 0, dtype=bool)

    intersection = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()

    if union == 0:
        # Both masks are completely empty (no crack) -> perfect agreement
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Dice Similarity Coefficient (F1-Score on pixels).
    Formula: Dice = (2 * |A ∩ B|) / (|A| + |B|)
    """
    mask_true = np.asarray(y_true > 0, dtype=bool)
    mask_pred = np.asarray(y_pred > 0, dtype=bool)

    intersection = np.logical_and(mask_true, mask_pred).sum()
    total = mask_true.sum() + mask_pred.sum()

    if total == 0:
        return 1.0
    return float((2.0 * intersection) / total)


def precision_recall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes pixel-level Precision, Recall, and F1-Score.
    """
    mask_true = np.asarray(y_true > 0, dtype=bool)
    mask_pred = np.asarray(y_pred > 0, dtype=bool)

    tp = np.logical_and(mask_true, mask_pred).sum()
    fp = np.logical_and(~mask_true, mask_pred).sum()
    fn = np.logical_and(mask_true, ~mask_pred).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn)
    }


def crack_area_percentage(mask: np.ndarray) -> float:
    """
    Computes percentage of cracked pixels relative to total surface area.
    Formula: (Cracked Pixels / Total Pixels) * 100
    """
    binary = np.asarray(mask > 0, dtype=bool)
    total_pixels = binary.size
    crack_pixels = binary.sum()
    if total_pixels == 0:
        return 0.0
    return float((crack_pixels / total_pixels) * 100.0)


def absolute_count_error(true_count: int, pred_count: int) -> int:
    """Absolute count error for object counting tasks."""
    return abs(int(true_count) - int(pred_count))


# =========================================================================
# Bonus: Lossless Run-Length Encoding (RLE) for Binary Masks & Compression Ratio
# =========================================================================

def run_length_encode(binary_mask: np.ndarray) -> list:
    """
    Encodes 2D binary mask using 1D flattened Run-Length Encoding (RLE).
    Returns list of pairs: [(value, run_length), ...].
    """
    flat = np.asarray(binary_mask > 0, dtype=np.uint8).flatten()
    if len(flat) == 0:
        return []

    runs = []
    current_val = flat[0]
    count = 1

    for val in flat[1:]:
        if val == current_val:
            count += 1
        else:
            runs.append((int(current_val), count))
            current_val = val
            count = 1
    runs.append((int(current_val), count))
    return runs


def run_length_decode(runs: list, shape: tuple) -> np.ndarray:
    """
    Decodes Run-Length Encoded pairs back to original 2D binary mask.
    """
    flat = []
    for val, count in runs:
        flat.extend([val] * count)
    return np.array(flat, dtype=np.uint8).reshape(shape)


def compute_compression_ratio(binary_mask: np.ndarray, rle_runs: list) -> dict:
    """
    Measures lossless compression metrics:
    - Original raw binary bits (1 bit per pixel or 8 bits per uint8 byte)
    - Compressed bits: each run stored as (1 bit for val + 16 bits for run length)
    - Compression Ratio = uncompressed_bits / compressed_bits
    """
    num_pixels = binary_mask.size
    uncompressed_bytes = num_pixels  # uint8 raw bytes
    # Compressed representation: 1 byte value + 2 bytes (uint16) length = 3 bytes per run
    compressed_bytes = len(rle_runs) * 3
    ratio = float(uncompressed_bytes / max(1, compressed_bytes))
    savings_percent = float((1.0 - (compressed_bytes / uncompressed_bytes)) * 100.0)

    return {
        "original_bytes": uncompressed_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": ratio,
        "space_savings_percent": savings_percent
    }


# =========================================================================
# Hand-worked Examples Verification (Section 3.4 requirement)
# =========================================================================

def verify_hand_worked_metrics():
    """
    Hand-worked verification on small 3x3 matrices:
    Image A (True):
    [[10, 20, 30],
     [40, 50, 60],
     [70, 80, 90]]

    Image B (Pred):
    [[10, 22, 30],
     [40, 48, 60],
     [70, 80, 93]]

    Differences: (22-20)^2 = 4, (48-50)^2 = 4, (93-90)^2 = 9. Total = 17.
    MSE = 17 / 9 = 1.888888...
    PSNR = 10 * log10(255^2 / (17/9)) = 10 * log10(65025 / 1.888888) = 45.3687 dB.

    Binary Masks for IoU:
    Mask A:
    [[1, 0, 0],
     [0, 1, 1],
     [0, 0, 1]]  -> |A| = 4

    Mask B:
    [[1, 1, 0],
     [0, 1, 0],
     [0, 0, 1]]  -> |B| = 4

    Intersection = (0,0), (1,1), (2,2) -> 3 pixels
    Union = (0,0), (0,1), (1,1), (1,2), (2,2) -> 5 pixels
    IoU = 3 / 5 = 0.600
    Dice = 2 * 3 / (4 + 4) = 6 / 8 = 0.750
    """
    img_a = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    img_b = np.array([[10, 22, 30], [40, 48, 60], [70, 80, 93]], dtype=np.uint8)

    calc_mse = mse(img_a, img_b)
    expected_mse = 17.0 / 9.0
    assert abs(calc_mse - expected_mse) < 1e-6, f"MSE mismatch: {calc_mse} vs {expected_mse}"

    calc_psnr = psnr(img_a, img_b)
    expected_psnr = 10.0 * np.log10(65025.0 / (17.0 / 9.0))
    assert abs(calc_psnr - expected_psnr) < 1e-4, f"PSNR mismatch: {calc_psnr} vs {expected_psnr}"

    mask_a = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]], dtype=np.uint8)
    mask_b = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)

    calc_iou = iou(mask_a, mask_b)
    expected_iou = 3.0 / 5.0
    assert abs(calc_iou - expected_iou) < 1e-6, f"IoU mismatch: {calc_iou} vs {expected_iou}"

    calc_dice = dice_coefficient(mask_a, mask_b)
    expected_dice = 6.0 / 8.0
    assert abs(calc_dice - expected_dice) < 1e-6, f"Dice mismatch: {calc_dice} vs {expected_dice}"

    # Verify RLE lossless round-trip
    runs = run_length_encode(mask_a)
    decoded = run_length_decode(runs, mask_a.shape)
    assert np.array_equal(mask_a, decoded), "RLE decoding failed round-trip equality"

    return True


if __name__ == "__main__":
    verify_hand_worked_metrics()
    print("All hand-worked mathematical tests passed successfully!")
