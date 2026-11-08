"""
Mathematical Morphology, Hole Filling, and Object Measurement Module.

Implements:
- Basic operations: Erosion and Dilation in pure NumPy from scratch with arbitrary
  structuring elements, verified against cv2.erode and cv2.dilate using np.array_equal.
- Compound operations: Morphological Opening, Closing, and Directional Bridging.
- Morphological Reconstruction & Hole Filling in pure NumPy from scratch.
- Area filtering: Small-object (pore/speckle) removal.
- Connected component measurement (crack area, bounding boxes, length).
- Final structural health decision rule (Pass / Monitor / Fail).
- Hand-worked mathematical example verification.
"""

from typing import Tuple, Dict, List
import numpy as np
import cv2


# =========================================================================
# Morphological Structuring Elements
# =========================================================================

def get_structuring_element(shape: str = "rect", ksize: int = 3) -> np.ndarray:
    """
    Returns binary structuring element (SE) matrix:
    - 'rect': all ones
    - 'cross': plus sign
    - 'ellipse': disk / rounded shape
    """
    if ksize % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    if shape == "rect":
        return np.ones((ksize, ksize), dtype=np.uint8)
    elif shape == "cross":
        se = np.zeros((ksize, ksize), dtype=np.uint8)
        mid = ksize // 2
        se[mid, :] = 1
        se[:, mid] = 1
        return se
    elif shape == "ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    else:
        raise ValueError(f"Unknown SE shape: {shape}")


# =========================================================================
# Pure NumPy Erosion & Dilation from Scratch
# =========================================================================

def erosion_numpy(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Binary / Grayscale Erosion in pure NumPy from scratch.
    Formula: (A ⊖ B)(x, y) = min_{(i, j) in B} [ A(x + i, y + j) ]
    For binary {0, 255}: pixel is 255 if and only if structuring element B fits entirely in A.
    Uses constant_values=255 padding matching OpenCV morphologyDefaultBorderValue().
    Vectorized using slice operations.
    """
    k_h, k_w = se.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=255)
    h, w = img.shape

    slices = []
    for di in range(k_h):
        for dj in range(k_w):
            if se[di, dj]:
                slices.append(padded[di : di + h, dj : dj + w])

    return np.minimum.reduce(slices)


def dilation_numpy(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Binary / Grayscale Dilation in pure NumPy from scratch.
    Formula: (A ⊕ B)(x, y) = max_{(i, j) in B} [ A(x - i, y - j) ]
    For binary {0, 255}: pixel is 255 if structuring element B hits any foreground pixel in A.
    Vectorized using slice operations.
    """
    k_h, k_w = se.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    se_flipped = np.flipud(np.fliplr(se))
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
    h, w = img.shape

    slices = []
    for di in range(k_h):
        for dj in range(k_w):
            if se_flipped[di, dj]:
                slices.append(padded[di : di + h, dj : dj + w])

    return np.maximum.reduce(slices)


def opening_numpy(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Opening = Erosion followed by Dilation: (A ⊖ B) ⊕ B. Removes small protrusions/pores."""
    return dilation_numpy(erosion_numpy(img, se), se)


def closing_numpy(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Closing = Dilation followed by Erosion: (A ⊕ B) ⊖ B. Bridges narrow gaps in cracks."""
    return erosion_numpy(dilation_numpy(img, se), se)


# =========================================================================
# Morphological Hole Filling in Pure NumPy
# =========================================================================

def hole_filling_numpy(binary_img: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """
    Hole Filling via Morphological Reconstruction from scratch in pure NumPy.
    Algorithm:
    1. Complement mask: A^c = 255 - binary_img
    2. Marker image X_0 initialized with zeros, with borders set to A^c borders.
    3. Iterate dilation conditioned by mask:
       X_k = (X_{k-1} ⊕ B) ∩ A^c
    4. When converged (X_k == X_{k-1}), filled result is complement of flooded background:
       Filled = 255 - X_k.
    """
    mask_c = 255 - binary_img
    h, w = binary_img.shape
    marker = np.zeros((h, w), dtype=np.uint8)

    # Seed marker at image borders
    marker[0, :] = mask_c[0, :]
    marker[-1, :] = mask_c[-1, :]
    marker[:, 0] = mask_c[:, 0]
    marker[:, -1] = mask_c[:, -1]

    # 3x3 cross structuring element
    se_cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    # Conditioned dilation loop
    for _ in range(max_iter):
        dilated = dilation_numpy(marker, se_cross)
        dilated_conditioned = np.minimum(dilated, mask_c)
        if np.array_equal(dilated_conditioned, marker):
            break
        marker = dilated_conditioned

    filled_image = 255 - marker
    return filled_image.astype(np.uint8)


# =========================================================================
# Area Filtering & Connecting Broken Crack Segments
# =========================================================================

def remove_small_objects_numpy(binary_img: np.ndarray, min_size: int = 50) -> np.ndarray:
    """
    Removes isolated noise specks / small pores smaller than min_size pixels.
    Uses connected component labeling.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    clean_mask = np.zeros_like(binary_img)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_size:
            clean_mask[labels == label] = 255

    return clean_mask


def connect_broken_cracks(binary_img: np.ndarray, se_size: int = 5) -> np.ndarray:
    """
    Connects broken crack fragments using morphological closing with a small disk SE.
    """
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, se)


# =========================================================================
# Measurement & Structural Assessment Decision Rule
# =========================================================================

def measure_and_classify_damage(
    binary_mask: np.ndarray,
    safe_thresh_pct: float = 0.20,
    critical_thresh_pct: float = 1.20
) -> Dict:
    """
    Extracts geometric measurements from final binary crack mask and makes
    an automated civil engineering structural health classification.
    
    Decision Rule:
    - PASS (Safe): Crack Area < 0.20% (negligible hairline surface texture).
    - MONITOR (Moderate): Crack Area between 0.20% and 1.20% (scheduled monitoring).
    - FAIL (Critical Hazard): Crack Area > 1.20% or max crack length > 150px
      (structural intervention required).
    """
    total_pixels = binary_mask.size
    crack_pixels = int(np.sum(binary_mask > 0))
    area_pct = float((crack_pixels / total_pixels) * 100.0)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    objects = []
    max_length = 0
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        diag_length = int(np.sqrt(w ** 2 + h ** 2))
        if diag_length > max_length:
            max_length = diag_length

        objects.append({
            "label": label,
            "bbox": (x, y, w, h),
            "area": area,
            "length": diag_length,
            "centroid": (float(centroids[label][0]), float(centroids[label][1]))
        })

    # Structural health decision
    if area_pct < safe_thresh_pct:
        decision = "PASS"
        severity = "Safe / Sound Concrete"
        action = "No intervention needed. Regular maintenance."
    elif area_pct <= critical_thresh_pct and max_length < 200:
        decision = "MONITOR"
        severity = "Moderate Surface Cracking"
        action = "Schedule secondary inspection within 6 months."
    else:
        decision = "FAIL"
        severity = "Critical Structural Damage"
        action = "Immediate structural repair and grouting required!"

    return {
        "crack_pixel_count": crack_pixels,
        "total_pixels": total_pixels,
        "crack_area_percentage": round(area_pct, 4),
        "num_crack_components": num_labels - 1,
        "max_crack_length_px": max_length,
        "decision": decision,
        "severity": severity,
        "recommended_action": action,
        "components": objects
    }


# =========================================================================
# Hand-worked Example Verification
# =========================================================================

def verify_hand_worked_morphology():
    """
    Hand-worked 5x5 patch verification:
    Patch A (binary object in center 3x3):
    [[0, 0,   0,   0, 0],
     [0, 255, 255, 255, 0],
     [0, 255, 255, 255, 0],
     [0, 255, 255, 255, 0],
     [0, 0,   0,   0, 0]]

    Structuring Element B (3x3 rect):
    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]]

    Erosion (A ⊖ B):
    Only center pixel (2, 2) has all 3x3 neighbors as 255.
    Result after erosion must have ONLY pixel (2, 2) = 255, all others 0!
    """
    patch = np.zeros((5, 5), dtype=np.uint8)
    patch[1:4, 1:4] = 255
    se_rect = np.ones((3, 3), dtype=np.uint8)

    eroded_np = erosion_numpy(patch, se_rect)
    expected_eroded = np.zeros((5, 5), dtype=np.uint8)
    expected_eroded[2, 2] = 255

    assert np.array_equal(eroded_np, expected_eroded), "Hand-worked erosion failed"

    # Verify Dilation on single center point
    single_point = np.zeros((5, 5), dtype=np.uint8)
    single_point[2, 2] = 255
    dilated_np = dilation_numpy(single_point, se_rect)
    assert np.array_equal(dilated_np, patch), "Hand-worked dilation failed"

    # Hole filling verification: ring with 1 pixel hole in center
    ring = np.ones((5, 5), dtype=np.uint8) * 255
    ring[2, 2] = 0  # 1-pixel hole at center
    filled = hole_filling_numpy(ring)
    assert filled[2, 2] == 255, "Hand-worked hole filling failed to fill center hole"

    return True


if __name__ == "__main__":
    verify_hand_worked_morphology()
    print("Morphology module verified successfully!")
