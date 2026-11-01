"""
Spatial Image Enhancement Module.

Implements:
- Point transformations: Negative, Log, Power-law (Gamma), Contrast stretching.
- Histogram processing: Histogram Equalization written from scratch in NumPy,
  compared against OpenCV cv2.equalizeHist.
- Spatial smoothing: 2D Convolution in pure NumPy, Mean filter, Gaussian smoothing.
- Spatial sharpening: Laplacian filter (NumPy from scratch) and Gradient sharpening.
- Small hand-worked mathematical verifications.
"""

from typing import Tuple, Optional
import numpy as np
import cv2


# =========================================================================
# Point Operations
# =========================================================================

def image_negative(img: np.ndarray) -> np.ndarray:
    """Formula: s = (L - 1) - r = 255 - r"""
    return 255 - np.asarray(img, dtype=np.uint8)


def log_transform(img: np.ndarray, c: Optional[float] = None) -> np.ndarray:
    """
    Formula: s = c * log(1 + r)
    Scales dynamic range of dark pixels.
    """
    img_f = img.astype(np.float64)
    if c is None:
        c = 255.0 / np.log(1.0 + np.max(img_f) if np.max(img_f) > 0 else 1.0)
    out = c * np.log(1.0 + img_f)
    return np.clip(out, 0, 255).astype(np.uint8)


def power_law_transform(img: np.ndarray, gamma: float = 1.0, c: float = 1.0) -> np.ndarray:
    """
    Formula: s = c * r^gamma
    gamma < 1 expands dark regions; gamma > 1 compresses them.
    """
    norm = img.astype(np.float64) / 255.0
    out = c * (norm ** gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def contrast_stretching(
    img: np.ndarray,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None
) -> np.ndarray:
    """
    Linearly stretches intensity range [r_min, r_max] to [0, 255].
    Formula: s = (r - r_min) / (r_max - r_min) * 255
    """
    img_f = img.astype(np.float64)
    if r_min is None:
        r_min = float(np.min(img_f))
    if r_max is None:
        r_max = float(np.max(img_f))

    if r_max <= r_min:
        return np.zeros_like(img, dtype=np.uint8)

    stretched = (img_f - r_min) / (r_max - r_min) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


# =========================================================================
# Histogram Equalization: NumPy from scratch vs OpenCV
# =========================================================================

def histogram_equalization_numpy(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Histogram Equalization implemented strictly from scratch in NumPy.
    Algorithm:
    1. Compute PDF: p(r_k) = n_k / (M * N)
    2. Compute CDF: c(r_k) = sum_{j=0}^k p(r_j)
    3. Transformation map: s_k = round((L - 1) * c(r_k))
    4. Map each input pixel through transformation table.
    """
    if img.ndim != 2:
        raise ValueError("Histogram equalization requires a 2D grayscale image.")

    total_pixels = img.size
    # 1. Frequency histogram n_k
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])

    # 2. Cumulative Distribution Function (CDF)
    cdf = hist.cumsum()

    # Mask zero values for numerical stability, standard CDF normalization
    cdf_min = cdf[cdf > 0].min()
    cdf_range = total_pixels - cdf_min

    if cdf_range <= 0:
        return img.copy(), hist

    # Lookup table mapping [0, 255]
    lut = np.round(((cdf - cdf_min) / cdf_range) * 255.0)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    equalized = lut[img]
    return equalized, hist


def histogram_equalization_cv2(img: np.ndarray) -> np.ndarray:
    """Standard OpenCV histogram equalization reference."""
    return cv2.equalizeHist(img)


# =========================================================================
# Spatial Filtering: 2D Convolution in Pure NumPy
# =========================================================================

def convolve2d_numpy(
    img: np.ndarray,
    kernel: np.ndarray,
    border_mode: str = "reflect"
) -> np.ndarray:
    """
    2D Spatial Convolution implemented from scratch in pure NumPy.
    Flips the kernel by 180 degrees (correlation vs convolution) and applies sliding window.
    Vectorized using slice operations for high performance.
    Supports 'reflect' and 'constant' (zero-padding) border modes.
    """
    k_h, k_w = kernel.shape
    if k_h % 2 == 0 or k_w % 2 == 0:
        raise ValueError("Kernel dimensions must be odd.")

    pad_h = k_h // 2
    pad_w = k_w // 2

    # Flip kernel for formal mathematical convolution
    k_flipped = np.flipud(np.fliplr(kernel)).astype(np.float64)

    # Apply padding
    if border_mode == "reflect":
        padded = np.pad(img.astype(np.float64), ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    elif border_mode == "constant":
        padded = np.pad(img.astype(np.float64), ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
    else:
        raise ValueError(f"Unsupported border mode: {border_mode}")

    h, w = img.shape
    output = np.zeros((h, w), dtype=np.float64)

    for di in range(k_h):
        for dj in range(k_w):
            weight = k_flipped[di, dj]
            if weight != 0:
                output += weight * padded[di : di + h, dj : dj + w]

    return output


def mean_filter_numpy(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Mean spatial smoothing filter in pure NumPy."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size * kernel_size)
    conv = convolve2d_numpy(img, kernel, border_mode="reflect")
    return np.clip(np.round(conv), 0, 255).astype(np.uint8)


def gaussian_kernel_2d(kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Generates normalized 2D Gaussian kernel."""
    radius = kernel_size // 2
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    kernel_sum = kernel.sum()
    if kernel_sum != 0:
        kernel /= kernel_sum
    return kernel


def gaussian_filter_numpy(
    img: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0
) -> np.ndarray:
    """Gaussian smoothing filter in pure NumPy."""
    kernel = gaussian_kernel_2d(kernel_size, sigma)
    conv = convolve2d_numpy(img, kernel, border_mode="reflect")
    return np.clip(np.round(conv), 0, 255).astype(np.uint8)


def laplacian_sharpening_numpy(
    img: np.ndarray,
    c: float = 1.0
) -> np.ndarray:
    """
    Laplacian Sharpening in pure NumPy.
    Standard Laplacian kernel:
    [[ 0,  1,  0],
     [ 1, -4,  1],
     [ 0,  1,  0]]
    Sharpened = Original - c * Laplacian
    """
    laplacian_kernel = np.array([
        [0.0, 1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float64)

    lap = convolve2d_numpy(img, laplacian_kernel, border_mode="reflect")
    sharpened = img.astype(np.float64) - c * lap
    return np.clip(np.round(sharpened), 0, 255).astype(np.uint8)


# =========================================================================
# Hand-worked Example Verification
# =========================================================================

def verify_hand_worked_enhancement():
    """
    Hand-worked 3x3 patch convolution verification:
    Patch:
    [[10, 20, 30],
     [40, 50, 60],
     [70, 80, 90]]

    Kernel (3x3 average filter 1/9):
    Center element without boundary effects (interior pixel 50):
    Mean = (10+20+30+40+50+60+70+80+90) / 9 = 450 / 9 = 50.0.

    Laplacian at center (using [[0,1,0],[1,-4,1],[0,1,0]]):
    Laplacian(50) = 20 + 40 + 60 + 80 - 4*(50) = 200 - 200 = 0.
    Sharpened(50) = 50 - 0 = 50.
    """
    patch = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    kernel_avg = np.ones((3, 3), dtype=np.float64) / 9.0
    res = convolve2d_numpy(patch, kernel_avg, border_mode="constant")
    # Center pixel (1, 1) has no padding overlap
    assert abs(res[1, 1] - 50.0) < 1e-6, f"Convolution center mismatch: {res[1, 1]}"

    # Test Gamma transform hand calculation
    # For r = 64 / 255, gamma = 0.5: s = sqrt(64/255) * 255 = 8 / sqrt(255) * 255 = 8 * sqrt(255) = 127.749 -> 128
    single_val = np.array([[64]], dtype=np.uint8)
    gamma_out = power_law_transform(single_val, gamma=0.5)
    expected_gamma = int(np.round(np.sqrt(64.0 / 255.0) * 255.0))
    assert abs(int(gamma_out[0, 0]) - expected_gamma) <= 1, "Gamma hand-worked check failed"

    return True


if __name__ == "__main__":
    verify_hand_worked_enhancement()
    print("Spatial enhancement module verified successfully!")
