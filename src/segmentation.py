"""
Image Segmentation and Edge Detection Module.

Implements:
- Gradient Edge Detectors: Roberts, Prewitt, Sobel Magnitude (NumPy from scratch vs cv2.Sobel).
- Second-Derivative: Laplacian and Laplacian of Gaussian (LoG).
- Thresholding:
  - Global Fixed Threshold (127)
  - Otsu Thresholding written in pure NumPy from scratch, verified vs cv2.threshold.
  - Adaptive Thresholding (Mean and Gaussian).
- Line and Edge Linking via Hough Transform.
- Hand-worked mathematical example verification.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
from src.enhancement import convolve2d_numpy, gaussian_kernel_2d


# =========================================================================
# Gradient Operators: Pure NumPy Implementations
# =========================================================================

def sobel_magnitude_numpy(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sobel gradient magnitude and directional components computed in pure NumPy.
    Sobel kernels:
    Gx:
    [[-1,  0,  1],
     [-2,  0,  2],
     [-1,  0,  1]]
    Gy:
    [[-1, -2, -1],
     [ 0,  0,  0],
     [ 1,  2,  1]]
    Magnitude: G = sqrt(Gx^2 + Gy^2)
    Uses reflection padding (BORDER_REFLECT_101) matching OpenCV.
    """
    kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float64)
    ky = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float64)

    pad = 1
    padded = np.pad(img.astype(np.float64), pad, mode="reflect")
    h, w = img.shape
    gx = np.zeros((h, w), dtype=np.float64)
    gy = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            region = padded[i : i + 3, j : j + 3]
            gx[i, j] = np.sum(region * kx)
            gy[i, j] = np.sum(region * ky)

    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag_uint8 = np.clip(np.round(mag), 0, 255).astype(np.uint8)

    return mag_uint8, gx, gy


def sobel_magnitude_cv2(img: np.ndarray) -> np.ndarray:
    """OpenCV Sobel reference for gradient magnitude."""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.clip(np.round(mag), 0, 255).astype(np.uint8)


def roberts_cross_numpy(img: np.ndarray) -> np.ndarray:
    """Roberts Cross diagonal gradient operator."""
    kx = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    ky = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    # Simple convolution via pad
    padded = np.pad(img.astype(np.float64), ((0, 1), (0, 1)), mode="reflect")
    gx = padded[:-1, :-1] - padded[1:, 1:]
    gy = padded[:-1, 1:] - padded[1:, :-1]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.clip(np.round(mag), 0, 255).astype(np.uint8)


def prewitt_gradient_numpy(img: np.ndarray) -> np.ndarray:
    """Prewitt gradient operator."""
    kx = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], dtype=np.float64)
    ky = np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    gx = convolve2d_numpy(img, kx, border_mode="reflect")
    gy = convolve2d_numpy(img, ky, border_mode="reflect")
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.clip(np.round(mag), 0, 255).astype(np.uint8)


def laplacian_of_gaussian_numpy(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Laplacian of Gaussian (LoG) operator."""
    # First Gaussian smoothing
    kernel = gaussian_kernel_2d(ksize, sigma)
    smoothed = convolve2d_numpy(img, kernel, border_mode="reflect")
    # Then standard Laplacian
    lap_kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    log_out = convolve2d_numpy(smoothed, lap_kernel, border_mode="reflect")
    return log_out


# =========================================================================
# Thresholding: Pure NumPy Otsu from scratch vs OpenCV
# =========================================================================

def global_threshold(img: np.ndarray, thresh: int = 127, invert: bool = False) -> np.ndarray:
    """
    Fixed global thresholding (Trivial Baseline).
    """
    if invert:
        binary = (img <= thresh).astype(np.uint8) * 255
    else:
        binary = (img >= thresh).astype(np.uint8) * 255
    return binary


def otsu_threshold_numpy(img: np.ndarray, invert: bool = False) -> Tuple[int, np.ndarray]:
    """
    Otsu's Thresholding algorithm written strictly in pure NumPy from scratch.
    Maximizes the between-class variance:
        sigma_B^2(t) = omega_0(t) * omega_1(t) * [mu_0(t) - mu_1(t)]^2
    Where:
        omega_0(t) = sum_{i=0}^t p_i
        omega_1(t) = 1 - omega_0(t)
        mu_0(t) = sum_{i=0}^t i * p_i / omega_0(t)
        mu_1(t) = sum_{i=t+1}^{L-1} i * p_i / omega_1(t)
    """
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    total_pixels = img.size
    p = hist.astype(np.float64) / total_pixels

    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_total = mu[-1]

    best_thresh = 0
    max_variance = -1.0

    # Test all possible threshold values 0 <= t < 255
    for t in range(256):
        w0 = omega[t]
        w1 = 1.0 - w0

        if w0 == 0 or w1 == 0:
            continue

        m0 = mu[t] / w0
        m1 = (mu_total - mu[t]) / w1

        # Between-class variance
        sigma_b_squared = w0 * w1 * ((m0 - m1) ** 2)

        if sigma_b_squared > max_variance:
            max_variance = sigma_b_squared
            best_thresh = t

    if invert:
        binary = (img <= best_thresh).astype(np.uint8) * 255
    else:
        binary = (img > best_thresh).astype(np.uint8) * 255

    return best_thresh, binary


def otsu_threshold_cv2(img: np.ndarray, invert: bool = False) -> Tuple[float, np.ndarray]:
    """OpenCV Otsu reference."""
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    val, binary = cv2.threshold(img, 0, 255, flag + cv2.THRESH_OTSU)
    return val, binary


def gaussian_blur_separable_numpy(img: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    """
    Separable 2D Gaussian smoothing in pure NumPy.
    Replaces O(K^2) 2D convolution with two O(K) 1D passes (horizontal then vertical).
    """
    radius = ksize // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    g1d = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    g1d /= np.sum(g1d)

    pad = radius
    padded = np.pad(img.astype(np.float64), ((pad, pad), (pad, pad)), mode="reflect")
    h, w = img.shape
    temp = np.zeros((h + 2 * pad, w), dtype=np.float64)
    for dj in range(ksize):
        temp += g1d[dj] * padded[:, dj : dj + w]

    out = np.zeros((h, w), dtype=np.float64)
    for di in range(ksize):
        out += g1d[di] * temp[di : di + h, :]

    return out


def adaptive_threshold_gaussian_numpy(
    img: np.ndarray,
    block_size: int = 15,
    c: float = 5.0,
    invert: bool = True
) -> np.ndarray:
    """
    Adaptive Gaussian Thresholding implemented in pure NumPy.
    Calculates weighted local Gaussian mean T(x,y) = local_mean - c.
    Robust against non-uniform illumination and shading on concrete surfaces.
    Uses fast separable 1D Gaussian filtering.
    """
    if block_size % 2 == 0:
        raise ValueError("block_size must be odd.")

    local_mean = gaussian_blur_separable_numpy(img, block_size, sigma=block_size / 6.0)
    threshold_map = local_mean - c

    if invert:
        # Dark cracks become 255
        binary = (img.astype(np.float64) <= threshold_map).astype(np.uint8) * 255
    else:
        binary = (img.astype(np.float64) >= threshold_map).astype(np.uint8) * 255

    return binary


def adaptive_threshold_cv2(
    img: np.ndarray,
    block_size: int = 15,
    c: float = 5.0,
    invert: bool = True
) -> np.ndarray:
    """OpenCV adaptive threshold reference."""
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, block_size, c
    )


# =========================================================================
# Hough Transform for Linear Cracks
# =========================================================================

def detect_hough_lines(
    edge_map: np.ndarray,
    threshold: int = 50,
    min_line_length: int = 30,
    max_line_gap: int = 10
) -> np.ndarray:
    """
    Extracts line segments representing major continuous cracks using Hough transform.
    """
    lines = cv2.HoughLinesP(
        edge_map,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    line_mask = np.zeros_like(edge_map, dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
    return line_mask


# =========================================================================
# Hand-worked Example Verification
# =========================================================================

def verify_hand_worked_segmentation():
    """
    Hand-worked verification for Otsu thresholding:
    Construct an image with two distinct pixel clusters:
    50 pixels of intensity 50, and 50 pixels of intensity 200.
    Hist: 50 at 50, 50 at 200.
    Total = 100.
    Between-class variance is maximized when threshold t splits the two peaks,
    specifically any t in [50, 199], with midpoint around (50 + 200) / 2 = 125.
    """
    img = np.array([50] * 50 + [200] * 50, dtype=np.uint8).reshape(10, 10)
    thresh_np, _ = otsu_threshold_numpy(img)
    thresh_cv, _ = otsu_threshold_cv2(img)

    assert thresh_np == int(thresh_cv), f"Otsu threshold mismatch: {thresh_np} vs {thresh_cv}"
    assert 50 <= thresh_np < 200, f"Otsu threshold {thresh_np} outside expected range [50, 199]"

    # Hand-worked Sobel gradient on horizontal edge:
    # Top row: 10, Middle row: 50, Bottom row: 90
    test_patch = np.array([
        [10, 10, 10],
        [50, 50, 50],
        [90, 90, 90]
    ], dtype=np.uint8)
    mag, gx, gy = sobel_magnitude_numpy(test_patch)
    # Gx should be 0 (no horizontal variation)
    assert abs(gx[1, 1]) < 1e-6, f"Expected Gx=0, got {gx[1, 1]}"
    # Gy: [[-1,-2,-1],[0,0,0],[1,2,1]] * patch
    # Gy = 1*(90) + 2*(90) + 1*(90) - [1*(10) + 2*(10) + 1*(10)] = 360 - 40 = 320
    assert abs(gy[1, 1] - 320.0) < 1e-6, f"Expected Gy=320, got {gy[1, 1]}"

    return True


if __name__ == "__main__":
    verify_hand_worked_segmentation()
    print("Segmentation module verified successfully!")
