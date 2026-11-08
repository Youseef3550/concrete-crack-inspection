"""
2D Frequency-Domain Image Processing Module.

Implements:
- 2D Discrete Fourier Transform (DFT), Centered Log-Magnitude Spectrum, and Inverse DFT.
- Ideal Low-Pass and High-Pass Filters.
- Butterworth Low-Pass and High-Pass Filters (NumPy from scratch).
- Gaussian Low-Pass and High-Pass Filters (NumPy from scratch).
- Ringing artifact and cutoff frequency (D0) analysis.
- Hand-worked mathematical verification of frequency filter transfer functions.
"""

from typing import Tuple
import numpy as np


def compute_dft_spectrum(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 2D DFT and returns:
    1. Shifted complex frequency domain representation F(u,v).
    2. Centered log-magnitude spectrum for visualization: S(u,v) = log(1 + |F(u,v)|).
    """
    f = np.fft.fft2(img.astype(np.float64))
    f_shift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(1.0 + np.abs(f_shift))
    # Normalize spectrum to [0, 255] for display
    norm_spectrum = ((magnitude_spectrum - magnitude_spectrum.min()) /
                     (magnitude_spectrum.max() - magnitude_spectrum.min() + 1e-8) * 255.0)
    return f_shift, norm_spectrum.astype(np.uint8)


def inverse_dft(f_shift: np.ndarray) -> np.ndarray:
    """
    Computes inverse DFT from shifted frequency representation.
    Returns real spatial domain image clipped to [0, 255].
    """
    f_ishift = np.fft.ifftshift(f_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_real = np.real(img_back)
    return np.clip(np.round(img_real), 0, 255).astype(np.uint8)


def distance_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """
    Computes Euclidean distance matrix D(u, v) from center of frequency plane.
    Formula: D(u,v) = sqrt((u - M//2)^2 + (v - N//2)^2)
    """
    M, N = shape
    u = np.arange(M) - (M // 2)
    v = np.arange(N) - (N // 2)
    U, V = np.meshgrid(u, v, indexing="ij")
    return np.sqrt(U ** 2 + V ** 2)


def ideal_filter_numpy(
    shape: Tuple[int, int],
    d0: float,
    filter_type: str = "low"
) -> np.ndarray:
    """
    Ideal Low-Pass / High-Pass filter transfer function H(u,v).
    H(u,v) = 1 if D(u,v) <= D0 else 0 (for low-pass).
    Known for pronounced ringing (Gibbs phenomenon) in spatial domain due to sinc profile.
    """
    D = distance_matrix(shape)
    if filter_type == "low":
        H = (D <= d0).astype(np.float64)
    elif filter_type == "high":
        H = (D > d0).astype(np.float64)
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")
    return H


def butterworth_filter_numpy(
    shape: Tuple[int, int],
    d0: float,
    order: int = 2,
    filter_type: str = "low"
) -> np.ndarray:
    """
    Butterworth Low-Pass / High-Pass filter transfer function in pure NumPy.
    Low-pass formula:
        H(u,v) = 1 / (1 + [D(u,v) / D0]^(2n))
    High-pass formula:
        H(u,v) = 1 / (1 + [D0 / D(u,v)]^(2n))
    Smooth transition without sharp cutoff eliminates visible ringing artifacts at order n=2.
    """
    D = distance_matrix(shape)
    # Avoid division by zero at DC component center
    D_safe = np.where(D == 0, 1e-10, D)

    if filter_type == "low":
        H = 1.0 / (1.0 + (D / float(d0)) ** (2 * order))
    elif filter_type == "high":
        H = 1.0 / (1.0 + (float(d0) / D_safe) ** (2 * order))
        H[D == 0] = 0.0
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")
    return H


def gaussian_frequency_filter_numpy(
    shape: Tuple[int, int],
    d0: float,
    filter_type: str = "low"
) -> np.ndarray:
    """
    Gaussian Low-Pass / High-Pass filter transfer function in pure NumPy.
    Low-pass formula:
        H(u,v) = exp(-D^2(u,v) / (2 * D0^2))
    High-pass formula:
        H(u,v) = 1 - exp(-D^2(u,v) / (2 * D0^2))
    Completely zero ringing artifacts because Fourier transform of Gaussian is also Gaussian.
    """
    D = distance_matrix(shape)
    H_low = np.exp(-(D ** 2) / (2.0 * (d0 ** 2)))
    if filter_type == "low":
        return H_low
    elif filter_type == "high":
        return 1.0 - H_low
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")


def apply_frequency_filter(img: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies frequency domain filter transfer function H to input spatial image.
    G(u,v) = H(u,v) * F(u,v)
    Returns:
    1. Filtered spatial image.
    2. Filtered log-magnitude spectrum.
    """
    f_shift, _ = compute_dft_spectrum(img)
    g_shift = f_shift * H
    filtered_img = inverse_dft(g_shift)

    # Compute magnitude spectrum of filtered signal
    mag = np.log(1.0 + np.abs(g_shift))
    norm_spec = ((mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255.0).astype(np.uint8)

    return filtered_img, norm_spec


# =========================================================================
# Hand-worked Mathematical Verification
# =========================================================================

def verify_hand_worked_frequency():
    """
    Hand-worked verification on small 5x5 grid:
    Center is at (2, 2).
    For D0 = 2.0, order n = 2:
    - At center (2, 2): D = 0 -> H_low = 1 / (1 + 0) = 1.0.
    - At distance D = 2.0 (e.g. point (2, 4) where |4 - 2| = 2):
      D / D0 = 1.0.
      H_low = 1 / (1 + 1^(2*2)) = 1 / (1 + 1) = 0.500 (cutoff definition!).
    - Gaussian filter at D = D0 = 2.0:
      H_gauss = exp(-2^2 / (2 * 2^2)) = exp(-0.5) = 0.60653.
    """
    shape = (5, 5)
    d0 = 2.0
    bw_low = butterworth_filter_numpy(shape, d0=d0, order=2, filter_type="low")
    assert abs(bw_low[2, 2] - 1.0) < 1e-6, f"Butterworth DC mismatch: {bw_low[2, 2]}"
    assert abs(bw_low[2, 4] - 0.5) < 1e-6, f"Butterworth cutoff point mismatch: {bw_low[2, 4]}"

    gauss_low = gaussian_frequency_filter_numpy(shape, d0=d0, filter_type="low")
    assert abs(gauss_low[2, 2] - 1.0) < 1e-6, f"Gaussian DC mismatch: {gauss_low[2, 2]}"
    expected_gauss_at_d0 = np.exp(-0.5)
    assert abs(gauss_low[2, 4] - expected_gauss_at_d0) < 1e-4, f"Gaussian cutoff mismatch: {gauss_low[2, 4]}"

    return True


if __name__ == "__main__":
    verify_hand_worked_frequency()
    print("Frequency-domain module verified successfully!")
