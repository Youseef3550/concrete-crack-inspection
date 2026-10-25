"""
Data Loading, Splitting, and Controlled Degradation Module.

Provides:
- Consistent image and ground truth loading (BGR to RGB/grayscale, float/uint8, resizing)
- Development / Test split (70% dev, 30% held-out test)
- Controlled noise degradation (Gaussian, Salt-and-Pepper, Speckle, Poisson) with fixed random seeds.
"""

import os
import glob
from typing import Tuple, List, Dict, Optional
import numpy as np
import cv2


def load_image(
    image_path: str,
    as_gray: bool = True,
    target_size: Optional[Tuple[int, int]] = None,
    as_float: bool = False
) -> np.ndarray:
    """
    Consistent loading function for the whole project.
    Always converts BGR to RGB or grayscale and standardizes data type.
    
    Parameters:
        image_path: Path to the image file.
        as_gray: If True, returns single-channel 2D array. If False, returns 3-channel RGB array.
        target_size: Optional (width, height) to resize.
        as_float: If True, returns float32 in [0.0, 1.0]. If False, returns uint8 in [0, 255].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to decode image from {image_path}")

    if as_gray:
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if target_size is not None:
        # target_size is (width, height) for cv2.resize
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    if as_float:
        img = img.astype(np.float32) / 255.0
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def load_mask(
    mask_path: str,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Loads binary ground truth mask, ensuring uint8 array with values in {0, 255}.
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found at {mask_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to decode mask from {mask_path}")

    if target_size is not None:
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    # Standardize to 0 and 255
    binary_mask = (mask > 127).astype(np.uint8) * 255
    return binary_mask


def load_dataset(
    images_dir: str,
    masks_dir: str,
    as_gray: bool = True,
    target_size: Optional[Tuple[int, int]] = None
) -> List[Dict]:
    """
    Loads paired images and ground truth masks from directory.
    Returns list of dicts: [{'name': ..., 'image': ..., 'mask': ...}, ...]
    """
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    dataset = []

    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # Match mask with png or jpg
        mask_candidates = [
            os.path.join(masks_dir, f"{base_name}.png"),
            os.path.join(masks_dir, f"{base_name}.jpg"),
        ]
        mask_path = next((p for p in mask_candidates if os.path.exists(p)), None)

        if mask_path is None:
            continue

        img = load_image(img_path, as_gray=as_gray, target_size=target_size)
        mask = load_mask(mask_path, target_size=target_size)

        dataset.append({
            "name": base_name,
            "image_path": img_path,
            "mask_path": mask_path,
            "image": img,
            "mask": mask
        })

    return dataset


def train_test_split(
    dataset: List[Dict],
    split_ratio: float = 0.7,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Splits dataset into development set (tuning) and held-out test set (used once at the end).
    """
    rng = np.random.RandomState(random_seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    split_idx = int(len(dataset) * split_ratio)
    dev_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    dev_set = [dataset[i] for i in dev_indices]
    test_set = [dataset[i] for i in test_indices]

    return dev_set, test_set


# =========================================================================
# Controlled Degradation Models (Section 3.2 requirement)
# =========================================================================

def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    sigma: float = 25.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Adds additive Gaussian noise to image.
    Formula: I_noisy = I + N(mean, sigma^2)
    """
    rng = np.random.RandomState(seed)
    noise = rng.normal(mean, sigma, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(
    image: np.ndarray,
    salt_prob: float = 0.02,
    pepper_prob: float = 0.02,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Adds impulse (Salt-and-Pepper) noise to image.
    Salt: 255 (white pixels) with probability salt_prob.
    Pepper: 0 (black pixels) with probability pepper_prob.
    """
    rng = np.random.RandomState(seed)
    noisy = image.copy()
    random_matrix = rng.rand(*image.shape[:2])

    noisy[random_matrix < pepper_prob] = 0
    noisy[(random_matrix >= pepper_prob) & (random_matrix < (pepper_prob + salt_prob))] = 255
    return noisy


def add_speckle_noise(
    image: np.ndarray,
    variance: float = 0.05,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Adds multiplicative speckle noise to image.
    Formula: I_noisy = I + I * N(0, variance)
    """
    rng = np.random.RandomState(seed)
    sigma = np.sqrt(variance)
    noise = rng.normal(0, sigma, image.shape)
    noisy = image.astype(np.float64) + image.astype(np.float64) * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_poisson_noise(
    image: np.ndarray,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Adds signal-dependent Poisson (shot) noise to image.
    """
    rng = np.random.RandomState(seed)
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals)) if vals > 0 else 256
    scaled = image.astype(np.float64) / vals
    noisy = rng.poisson(scaled * vals)
    return np.clip(noisy, 0, 255).astype(np.uint8)
