"""
Dataset Downloader and Preparation Script for Concrete Crack Inspection.

Downloads the CrackForest Benchmark Dataset (118 high-resolution pavement/concrete
images with pixel-level ground truth masks) and converts ground truth .mat files
to clean binary PNG masks. Includes an offline benchmark generator fallback.
"""

import os
import sys
import glob
import concurrent.futures
import urllib.request
import numpy as np
import cv2
import scipy.io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
IMAGES_DIR = os.path.join(RAW_DIR, "images")
MASKS_DIR = os.path.join(RAW_DIR, "masks")

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/cuilimeng/CrackForest-dataset/master"

def ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)

def download_file(url: str, dest_path: str, timeout: int = 15) -> bool:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response, open(dest_path, "wb") as out_file:
            out_file.write(response.read())
        return True
    except Exception as e:
        return False

def convert_mat_to_png(mat_path: str, png_path: str) -> bool:
    """Converts CrackForest .mat groundTruth struct to 8-bit binary PNG mask (0=bg, 255=crack)."""
    try:
        data = scipy.io.loadmat(mat_path)
        gt = data["groundTruth"][0, 0]
        seg = gt["Segmentation"]
        # In CrackForest dataset, class 1 is background, class 2 is crack
        binary_mask = (seg == 2).astype(np.uint8) * 255
        cv2.imwrite(png_path, binary_mask)
        return True
    except Exception as e:
        print(f"Error converting {mat_path}: {e}")
        return False

def generate_fallback_dataset(num_samples: int = 118):
    """
    Generates synthetic concrete surface benchmark images with hairline cracks
    and ground truth masks if remote repository is unreachable.
    """
    print(f"Generating {num_samples} benchmark concrete images with realistic noise and cracks...")
    np.random.seed(42)
    height, width = 320, 480

    for i in range(1, num_samples + 1):
        filename = f"{i:03d}.jpg"
        img_path = os.path.join(IMAGES_DIR, filename)
        mask_path = os.path.join(MASKS_DIR, f"{i:03d}.png")

        if os.path.exists(img_path) and os.path.exists(mask_path):
            continue

        # 1. Base concrete surface with uneven lighting gradient
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xv, yv = np.meshgrid(x, y)
        illumination = 140 + 30 * np.sin(xv * np.pi) * np.cos(yv * np.pi)
        
        # 2. Surface roughness and granular aggregate texture
        roughness = np.random.normal(0, 15, (height, width))
        base_concrete = np.clip(illumination + roughness, 40, 240).astype(np.uint8)

        # 3. Add random concrete pores (small dark specks)
        num_pores = np.random.randint(150, 400)
        for _ in range(num_pores):
            px = np.random.randint(0, width)
            py = np.random.randint(0, height)
            radius = np.random.randint(1, 4)
            cv2.circle(base_concrete, (px, py), radius, int(np.random.randint(20, 70)), -1)

        # 4. Synthesize crack path (random walk)
        mask = np.zeros((height, width), dtype=np.uint8)
        num_cracks = np.random.randint(1, 3)
        for _ in range(num_cracks):
            curr_x = np.random.randint(width // 4, 3 * width // 4)
            curr_y = np.random.randint(10, 50)
            crack_length = np.random.randint(200, 350)
            thickness = np.random.randint(1, 3)

            for step in range(crack_length):
                next_x = curr_x + int(np.random.choice([-2, -1, 0, 1, 2], p=[0.1, 0.25, 0.3, 0.25, 0.1]))
                next_y = curr_y + int(np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2]))
                next_x = max(0, min(width - 1, next_x))
                next_y = max(0, min(height - 1, next_y))

                cv2.line(mask, (curr_x, curr_y), (next_x, next_y), 255, thickness)
                curr_x, curr_y = next_x, next_y
                if curr_y >= height - 10:
                    break

        # Blend crack onto concrete: crack pixels are dark with noise
        crack_indices = mask > 0
        crack_noise = np.random.normal(35, 10, crack_indices.sum()).astype(np.int32)
        crack_noise = np.clip(crack_noise, 10, 80).astype(np.uint8)
        base_concrete[crack_indices] = crack_noise

        # Convert to 3-channel RGB
        rgb_img = cv2.cvtColor(base_concrete, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(img_path, rgb_img)
        cv2.imwrite(mask_path, mask)

    print(f"Successfully generated {num_samples} images and masks.")

def download_dataset():
    ensure_dirs()
    print("Checking dataset status...")
    existing_images = len(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    existing_masks = len(glob.glob(os.path.join(MASKS_DIR, "*.png")))

    if existing_images >= 118 and existing_masks >= 118:
        print(f"Dataset already downloaded and prepared: {existing_images} images, {existing_masks} masks.")
        return

    print("Attempting to download CrackForest Dataset from GitHub repository...")
    num_samples = 118
    download_tasks = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for i in range(1, num_samples + 1):
            img_name = f"{i:03d}.jpg"
            mat_name = f"{i:03d}.mat"
            img_url = f"{GITHUB_RAW_BASE}/image/{img_name}"
            mat_url = f"{GITHUB_RAW_BASE}/groundTruth/{mat_name}"
            
            img_path = os.path.join(IMAGES_DIR, img_name)
            mat_path = os.path.join(RAW_DIR, mat_name)
            
            download_tasks.append((executor.submit(download_file, img_url, img_path), img_name, mat_url, mat_path, i))

        downloaded_count = 0
        for task, img_name, mat_url, mat_path, idx in download_tasks:
            success = task.result()
            if success:
                mat_success = download_file(mat_url, mat_path)
                if mat_success:
                    png_path = os.path.join(MASKS_DIR, f"{idx:03d}.png")
                    if convert_mat_to_png(mat_path, png_path):
                        downloaded_count += 1
                        try:
                            os.remove(mat_path)
                        except OSError:
                            pass

    print(f"Downloaded and converted {downloaded_count}/{num_samples} real images & masks.")
    if downloaded_count < 50:
        print("Network download incomplete or restricted. Activating benchmark generator fallback...")
        generate_fallback_dataset(num_samples)
    else:
        print(f"Dataset ready: {len(glob.glob(os.path.join(IMAGES_DIR, '*.jpg')))} images verified.")

if __name__ == "__main__":
    download_dataset()
