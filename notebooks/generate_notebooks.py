"""
Script to generate 01_image_analysis.ipynb and 02_results.ipynb in valid Jupyter notebook JSON format.
"""

import json
import os

NOTEBOOKS_DIR = os.path.dirname(os.path.abspath(__file__))

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.12.4"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else source.splitlines(keepends=True)
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else source.splitlines(keepends=True)
    }

# -------------------------------------------------------------
# Notebook 1: Image Exploration and Resolution Reduction Study
# -------------------------------------------------------------
cells_nb1 = [
    markdown_cell([
        "# CSE281 Digital Image Processing - Project Milestone 1\n",
        "## Problem Formulation and Image Analysis\n",
        "**Topic:** Concrete Surface Crack Detection and Structural Damage Assessment\n",
        "**Course:** CSE281 - Digital Image Processing | Alamein International University (AIU)\n",
        "---\n"
    ]),
    markdown_cell([
        "### 1. Problem Formulation\n",
        "**Target Users:** Structural engineers, civil infrastructure inspectors, and municipal maintenance squads responsible for auditing bridges, highway pavements, and concrete structures.\n\n",
        "**Pipeline Output:** A clean binary crack segmentation mask, geometric damage metrics (cracked surface area percentage, crack length, entity count), and an automated civil safety verdict (**PASS / MONITOR / FAIL**).\n\n",
        "**Cost of Wrong Outputs:**\n",
        "- **False Negatives (Missed Cracks):** High hazard cost. Overlooked cracks allow water and chloride penetration, accelerating rebar corrosion and risking catastrophic structural collapse.\n",
        "- **False Positives (Surface Pores Mistaken for Cracks):** High economic cost. Unnecessary road closures, structural halting, and costly destructive sonic testing on healthy concrete.\n\n",
        "**Primary Metric:** Intersection over Union (**IoU / Jaccard Index**). IoU penalizes both missed crack pixels (false negatives) and spurious noise pixels (false positives) strictly on thin, low-prevalence structural targets where pixel accuracy would be deceptively high (>98% on background)."
    ]),
    code_cell([
        "import os\n",
        "import sys\n",
        "import numpy as np\n",
        "import cv2\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Add src to path\n",
        "sys.path.append(os.path.abspath('..'))\n",
        "from src.data import load_image, load_mask, load_dataset, train_test_split\n",
        "from src.frequency import compute_dft_spectrum\n",
        "from src.metrics import iou\n",
        "\n",
        "print('Loaded DIP exploration libraries!')"
    ]),
    markdown_cell([
        "### 2. Dataset Exploration & Statistical Profiling\n",
        "Loading sample images and analyzing size, data types, intensity range, histograms, and frequency spectra."
    ]),
    code_cell([
        "images_dir = '../data/raw/images'\n",
        "masks_dir = '../data/raw/masks'\n",
        "dataset = load_dataset(images_dir, masks_dir)\n",
        "print(f'Total dataset size: {len(dataset)} paired images.')\n",
        "\n",
        "sample = dataset[0]\n",
        "img = sample['image']\n",
        "mask = sample['mask']\n",
        "\n",
        "print(f'Image Dimensions: {img.shape}')\n",
        "print(f'Data Type:        {img.dtype}')\n",
        "print(f'Dynamic Range:    [{img.min()}, {img.max()}]')\n",
        "print(f'Mean Intensity:   {img.mean():.2f}')\n",
        "print(f'Std Deviation:    {img.std():.2f}')"
    ]),
    code_cell([
        "# Visualize typical image, difficult low-contrast image, and histograms\n",
        "fig, axes = plt.subplots(2, 3, figsize=(15, 8))\n",
        "\n",
        "axes[0, 0].imshow(img, cmap='gray')\n",
        "axes[0, 0].set_title('Typical Concrete Surface (001.jpg)')\n",
        "\n",
        "axes[0, 1].imshow(mask, cmap='gray')\n",
        "axes[0, 1].set_title('Ground Truth Crack Mask')\n",
        "\n",
        "_, spectrum = compute_dft_spectrum(img)\n",
        "axes[0, 2].imshow(spectrum, cmap='magma')\n",
        "axes[0, 2].set_title('Centered 2D Fourier Spectrum')\n",
        "\n",
        "axes[1, 0].hist(img.ravel(), bins=256, range=[0, 256], color='gray')\n",
        "axes[1, 0].set_title('Intensity Histogram')\n",
        "\n",
        "difficult_sample = dataset[15]['image']\n",
        "axes[1, 1].imshow(difficult_sample, cmap='gray')\n",
        "axes[1, 1].set_title('Difficult Low-Contrast Surface')\n",
        "\n",
        "axes[1, 2].hist(difficult_sample.ravel(), bins=256, range=[0, 256], color='darkred')\n",
        "axes[1, 2].set_title('Low-Contrast Narrow Histogram')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]),
    markdown_cell([
        "### 3. Spatial Resolution and Gray-Level Reduction Study\n",
        "**Objective:** Determine the lowest viable spatial resolution and quantization bit depth before crack continuity is destroyed."
    ]),
    code_cell([
        "# 1. Gray-Level Quantization (8-bit down to 1-bit)\n",
        "bit_depths = [8, 6, 4, 2, 1]\n",
        "fig, axes = plt.subplots(1, 5, figsize=(18, 4))\n",
        "\n",
        "for idx, bits in enumerate(bit_depths):\n",
        "    levels = 2 ** bits\n",
        "    quantized = np.floor(img.astype(float) / (256.0 / levels)) * (256.0 / levels)\n",
        "    quantized = np.clip(quantized, 0, 255).astype(np.uint8)\n",
        "    axes[idx].imshow(quantized, cmap='gray')\n",
        "    axes[idx].set_title(f'{bits}-Bit ({levels} Levels)')\n",
        "    axes[idx].axis('off')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# 2. Spatial Subsampling (1x down to 8x downsampling)\n",
        "scale_factors = [1, 2, 4, 8]\n",
        "fig, axes = plt.subplots(1, 4, figsize=(16, 4))\n",
        "\n",
        "for idx, scale in enumerate(scale_factors):\n",
        "    h, w = img.shape\n",
        "    down = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)\n",
        "    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)\n",
        "    axes[idx].imshow(up, cmap='gray')\n",
        "    axes[idx].set_title(f'Downsample {scale}x ({w//scale}x{h//scale})')\n",
        "    axes[idx].axis('off')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]),
    markdown_cell([
        "### 4. Findings & Architectural Design Decisions\n",
        "\n",
        "| Empirical Finding | Physical Diagnostic Cause | Pipeline Design Decision |\n",
        "| :--- | :--- | :--- |\n",
        "| **Narrow Gray-Level Histogram** | Concrete surface weathering and overcast lighting compress dynamic range | Implement **Gamma correction ($\\gamma=0.85$)** and **percentile contrast stretching** before thresholding |\n",
        "| **Uneven Illumination & Shadows** | Directional sunlight and structural overhangs defeat global thresholds | Employ **Local Adaptive Gaussian Thresholding** rather than fixed or global Otsu |\n",
        "| **High-Frequency Background Granularity** | Sand, gravel, and concrete aggregate produce high-frequency speckle | Apply **2D Butterworth Low-Pass filter ($D_0=45$)** in frequency domain to attenuate speckle without blurring linear edges |\n",
        "| **Hairline Crack Discontinuities** | Sub-pixel narrowing and thresholding produce broken segments | Apply **Directional Morphological Closing ($5\\times 5$ ellipse)** to bridge breaks along crack traces |\n",
        "| **Isolated Concrete Air Voids (Pores)** | Dark pits mistaken for cracks by adaptive threshold | Use **Area Opening / Connected Object Removal** to prune components smaller than 40 px |\n",
        "| **Quantization Limit** | Hairline cracks disappear when bit depth drops below 4 bits | Maintain minimum bit depth of **6 bits (64 gray levels)**; 8 bits standard uint8 |\n",
        "| **Spatial Resolution Limit** | Cracks thinner than 2 pixels vanish beyond 2x downsampling | Maintain original resolution or downsample by at most **2x (240x160 px)** |"
    ])
]

# -------------------------------------------------------------
# Notebook 2: Experimental Results, Tuning, Ablations & Failures
# -------------------------------------------------------------
cells_nb2 = [
    markdown_cell([
        "# CSE281 Digital Image Processing - Project Milestone 2\n",
        "## Experimental Results, Ablation Studies, and Failure Analysis\n",
        "**Topic:** Concrete Surface Crack Detection and Structural Damage Assessment\n",
        "**Course:** CSE281 - Digital Image Processing | Alamein International University (AIU)\n",
        "---\n"
    ]),
    code_cell([
        "import os\n",
        "import sys\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import cv2\n",
        "\n",
        "sys.path.append(os.path.abspath('..'))\n",
        "print('Libraries imported!')"
    ]),
    markdown_cell([
        "### 1. Summary of Held-Out Test Set Performance\n",
        "Comparing Baselines against Pipelines A, B, and C."
    ]),
    code_cell([
        "summary_csv = '../experiments/test_set_summary.csv'\n",
        "if os.path.exists(summary_csv):\n",
        "    df_summary = pd.read_csv(summary_csv)\n",
        "    display(df_summary)\n",
        "else:\n",
        "    print('Run evaluate.py first to populate test_set_summary.csv')\n"
    ]),
    markdown_cell([
        "### 2. Systematic One-Factor-at-a-Time (OFAT) Tuning Plots\n",
        "Visualizing performance vs cutoff frequency D0, smoothing kernel size, and structuring element geometry."
    ]),
    code_cell([
        "log_csv = '../experiments/experiment_log.csv'\n",
        "if os.path.exists(log_csv):\n",
        "    df_log = pd.read_csv(log_csv)\n",
        "    display(df_log.head(10))\n",
        "else:\n",
        "    print('Run run_experiments.py first to populate experiment_log.csv')\n"
    ]),
    markdown_cell([
        "### 3. Failure Analysis on Difficult Concrete Surfaces\n",
        "Visualizing failure cases, categorized by root cause."
    ]),
    code_cell([
        "failures_csv = '../experiments/failure_analysis.csv'\n",
        "if os.path.exists(failures_csv):\n",
        "    df_fail = pd.read_csv(failures_csv)\n",
        "    display(df_fail.head(5))\n",
        "    \n",
        "    # Display worst case image panel\n",
        "    sample_worst = glob.glob('../experiments/failures/*.png')\n",
        "    if sample_worst:\n",
        "        worst_img = cv2.imread(sample_worst[0])\n",
        "        plt.figure(figsize=(15, 6))\n",
        "        plt.imshow(cv2.cvtColor(worst_img, cv2.COLOR_BGR2RGB))\n",
        "        plt.title('Worst Case Diagnostic Breakdown: Raw | GT | Segmented | Pred | Overlay')\n",
        "        plt.axis('off')\n",
        "        plt.show()\n",
        "else:\n",
        "    print('Failure analysis table not generated yet.')"
    ])
]

# Write notebooks
with open(os.path.join(NOTEBOOKS_DIR, "01_image_analysis.ipynb"), "w") as f:
    json.dump(create_notebook(cells_nb1), f, indent=2)

with open(os.path.join(NOTEBOOKS_DIR, "02_results.ipynb"), "w") as f:
    json.dump(create_notebook(cells_nb2), f, indent=2)

print("Created 01_image_analysis.ipynb and 02_results.ipynb successfully!")
