"""
Production Entrypoint for Concrete Crack Inspection Pipeline.

Applies Pipeline C to a single input image and outputs:
- Stage-by-stage timings
- Crack area measurement and bounding box metrics
- Civil engineering structural health classification (PASS / MONITOR / FAIL)
- Lossless RLE compression metrics (Bonus)
- Diagnostic before-and-after visualization panel.
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import cv2

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data import load_image
from src.pipeline import run_pipeline_c
from src.metrics import run_length_encode, compute_compression_ratio


def inspect_image(image_path: str, output_dir: str = None, config_path: str = None) -> dict:
    """
    Executes the full inspection pipeline on a raw image file.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image file not found: {image_path}")

    config = None
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

    # 1. Consistent Loading
    raw_img = load_image(image_path, as_gray=True)

    # 2. Run Pipeline C
    result = run_pipeline_c(raw_img, config=config)

    # 3. Calculate Lossless RLE Compression
    final_mask = result["pred_mask"]
    rle_runs = run_length_encode(final_mask)
    comp_metrics = compute_compression_ratio(final_mask, rle_runs)

    # 4. Assemble Report
    health = result["structural_health"]
    timings = result["stage_timings_ms"]

    print("\n" + "=" * 65)
    print("      CONCRETE CRACK INSPECTION SYSTEM - STRUCTURAL HEALTH REPORT")
    print("=" * 65)
    print(f"Target Image:           {os.path.basename(image_path)}")
    print(f"Image Resolution:       {raw_img.shape[1]} x {raw_img.shape[0]} px")
    print(f"Total Execution Time:   {result['total_time_ms']:.2f} ms")
    print("-" * 65)
    print("STAGE BREAKDOWN:")
    for stage, t in timings.items():
        print(f"  - {stage:<25}: {t:6.2f} ms")
    print("-" * 65)
    print("STRUCTURAL HEALTH ASSESSMENT:")
    print(f"  - Cracked Pixel Count:  {health['crack_pixel_count']:,} px")
    print(f"  - Crack Surface Ratio:  {health['crack_area_percentage']:.4f} %")
    print(f"  - Major Crack Entities: {health['num_crack_components']}")
    print(f"  - Max Crack Length:     {health['max_crack_length_px']} px")
    print(f"  - VERDICT:              [{health['decision']}] - {health['severity']}")
    print(f"  - ACTION:               {health['recommended_action']}")
    print("-" * 65)
    print("BONUS: LOSSLESS RLE COMPRESSION:")
    print(f"  - Original Mask Size:   {comp_metrics['original_bytes']:,} bytes")
    print(f"  - Compressed Size:      {comp_metrics['compressed_bytes']:,} bytes")
    print(f"  - Compression Ratio:    {comp_metrics['compression_ratio']:.2f}x ({comp_metrics['space_savings_percent']:.1f}% savings)")
    print("=" * 65 + "\n")

    # 5. Save Output Panel if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Multi-panel diagnostic overview
        stages = result["intermediate_images"]
        panel_top = np.hstack([
            cv2.cvtColor(stages["input"], cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(stages["enhanced"], cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(stages["spectrum"], cv2.COLOR_GRAY2RGB)
        ])
        panel_bottom = np.hstack([
            cv2.cvtColor(stages["raw_segmented"], cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(stages["final_mask"], cv2.COLOR_GRAY2RGB),
            result["overlay"]
        ])
        diagnostic_panel = np.vstack([panel_top, panel_bottom])

        out_path = os.path.join(output_dir, f"{base_name}_inspection_result.png")
        cv2.imwrite(out_path, diagnostic_panel)
        print(f"Saved diagnostic overview panel to: {out_path}")

    return {
        "result": result,
        "compression": comp_metrics,
        "health": health
    }


def main():
    parser = argparse.ArgumentParser(description="Concrete Crack Inspection Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="results", help="Directory to save output panel")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    args = parser.parse_args()

    inspect_image(args.image, output_dir=args.output, config_path=args.config)


if __name__ == "__main__":
    main()
