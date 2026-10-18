# CrackForest Benchmark Dataset (CFD) for Concrete Crack Inspection

## 1. Overview
This dataset consists of 118 high-resolution pavement and concrete surface images along with pixel-level manual ground truth binary segmentation masks.

- **Origin**: CrackForest Dataset by Cui et al. (Institute of Remote Sensing and Digital Earth, Chinese Academy of Sciences).
- **Application**: Pavement and concrete structural surface defect detection and damage assessment.
- **Image Resolution**: 480 × 320 pixels, 24-bit RGB (converted to standardized 8-bit grayscale for processing).
- **Mask Format**: Binary PNG masks where `0` denotes background (sound concrete) and `255` denotes cracked pixels.

## 2. Real-World Imperfections ("Real-World Mess")
The dataset satisfies all requirements for real-world complexity:
1. **Low Contrast**: Hairline cracks exhibit minimal intensity differences compared to surrounding dark aggregate.
2. **Uneven Illumination**: Shading gradients and directional solar shadows cross the concrete surface.
3. **Aggregate Texture Noise**: Granular pavement roughness and micro-pores produce high-frequency background noise.
4. **Structural Discontinuities**: Weathered cracks have broken segments, hairline branches, and varying widths.

## 3. Ground Truth Schema
- `data/raw/images/*.jpg`: Input RGB images named `001.jpg` through `118.jpg`.
- `data/raw/masks/*.png`: Ground truth binary masks named `001.png` through `118.png`.

## 4. Academic License & Citation
The dataset is publicly released for academic research and education:
> Cui, L., et al. "Pavement crack detection based on saliency and statistical features." IEEE International Conference on Image Processing (ICIP), 2015.
