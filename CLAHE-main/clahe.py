"""
CLAHE - Contrast Limited Adaptive Histogram Equalization
=========================================================
This module implements CLAHE for underwater image enhancement.
"""

import cv2
import numpy as np


def RecoverCLAHE(img, clipLimit=2.0, tileGridSize=(4, 4)):
    """
    Apply CLAHE to enhance underwater image.

    Args:
        img: Input BGR image (numpy array, 0-255, uint8)
        clipLimit: Threshold for contrast limiting (default: 2.0)
        tileGridSize: Size of grid for histogram equalization (default: (4, 4))

    Returns:
        enhanced: Enhanced image (BGR, uint8)
    """
    print("[CLAHE] Applying Contrast Limited Adaptive Histogram Equalization")
    print(f"   Parameters: clipLimit={clipLimit}, tileGridSize={tileGridSize}")

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    enhanced = img.copy()

    for i in range(3):
        channel_name = ['Blue', 'Green', 'Red'][i]
        print(f"   ✓ Processing {channel_name} channel...")
        enhanced[:, :, i] = clahe.apply(enhanced[:, :, i])

    print("   ✓ CLAHE completed")
    return enhanced


def process_clahe(img_bgr, clipLimit=2.0, tileGridSize=(4, 4)):
    """
    Main CLAHE processing function.

    Args:
        img_bgr: Input BGR image
        clipLimit: CLAHE clip limit
        tileGridSize: CLAHE tile grid size

    Returns:
        dict: Processing results
    """
    print("\n" + "=" * 80)
    print("CLAHE - Contrast Limited Adaptive Histogram Equalization")
    print("=" * 80)

    output = RecoverCLAHE(img_bgr, clipLimit=clipLimit, tileGridSize=tileGridSize)

    print("=" * 80)
    print("✅ Processing Complete")
    print("=" * 80 + "\n")

    return {
        'output': output,
        'clipLimit': clipLimit,
        'tileGridSize': tileGridSize
    }


def compare_histograms(img_before, img_after):
    """Compare histograms before and after enhancement."""
    stats = {}

    for i, channel_name in enumerate(['Blue', 'Green', 'Red']):
        stats[channel_name] = {
            'mean_before': np.mean(img_before[:, :, i]),
            'mean_after': np.mean(img_after[:, :, i]),
            'std_before': np.std(img_before[:, :, i]),
            'std_after': np.std(img_after[:, :, i]),
            'min_before': np.min(img_before[:, :, i]),
            'min_after': np.min(img_after[:, :, i]),
            'max_before': np.max(img_before[:, :, i]),
            'max_after': np.max(img_after[:, :, i])
        }

    return stats