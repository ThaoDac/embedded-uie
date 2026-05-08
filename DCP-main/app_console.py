"""
DCP Console Application
========================
Command-line interface for Dark Channel Prior (DCP) image dehazing.
Processes directories of images with comprehensive metrics evaluation.

Usage:
    python app_console.py --input <input_dir> --output <output_dir> [--gt <gt_dir>]

Example:
    python app_console.py --input test_images --output results --gt ground_truth
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from memory_profiler import memory_usage

from dcp import DCPProcessor

# Import only what we need from metrics (avoid torch dependency)
try:
    from metrics import evaluate_all_image_metrics
except ImportError:
    # Fallback: define minimal metrics without torch
    def evaluate_all_image_metrics(input_img, output_img, output_tensor=None):
        """Minimal metrics without torch dependency."""
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity

        metrics_dict = {
            'psnr': -1.0,
            'ssim': -1.0,
            'uiqm': -1.0,
            'uciqe': -1.0,
            'niqe': -1.0
        }

        # Calculate PSNR and SSIM if ground truth available
        if input_img is not None and output_img is not None:
            try:
                metrics_dict['psnr'] = peak_signal_noise_ratio(input_img, output_img, data_range=255)
                metrics_dict['ssim'] = structural_similarity(input_img, output_img,
                                                            multichannel=True,
                                                            channel_axis=-1,
                                                            data_range=255)
            except:
                pass

        # Calculate UIQM if possible
        try:
            from metrics import calculate_uiqm
            metrics_dict['uiqm'] = calculate_uiqm(output_img)
        except:
            pass

        # Calculate UCIQE if possible
        try:
            from metrics import calculate_uciqe
            metrics_dict['uciqe'] = calculate_uciqe(output_img)
        except:
            pass

        return metrics_dict


def calculate_dcp_flops(image_shape, patch_size=15, guided_r=60):
    """
    Estimate FLOPs for DCP (Dark Channel Prior) algorithm (corrected version).

    DCP Pipeline (He et al., 2011):
    1. Dark Channel computation
    2. Atmospheric Light estimation
    3. Transmission estimation
    4. Guided Filter for transmission refinement
    5. Scene Radiance recovery

    Args:
        image_shape: (H, W, C) of input image
        patch_size: Dark channel patch size (default: 15)
        guided_r: Guided filter radius (default: 60)

    Returns:
        flops_mflops: Estimated FLOPs in millions

    Note:
        Only counts FLOPs for image enhancement, not quality metrics computation.
        Guided filter uses box filter optimization O(1) per pixel, not O(r²).
    """
    h, w, c = image_shape
    n_pixels = h * w

    # 1. Dark channel calculation
    # Min across 3 RGB channels: 2 comparisons per pixel
    flops_dark = n_pixels * 2
    # Erosion (minimum filter) with separable optimization: ~2*patch_size ops/pixel
    flops_dark += n_pixels * (2 * patch_size)

    # 2. Atmospheric light estimation
    # Find top 0.1% brightest pixels in dark channel
    n_top_pixels = max(int(n_pixels * 0.001), 1)
    flops_atm = n_pixels * 2  # find top 0.1% bright pixels (quickselect O(n))
    flops_atm += n_top_pixels * 3  # average RGB values

    # 3. Transmission estimation
    # Normalize by atmospheric light A
    flops_trans = n_pixels * c * 2  # divide I by A for each channel
    # Dark channel on normalized image
    flops_trans += n_pixels * 2  # min across channels
    flops_trans += n_pixels * (2 * patch_size)  # erosion
    # t(x) = 1 - omega * dark(x)
    flops_trans += n_pixels * 2  # multiply + subtract

    # 4. Guided filter (CORRECTED - uses box filter optimization)
    # Box filter is O(1) per pixel using integral image method
    # 6 box filters needed: mean_I, mean_p, mean_Ip, mean_II, mean_a, mean_b
    flops_guided = n_pixels * 4 * 6  # each box filter ~4 ops/pixel
    # Per-pixel computations: variance, covariance, linear coefficients
    flops_guided += n_pixels * 10

    # 5. Scene recovery
    # J(x) = (I(x) - A) / max(t(x), t0) + A for each channel
    flops_recover = n_pixels * c * 5  # subtract, max, divide, add per channel

    total_flops = flops_dark + flops_atm + flops_trans + flops_guided + flops_recover
    flops_mflops = total_flops / 1e6

    return flops_mflops


class DCPConsoleProcessor:
    """Console wrapper for DCP processing with metrics."""

    def __init__(self, patch_size=15, omega=0.95, t0=0.1, guided_r=60, guided_eps=0.0001):
        """Initialize DCP processor with parameters."""
        self.processor = DCPProcessor(
            patch_size=patch_size,
            omega=omega,
            t0=t0,
            guided_r=guided_r,
            guided_eps=guided_eps
        )
        self.patch_size = patch_size
        self.omega = omega
        self.t0 = t0
        self.guided_r = guided_r
        self.guided_eps = guided_eps

    def _process_image_core(self, image):
        """Core image processing function for memory profiling"""
        pipeline_start = time.time()

        # Preprocessing (just timing, DCP handles internally)
        preprocess_start = time.time()
        # No explicit preprocessing needed for DCP
        preprocess_time = time.time() - preprocess_start

        # DCP Processing (main algorithm)
        inference_start = time.time()
        results = self.processor.process(image)
        inference_time = time.time() - inference_start

        # Postprocessing
        postprocess_start = time.time()
        output_bgr = results['recovered_image']  # Already uint8 BGR
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        output_rgb_float = output_rgb.astype(np.float32) / 255.0
        postprocess_time = time.time() - postprocess_start

        total_time = time.time() - pipeline_start

        return output_bgr, output_rgb, inference_time, output_rgb_float, preprocess_time, postprocess_time, total_time

    def process_image(self, image):
        """
        Process single image with timing.

        Args:
            image: Input BGR image (uint8)

        Returns:
            output_bgr: Enhanced BGR image (uint8)
            output_rgb: Enhanced RGB image (uint8)
            inference_time: Processing time (seconds)
            output_rgb_float: Enhanced RGB normalized to [0, 1]
            preprocess_time: Preprocessing time
            postprocess_time: Postprocessing time
            total_time: Total pipeline time
            ram_used_mb: RAM usage (MB)
        """
        mem_before = memory_usage()[0]

        # Run core processing and track memory
        mem_usage_list = memory_usage((self._process_image_core, (image,)), interval=0.001, timeout=None, max_usage=True)
        mem_peak = mem_usage_list if isinstance(mem_usage_list, (int, float)) else max(mem_usage_list)

        # Calculate memory used only for image processing
        ram_used_mb = mem_peak - mem_before

        # Get the actual results by running the function again
        output_bgr, output_rgb, inference_time, output_rgb_float, preprocess_time, postprocess_time, total_time = self._process_image_core(image)

        return output_bgr, output_rgb, inference_time, output_rgb_float, preprocess_time, postprocess_time, total_time, ram_used_mb


def process_directory(processor, input_dir, output_dir, gt_dir=None):
    """
    Process all images in directory with DCP.

    Args:
        processor: DCPConsoleProcessor instance
        input_dir: Input directory path
        output_dir: Output directory path
        gt_dir: Ground truth directory (optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nDehazing images from: {input_dir}")
    if gt_dir:
        print(f"Using Ground Truth folder: {gt_dir}")

    image_files = [f for f in os.listdir(input_dir)
                   if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    all_inference_times = []
    all_total_times = []
    all_preprocess_times = []
    all_postprocess_times = []
    all_flops = []
    all_ram_usage = []
    all_metrics = []

    for img_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, img_file)
        image = cv2.imread(input_path)

        orig_h, orig_w = image.shape[:2]

        if image is None:
            print(f"⚠️ Skipping {img_file} (unreadable)")
            continue

        # Resize to 256x256 for processing (to standardize FLOPs calculation)
        image_resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

        # Process resized image
        output_bgr, output_rgb, inference_time, output_rgb_float, preprocess_time, postprocess_time, total_time, ram_used_mb = processor.process_image(image_resized)

        all_inference_times.append(inference_time)
        all_total_times.append(total_time)
        all_preprocess_times.append(preprocess_time)
        all_postprocess_times.append(postprocess_time)
        all_ram_usage.append(ram_used_mb)

        # Calculate FLOPs for 256x256 image (standardized)
        flops = calculate_dcp_flops(image_resized.shape, processor.patch_size, processor.guided_r)
        all_flops.append(flops)

        # Resize output back to original size before saving
        output_bgr_resized = cv2.resize(output_bgr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        output_rgb_resized = cv2.resize(output_rgb, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(output_path, output_bgr_resized)

        # Load Ground Truth (if provided)
        gt_img = None
        if gt_dir:
            gt_path = os.path.join(gt_dir, img_file)
            if os.path.exists(gt_path):
                gt_img = cv2.imread(gt_path)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                # Resize GT to match output_rgb_resized (original size)
                gt_img = cv2.resize(gt_img, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            else:
                print(f"⚠️ GT not found for {img_file}, skipping PSNR/SSIM")

        # Compute metrics (use resized output for comparison)
        img_metrics = evaluate_all_image_metrics(
            input_img=gt_img,
            output_img=output_rgb_resized,
            output_tensor=output_rgb_float
        )

        all_metrics.append(img_metrics)

    # Calculate FPS metrics
    fps_metrics = calculate_fps_metrics(all_inference_times)

    # Average metrics with standard deviation
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            vals = [m[key] for m in all_metrics if m[key] > 0]
            if vals:
                avg_metrics[key] = np.mean(vals)
                avg_metrics[f'{key}_std'] = np.std(vals)
            else:
                avg_metrics[key] = -1.0
                avg_metrics[f'{key}_std'] = 0.0
    else:
        avg_metrics = {
            'psnr': -1, 'psnr_std': 0,
            'ssim': -1, 'ssim_std': 0,
            'uiqm': -1, 'uiqm_std': 0,
            'uciqe': -1, 'uciqe_std': 0,
            'niqe': -1, 'niqe_std': 0
        }

    # Calculate performance metrics
    avg_inference_time = np.mean(all_inference_times) if all_inference_times else 0.0
    avg_total_time = np.mean(all_total_times) if all_total_times else 0.0
    avg_preprocess_time = np.mean(all_preprocess_times) if all_preprocess_times else 0.0
    avg_postprocess_time = np.mean(all_postprocess_times) if all_postprocess_times else 0.0
    avg_flops = np.mean(all_flops) if all_flops else 0.0

    # RAM Usage: AVERAGE RAM per image (from memory_profiler)
    avg_ram_usage = np.mean(all_ram_usage) if all_ram_usage else 0.0
    ram_usage_std = np.std(all_ram_usage) if all_ram_usage else 0.0

    # Energy calculation (average per image)
    tdp_watts = 15.0  # Typical CPU TDP for image processing
    avg_energy = avg_total_time * tdp_watts  # Joules
    avg_battery = avg_energy / 3600.0  # Wh

    # Memory estimation (DCP is lightweight, mainly image buffers)
    # Estimate: original image + dark channel + transmission maps + guided filter buffers
    sample_h, sample_w = 256, 256  # Assume average resolution
    if image_files:
        sample_img = cv2.imread(os.path.join(input_dir, image_files[0]))
        if sample_img is not None:
            sample_h, sample_w = sample_img.shape[:2]

    memory_mb = (sample_h * sample_w * 3 * 8) / (1024 * 1024)  # 8 buffers (rough estimate)

    avg_metrics['inference_time'] = avg_inference_time
    avg_metrics['total_time'] = avg_total_time
    avg_metrics['preprocess_time'] = avg_preprocess_time
    avg_metrics['postprocess_time'] = avg_postprocess_time
    avg_metrics['flops_mflops'] = avg_flops
    avg_metrics['memory_mb'] = memory_mb
    avg_metrics['ram_usage_mb'] = avg_ram_usage
    avg_metrics['ram_usage_std'] = ram_usage_std
    avg_metrics['energy_joules'] = avg_energy
    avg_metrics['battery_wh'] = avg_battery

    # Display results
    print("\n" + "=" * 70)
    print("BATCH DEHAZING RESULTS (DCP)")
    print("=" * 70)
    print(f"  Total Images   : {len(image_files)}")
    print(f"  Processed      : {len(all_inference_times)}")
    print("-" * 70)

    # Full-reference metrics (if GT available)
    if gt_dir and avg_metrics.get('psnr', -1) > 0:
        print("FULL-REFERENCE QUALITY METRICS (Average vs Ground Truth)")
        print(f"  PSNR           : {avg_metrics.get('psnr', -1):.9f} ± {avg_metrics.get('psnr_std', 0):.9f} dB")
        print(f"  SSIM           : {avg_metrics.get('ssim', -1):.9f} ± {avg_metrics.get('ssim_std', 0):.9f}")
        print("-" * 70)

    # No-reference metrics
    if avg_metrics.get('uciqe', -1) > 0 or avg_metrics.get('uiqm', -1) > 0 or avg_metrics.get('niqe', -1) > 0:
        print("NO-REFERENCE QUALITY METRICS (Average)")
        if avg_metrics.get('uciqe', -1) > 0:
            print(f"  UCIQE          : {avg_metrics.get('uciqe', -1):.9f} ± {avg_metrics.get('uciqe_std', 0):.9f}")
        if avg_metrics.get('uiqm', -1) > 0:
            print(f"  UIQM           : {avg_metrics.get('uiqm', -1):.9f} ± {avg_metrics.get('uiqm_std', 0):.9f}")
        if avg_metrics.get('niqe', -1) > 0:
            print(f"  NIQE           : {avg_metrics.get('niqe', -1):.9f} ± {avg_metrics.get('niqe_std', 0):.9f} (lower is better)")
        print("-" * 70)

    print("PERFORMANCE (Inference only)")
    print("  Total Frames   : {}".format(fps_metrics['total_frames']))
    print("  Frames for FPS : {} (excluding 1st warmup frame)".format(fps_metrics['frames_used_for_fps']))
    print("  1st Frame Time : {:.4f} s (warmup, excluded from FPS)".format(fps_metrics['first_frame_time']))
    print("  Total Time     : {:.2f} s".format(fps_metrics['total_time']))
    print("  Avg FPS        : {:.2f} (from frame 2 onwards)".format(fps_metrics['avg_fps']))
    print("  Min FPS        : {:.2f}".format(fps_metrics['min_fps']))
    print("  Max FPS        : {:.2f}".format(fps_metrics['max_fps']))
    print("=" * 70)
    print("PERFORMANCE METRICS")
    print(f"  Avg Inference Time    : {avg_metrics['inference_time']:.4f} s  (DCP algorithm)")
    print(f"  Avg Total Time        : {avg_metrics['total_time']:.4f} s  (full pipeline)")
    print(f"  Avg FPS (1/inference) : {fps_metrics['avg_fps']:.2f}")
    print("-" * 70)
    print("DEVICE PERFORMANCE METRICS (Per Image Average)")
    print(f"  Algorithm Type        : Classical (No model file)")
    print(f"  FLOPs                 : {avg_metrics['flops_mflops']:.2f} MFLOPs  (estimated)")
    print(f"  Memory Usage          : {avg_metrics['memory_mb']:.2f} MB  (estimated: image buffers)")
    print(f"  RAM Usage (Profiler)  : {avg_metrics['ram_usage_mb']:.2f} ± {avg_metrics['ram_usage_std']:.2f} MB  (image processing only)")
    print(f"  Energy Consumption    : {avg_metrics['energy_joules']:.2f} J ({avg_metrics['battery_wh']:.6f} Wh)")
    print(f"                          (avg per image: full pipeline)")
    print("-" * 70)
    print("TIMING BREAKDOWN")
    print(f"  Preprocess  : {avg_metrics['preprocess_time']:.4f} s")
    print(f"  Inference   : {avg_metrics['inference_time']:.4f} s  (DCP algorithm)")
    print(f"  Postprocess : {avg_metrics['postprocess_time']:.4f} s")
    print(f"  Total       : {avg_metrics['total_time']:.4f} s")
    print("=" * 70)
    print("\nDCP PARAMETERS:")
    print(f"  Patch Size    : {processor.patch_size}")
    print(f"  Omega         : {processor.omega}")
    print(f"  t0            : {processor.t0}")
    print(f"  Guided r      : {processor.guided_r}")
    print(f"  Guided eps    : {processor.guided_eps}")
    print("=" * 70)


def calculate_fps_metrics(frame_times):
    """
    Calculate FPS metrics from frame processing times.

    Args:
        frame_times: List of processing times per frame

    Returns:
        dict with FPS metrics
    """
    if len(frame_times) == 0:
        return {
            'total_frames': 0,
            'frames_used_for_fps': 0,
            'first_frame_time': 0.0,
            'total_time': 0.0,
            'avg_fps': 0.0,
            'min_fps': 0.0,
            'max_fps': 0.0
        }

    total_frames = len(frame_times)
    first_frame_time = frame_times[0] if total_frames > 0 else 0.0

    # Exclude first frame (warmup)
    if total_frames > 1:
        times_for_fps = frame_times[1:]
    else:
        times_for_fps = frame_times

    total_time = sum(times_for_fps)
    frames_used = len(times_for_fps)

    if total_time > 0:
        avg_fps = frames_used / total_time
        individual_fps = [1.0 / t if t > 0 else 0.0 for t in times_for_fps]
        min_fps = min(individual_fps) if individual_fps else 0.0
        max_fps = max(individual_fps) if individual_fps else 0.0
    else:
        avg_fps = 0.0
        min_fps = 0.0
        max_fps = 0.0

    return {
        'total_frames': total_frames,
        'frames_used_for_fps': frames_used,
        'first_frame_time': first_frame_time,
        'total_time': total_time,
        'avg_fps': avg_fps,
        'min_fps': min_fps,
        'max_fps': max_fps
    }


def main():
    """Main entry point for DCP console application."""
    parser = argparse.ArgumentParser(
        description="DCP Console Application - Image Dehazing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app_console.py --input test_images --output results
  python app_console.py --input test_images --output results --gt ground_truth
  python app_console.py --input hazy_imgs --output dehazed --patch_size 21 --omega 0.9
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Input image directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--gt', type=str, default=None,
                        help='Ground truth directory (optional, for PSNR/SSIM)')
    parser.add_argument('--patch_size', type=int, default=15,
                        help='Dark channel patch size (default: 15)')
    parser.add_argument('--omega', type=float, default=0.95,
                        help='Haze retention parameter (default: 0.95)')
    parser.add_argument('--t0', type=float, default=0.1,
                        help='Lower bound for transmission (default: 0.1)')
    parser.add_argument('--guided_r', type=int, default=60,
                        help='Guided filter radius (default: 60)')
    parser.add_argument('--guided_eps', type=float, default=0.0001,
                        help='Guided filter epsilon (default: 0.0001)')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("DCP CONSOLE APPLICATION - DARK CHANNEL PRIOR")
    print("=" * 70)
    print(f"Input Directory  : {args.input}")
    print(f"Output Directory : {args.output}")
    if args.gt:
        print(f"Ground Truth Dir : {args.gt}")
    print("=" * 70)

    # Initialize processor
    processor = DCPConsoleProcessor(
        patch_size=args.patch_size,
        omega=args.omega,
        t0=args.t0,
        guided_r=args.guided_r,
        guided_eps=args.guided_eps
    )

    # Process directory
    process_directory(processor, args.input, args.output, gt_dir=args.gt)

    print("\n✓ Processing completed!")
    print(f"Results saved to: {args.output}\n")


if __name__ == "__main__":
    main()
