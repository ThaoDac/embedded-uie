import os
import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from memory_profiler import memory_usage

import metrics
from global_stretching_RGB import stretching
from LabStretching import LABStretching


class RGHSProcessor:
    def __init__(self):
        """RGHS is a non-model-based method, no model loading needed"""
        print("RGHS Image Enhancement Processor initialized")
        print("Method: Relative Global Histogram Stretching")
        print("Paper: Huang et al. 2018 - Shallow-water Image Enhancement\n")

    def preprocess_image(self, image, target_size=(256, 256)):
        """Resize image to 256x256 as specified"""
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb

    def postprocess_image(self, output_rgb):
        """Convert output to uint8 format"""
        output_rgb = np.clip(output_rgb, 0, 255)
        return output_rgb.astype(np.uint8)

    def _process_image_core(self, image):
        """Core RGHS processing function"""
        pipeline_start = time.time()

        # Preprocessing - resize to 256x256
        preprocess_start = time.time()
        input_rgb = self.preprocess_image(image)
        preprocess_time = time.time() - preprocess_start

        # RGHS Enhancement (contrast correction in RGB + color correction in Lab)
        inference_start = time.time()

        # Step 1: Global histogram stretching in RGB space
        enhanced_rgb = stretching(input_rgb)

        # Step 2: Lab color space stretching
        enhanced_rgb = LABStretching(enhanced_rgb)

        inference_time = time.time() - inference_start

        # Postprocessing
        postprocess_start = time.time()
        output_rgb = self.postprocess_image(enhanced_rgb)
        postprocess_time = time.time() - postprocess_start

        total_time = time.time() - pipeline_start

        return output_rgb, input_rgb, inference_time, preprocess_time, postprocess_time, total_time

    def process_image(self, image):
        """Process image with memory tracking"""
        mem_before = memory_usage()[0]

        # Run core processing and track memory
        mem_usage_list = memory_usage((self._process_image_core, (image,)), interval=0.001, timeout=None, max_usage=True)
        mem_peak = mem_usage_list if isinstance(mem_usage_list, (int, float)) else max(mem_usage_list)

        # Calculate memory used for image processing
        ram_used_mb = mem_peak - mem_before

        # Get actual results
        output_rgb, input_rgb, inference_time, preprocess_time, postprocess_time, total_time = self._process_image_core(image)

        return output_rgb, input_rgb, inference_time, preprocess_time, postprocess_time, total_time, ram_used_mb


def process_directory(processor, input_dir, output_dir, gt_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nEnhancing images from: {input_dir}")
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

        # RGHS enhancement
        output_rgb, input_rgb, inference_time, preprocess_time, postprocess_time, total_time, ram_used_mb = processor.process_image(image)
        all_inference_times.append(inference_time)
        all_total_times.append(total_time)
        all_preprocess_times.append(preprocess_time)
        all_postprocess_times.append(postprocess_time)
        all_ram_usage.append(ram_used_mb)

        # Resize output to original size
        output_rgb_resized = cv2.resize(output_rgb, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        # Save enhanced result
        output_bgr = cv2.cvtColor(output_rgb_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_bgr)

        # Load Ground Truth (if provided)
        if gt_dir:
            gt_path = os.path.join(gt_dir, img_file)
            if os.path.exists(gt_path):
                gt_img = cv2.imread(gt_path)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                # Resize GT to match enhanced output (256x256)
                gt_img = cv2.resize(gt_img, (output_rgb.shape[1], output_rgb.shape[0]))
            else:
                print(f"⚠️ GT not found for {img_file}, skipping PSNR/SSIM")
                gt_img = None
        else:
            gt_img = None

        # Compute metrics (use 256x256 output for fair comparison)
        # For NIQE, pass the numpy array directly (metrics module will handle conversion)
        img_metrics = metrics.evaluate_all_image_metrics(
            input_img=gt_img,
            output_img=output_rgb,
            output_tensor=output_rgb,  # Pass numpy array for NIQE calculation
            device=None
        )

        all_metrics.append(img_metrics)

    # FPS Calculation
    fps_metrics = metrics.calculate_video_fps_metrics(all_inference_times)

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

    # Performance metrics
    avg_inference_time = np.mean(all_inference_times) if all_inference_times else 0.0
    avg_total_time = np.mean(all_total_times) if all_total_times else 0.0
    avg_preprocess_time = np.mean(all_preprocess_times) if all_preprocess_times else 0.0
    avg_postprocess_time = np.mean(all_postprocess_times) if all_postprocess_times else 0.0

    # RAM Usage
    avg_ram_usage = np.mean(all_ram_usage) if all_ram_usage else 0.0
    ram_usage_std = np.std(all_ram_usage) if all_ram_usage else 0.0

    # Energy
    avg_energy, avg_battery = metrics.calculate_energy_consumption(avg_total_time)

    # FLOPs calculation (simplified for non-DL method)
    try:
        # Estimate FLOPs for traditional image processing
        # RGHS involves histogram operations, stretching, color space conversions
        image_shape = (256, 256, 3)
        estimated_ops = np.prod(image_shape) * 50  # Rough estimate for histogram + stretching operations
        flops_gflops = estimated_ops / 1e9
        print(f"\nEstimated FLOPs: {flops_gflops:.3f} GFLOPs (traditional method, not DL)")
    except Exception as e:
        print(f"Could not estimate FLOPs: {e}")
        flops_gflops = -1.0

    avg_metrics['inference_time'] = avg_inference_time
    avg_metrics['total_time'] = avg_total_time
    avg_metrics['preprocess_time'] = avg_preprocess_time
    avg_metrics['postprocess_time'] = avg_postprocess_time
    avg_metrics['model_size_mb'] = 0.0  # RGHS has no model
    avg_metrics['ram_usage_mb'] = avg_ram_usage
    avg_metrics['ram_usage_std'] = ram_usage_std
    avg_metrics['energy_joules'] = avg_energy
    avg_metrics['battery_wh'] = avg_battery
    avg_metrics['flops_gflops'] = flops_gflops

    # Display results
    print("\n" + "=" * 70)
    print("BATCH ENHANCEMENT RESULTS - RGHS METHOD")
    print("=" * 70)
    print(f"  Total Images   : {len(image_files)}")
    print(f"  Processed      : {len(all_inference_times)}")
    print("-" * 70)

    # Full-reference metrics
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
    print(f"  Avg Inference Time    : {avg_metrics['inference_time']:.4f} s  (RGHS algorithm)")
    print(f"  Avg Total Time        : {avg_metrics['total_time']:.4f} s  (full pipeline)")
    print(f"  Avg FPS (1/inference) : {fps_metrics['avg_fps']:.2f}")
    print("-" * 70)
    print("DEVICE PERFORMANCE METRICS (Per Image Average)")
    print(f"  Model Size            : {avg_metrics['model_size_mb']:.2f} MB  (no model - traditional method)")
    if flops_gflops > 0:
        print(f"  Estimated FLOPs       : {flops_gflops:.3f} GFLOPs  (traditional method)")
    print(f"  RAM Usage (Profiler)  : {avg_metrics['ram_usage_mb']:.2f} ± {avg_metrics['ram_usage_std']:.2f} MB")
    print(f"  Energy Consumption    : {avg_metrics['energy_joules']:.2f} J ({avg_metrics['battery_wh']:.6f} Wh)")
    print(f"                          (avg per image: preprocess + inference + postprocess)")
    print("-" * 70)
    print("TIMING BREAKDOWN")
    print(f"  Preprocess  : {avg_metrics['preprocess_time']:.4f} s")
    print(f"  Inference   : {avg_metrics['inference_time']:.4f} s")
    print(f"  Postprocess : {avg_metrics['postprocess_time']:.4f} s")
    print(f"  Total       : {avg_metrics['total_time']:.4f} s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="RGHS CLI for Underwater Image Enhancement")
    parser.add_argument('--input', type=str, required=True, help='Input image directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--gt', type=str, default=None, help='Ground truth directory (optional)')

    args = parser.parse_args()
    processor = RGHSProcessor()

    process_directory(processor, args.input, args.output, gt_dir=args.gt)


if __name__ == "__main__":
    main()
