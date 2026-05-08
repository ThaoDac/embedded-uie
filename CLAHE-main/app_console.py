import os
import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from memory_profiler import memory_usage
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    print("Warning: memory_profiler not available, RAM usage tracking will be disabled")

import clahe

# Import metrics functions directly to avoid torch dependency in main
from metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_uiqm,
    calculate_uciqe,
    calculate_niqe,
    calculate_energy_consumption
)


class CLAHEProcessor:
    def __init__(self, clip_limit=2.0, tile_size=8):
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        print(f"CLAHE Processor initialized")
        print(f"  Clip Limit: {clip_limit}")
        print(f"  Tile Grid Size: {tile_size}x{tile_size}")

        # Calculate model complexity metrics
        print("\nModel Complexity Analysis:")
        # CLAHE is not a neural network model, so we provide algorithmic complexity
        print(f"  Algorithm: CLAHE (Contrast Limited Adaptive Histogram Equalization)")
        print(f"  Model Type: Traditional CV Algorithm (no trainable parameters)")
        print(f"  FLOPs: ~{self._estimate_flops():.3f} MFLOPs (per 256x256 image)")
        print(f"  Parameters: 0.000 M (no neural network)")
        print(f"  Model Memory: 0.00 MB (no model weights)\n")

    def _estimate_flops(self):
        """Estimate FLOPs for CLAHE algorithm on 256x256 image"""
        # CLAHE complexity estimation:
        # - Histogram computation per tile: O(pixels_per_tile * 256)
        # - Clipping and redistribution: O(256 * num_tiles)
        # - Interpolation: O(total_pixels * 4)
        H, W = 256, 256
        tiles_h = H // self.tile_size
        tiles_w = W // self.tile_size
        num_tiles = tiles_h * tiles_w
        pixels_per_tile = self.tile_size * self.tile_size

        # Rough FLOP estimate (per channel)
        hist_ops = num_tiles * pixels_per_tile * 256
        clip_ops = num_tiles * 256 * 10
        interp_ops = H * W * 20
        total_ops_per_channel = hist_ops + clip_ops + interp_ops
        total_ops = total_ops_per_channel * 3  # 3 channels

        return total_ops / 1e6  # Convert to MFLOPs

    def preprocess_image(self, image, target_size=(256, 256)):
        """Resize image to target size"""
        resized = cv2.resize(image, target_size)
        return resized

    def postprocess_image(self, output_bgr):
        """Convert to RGB"""
        return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

    def _process_image_core(self, image):
        """Core image processing function for memory profiling"""
        pipeline_start = time.time()

        # Preprocessing
        preprocess_start = time.time()
        img_resized = self.preprocess_image(image)
        preprocess_time = time.time() - preprocess_start

        # CLAHE Processing
        inference_start = time.time()
        output_bgr = clahe.RecoverCLAHE(
            img_resized,
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_size, self.tile_size)
        )
        inference_time = time.time() - inference_start

        # Postprocessing
        postprocess_start = time.time()
        output_rgb = self.postprocess_image(output_bgr)
        postprocess_time = time.time() - postprocess_start

        total_time = time.time() - pipeline_start

        return output_rgb, output_bgr, inference_time, preprocess_time, postprocess_time, total_time

    def process_image(self, image):
        """Process image with memory tracking"""
        if MEMORY_PROFILER_AVAILABLE:
            mem_before = memory_usage()[0]

            # Run core processing and track memory
            mem_usage_list = memory_usage(
                (self._process_image_core, (image,)),
                interval=0.001,
                timeout=None,
                max_usage=True
            )
            mem_peak = mem_usage_list if isinstance(mem_usage_list, (int, float)) else max(mem_usage_list)

            # Calculate memory used only for image processing
            ram_used_mb = mem_peak - mem_before
        else:
            ram_used_mb = 0.0

        # Get the actual results
        output_rgb, output_bgr, inference_time, preprocess_time, postprocess_time, total_time = self._process_image_core(image)

        return output_rgb, output_bgr, inference_time, preprocess_time, postprocess_time, total_time, ram_used_mb


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
    all_memory_usage = []
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

        # Process image
        output_rgb, output_bgr, inference_time, preprocess_time, postprocess_time, total_time, ram_used_mb = processor.process_image(image)
        all_inference_times.append(inference_time)
        all_total_times.append(total_time)
        all_preprocess_times.append(preprocess_time)
        all_postprocess_times.append(postprocess_time)
        all_ram_usage.append(ram_used_mb)

        # Measure memory usage per image
        memory_per_image = ram_used_mb
        all_memory_usage.append(memory_per_image)

        # Resize output to original size
        output_rgb_resized = cv2.resize(output_rgb, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        output_bgr_final = cv2.cvtColor(output_rgb_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_bgr_final)

        # Load Ground Truth (if provided)
        if gt_dir:
            gt_path = os.path.join(gt_dir, img_file)
            if os.path.exists(gt_path):
                gt_img = cv2.imread(gt_path)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                # Resize GT to match enhanced output
                gt_img = cv2.resize(gt_img, (output_rgb.shape[1], output_rgb.shape[0]))
            else:
                print(f"⚠️ GT not found for {img_file}, skipping PSNR/SSIM")
                gt_img = None
        else:
            gt_img = None

        # Compute metrics (CLAHE doesn't have tensor output, so NIQE uses numpy)
        img_metrics = {}

        # PSNR and SSIM (if GT available)
        if gt_img is not None:
            try:
                img_metrics['psnr'] = calculate_psnr(gt_img, output_rgb)
            except Exception as e:
                img_metrics['psnr'] = -1.0

            try:
                img_metrics['ssim'] = calculate_ssim(gt_img, output_rgb)
            except Exception as e:
                img_metrics['ssim'] = -1.0
        else:
            img_metrics['psnr'] = -1.0
            img_metrics['ssim'] = -1.0

        # No-reference metrics
        try:
            img_metrics['uiqm'] = calculate_uiqm(output_rgb)
        except Exception as e:
            img_metrics['uiqm'] = -1.0

        try:
            img_metrics['uciqe'] = calculate_uciqe(output_rgb)
        except Exception as e:
            img_metrics['uciqe'] = -1.0

        try:
            img_metrics['niqe'] = calculate_niqe(output_rgb, device=None)
        except Exception as e:
            img_metrics['niqe'] = -1.0

        all_metrics.append(img_metrics)

    # FPS Calculation
    fps_metrics = calculate_video_fps_metrics(all_inference_times)

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
    model_size_mb = 0.0  # CLAHE has no model file
    avg_inference_time = np.mean(all_inference_times) if all_inference_times else 0.0
    avg_total_time = np.mean(all_total_times) if all_total_times else 0.0
    avg_preprocess_time = np.mean(all_preprocess_times) if all_preprocess_times else 0.0
    avg_postprocess_time = np.mean(all_postprocess_times) if all_postprocess_times else 0.0

    # Memory
    avg_memory = np.mean(all_memory_usage) if all_memory_usage else 0.0
    avg_ram_usage = np.mean(all_ram_usage) if all_ram_usage else 0.0
    ram_usage_std = np.std(all_ram_usage) if all_ram_usage else 0.0

    # Energy
    avg_energy, avg_battery = calculate_energy_consumption(avg_total_time)

    # FLOPs
    flops_mflops = processor._estimate_flops()

    avg_metrics['inference_time'] = avg_inference_time
    avg_metrics['total_time'] = avg_total_time
    avg_metrics['preprocess_time'] = avg_preprocess_time
    avg_metrics['postprocess_time'] = avg_postprocess_time
    avg_metrics['model_size_mb'] = model_size_mb
    avg_metrics['memory_mb'] = avg_memory
    avg_metrics['ram_usage_mb'] = avg_ram_usage
    avg_metrics['ram_usage_std'] = ram_usage_std
    avg_metrics['energy_joules'] = avg_energy
    avg_metrics['battery_wh'] = avg_battery
    avg_metrics['flops_mflops'] = flops_mflops

    # Display results
    print("\n" + "=" * 70)
    print("BATCH ENHANCEMENT RESULTS")
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
    print(f"  Avg Inference Time    : {avg_metrics['inference_time']:.4f} s  (CLAHE processing)")
    print(f"  Avg Total Time        : {avg_metrics['total_time']:.4f} s  (full pipeline)")
    print(f"  Avg FPS (1/inference) : {fps_metrics['avg_fps']:.2f}")
    print("-" * 70)
    print("DEVICE PERFORMANCE METRICS (Per Image Average)")
    print(f"  Model Size            : {avg_metrics['model_size_mb']:.2f} MB  (no model file)")
    print(f"  FLOPs                 : {avg_metrics['flops_mflops']:.2f} MFLOPs  (algorithm complexity)")
    print(f"  Memory Usage          : {avg_metrics['memory_mb']:.2f} MB  (avg per image)")
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


def calculate_video_fps_metrics(frame_times):
    """Calculate FPS metrics from frame processing times"""
    if len(frame_times) == 0:
        return {
            'total_frames': 0,
            'frames_used_for_fps': 0,
            'first_frame_time': 0,
            'total_time': 0,
            'avg_fps': 0,
            'min_fps': 0,
            'max_fps': 0
        }

    first_frame_time = frame_times[0]
    frames_for_fps = frame_times[1:] if len(frame_times) > 1 else frame_times

    if len(frames_for_fps) == 0:
        frames_for_fps = frame_times

    total_time = sum(frames_for_fps)
    avg_fps = len(frames_for_fps) / total_time if total_time > 0 else 0
    min_fps = 1.0 / max(frames_for_fps) if frames_for_fps else 0
    max_fps = 1.0 / min(frames_for_fps) if frames_for_fps else 0

    return {
        'total_frames': len(frame_times),
        'frames_used_for_fps': len(frames_for_fps),
        'first_frame_time': first_frame_time,
        'total_time': total_time,
        'avg_fps': avg_fps,
        'min_fps': min_fps,
        'max_fps': max_fps
    }


def main():
    parser = argparse.ArgumentParser(description="CLAHE CLI for Underwater Image Enhancement")
    parser.add_argument('--input', type=str, required=True, help='Input image directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--gt', type=str, default=None, help='Ground truth directory (optional)')
    parser.add_argument('--clip_limit', type=float, default=2.0, help='CLAHE clip limit (default: 2.0)')
    parser.add_argument('--tile_size', type=int, default=8, help='CLAHE tile grid size (default: 8)')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("CLAHE - Contrast Limited Adaptive Histogram Equalization")
    print("=" * 70)

    processor = CLAHEProcessor(clip_limit=args.clip_limit, tile_size=args.tile_size)
    process_directory(processor, args.input, args.output, gt_dir=args.gt)


if __name__ == "__main__":
    main()
