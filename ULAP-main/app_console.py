import os
import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from memory_profiler import memory_usage

from ulap import process_ulap
import metrics


class ULAPProcessor:
    def __init__(self):
        print("Initializing ULAP processor...")
        print("ULAP is a traditional algorithm (no model file required)")
        print("Algorithm: Underwater Light Attenuation Prior (Song et al. 2018)")

    def _process_image_core(self, image):
        """Core image processing function for memory profiling"""
        pipeline_start = time.time()

        # Preprocessing (minimal for ULAP)
        preprocess_start = time.time()
        input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocess_time = time.time() - preprocess_start

        # ULAP Processing (depth estimation + restoration)
        inference_start = time.time()
        result = process_ulap(image, blockSize=9, gimfiltR=50, eps=0.001)
        inference_time = time.time() - inference_start

        # Postprocessing (convert output)
        postprocess_start = time.time()
        output_bgr = result['output']
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        postprocess_time = time.time() - postprocess_start

        total_time = time.time() - pipeline_start

        return output_rgb, input_rgb, inference_time, output_bgr, preprocess_time, postprocess_time, total_time

    def process_image(self, image):
        """Process single image with ULAP"""
        mem_before = memory_usage()[0]

        # Run core processing and track memory
        mem_usage_list = memory_usage((self._process_image_core, (image,)), interval=0.001, timeout=None, max_usage=True)
        mem_peak = mem_usage_list if isinstance(mem_usage_list, (int, float)) else max(mem_usage_list)

        # Calculate memory used only for image processing
        ram_used_mb = mem_peak - mem_before

        # Get the actual results by running the function again
        output_rgb, input_rgb, inference_time, output_bgr, preprocess_time, postprocess_time, total_time = self._process_image_core(image)

        return output_rgb, input_rgb, inference_time, output_bgr, preprocess_time, postprocess_time, total_time, ram_used_mb


def estimate_ulap_flops(image_shape):
    """
    Estimate FLOPs for ULAP algorithm (corrected version).

    ULAP Pipeline:
    1. Underwater Light Attenuation Prior (depth estimation from color channels)
    2. Global Histogram Stretching
    3. Guided Image Filter (refined depth map)
    4. Background Light Estimation
    5. Transmission Maps (per channel with Beer-Lambert law)
    6. Scene Radiance Recovery

    Args:
        image_shape: (H, W, C) of input image

    Returns:
        mflops: Estimated FLOPs in millions

    Note:
        ULAP uses guided filter with box filter optimization O(1) per pixel.
    """
    h, w, c = image_shape
    pixels = h * w

    # 1. Depth estimation using Underwater Light Attenuation Prior
    # Depth = max(R,G,B) - min(R,G,B) or similar linear combination
    depth_flops = pixels * 4  # max + min operations across RGB

    # 2. Global Histogram Stretching (contrast enhancement)
    # Compute histogram, cumulative distribution, mapping
    hist_flops = pixels * 3  # histogram lookup per pixel
    hist_flops += 256 * 2  # cumulative sum (256 bins)
    hist_flops += pixels * 2  # remap each pixel

    # 3. Guided Image Filter on depth map (r=50, eps=0.001)
    # Box filter optimization: O(1) per pixel using integral image
    # 6 box filters (mean_I, mean_p, mean_Ip, mean_II, mean_a, mean_b)
    guided_filter_flops = pixels * 4 * 6  # 6 box filters at ~4 ops/pixel each
    guided_filter_flops += pixels * 10  # variance, covariance, coefficients, output

    # 4. Background light estimation
    # Maximum intensity in dark channel or brightest pixels
    bg_light_flops = pixels * 2  # find max values
    bg_light_flops += 100 * 3  # average top pixels

    # 5. Transmission maps for 3 channels (Beer-Lambert law)
    # t_c(x) = exp(-beta_c * d(x)) for each channel c
    # Approximation of exp using lookup table or polynomial
    transmission_flops = pixels * c * 6  # exp approximation per channel (~6 ops)
    transmission_flops += pixels * c * 2  # multiply by beta, negate

    # 6. Scene radiance recovery
    # J_c(x) = (I_c(x) - A_c) / max(t_c(x), t0) + A_c for each channel
    recovery_flops = pixels * c * 5  # subtract, max, divide, add

    total_flops = (depth_flops + hist_flops + guided_filter_flops +
                   bg_light_flops + transmission_flops + recovery_flops)

    # Convert to MFLOPs
    mflops = total_flops / 1e6

    return mflops


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
    all_flops = []
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

        # --- ULAP processing on resized image
        output_rgb, input_rgb, inference_time, output_bgr, preprocess_time, postprocess_time, total_time, ram_used_mb = processor.process_image(image_resized)
        all_inference_times.append(inference_time)
        all_total_times.append(total_time)
        all_preprocess_times.append(preprocess_time)
        all_postprocess_times.append(postprocess_time)
        all_ram_usage.append(ram_used_mb)

        # Estimate FLOPs for 256x256 image (standardized)
        flops_mflops = estimate_ulap_flops(image_resized.shape)
        all_flops.append(flops_mflops)

        # Measure memory usage (process memory)
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            all_memory_usage.append(memory_mb)
        except ImportError:
            # Fallback if psutil not available
            all_memory_usage.append(0.0)

        # --- Save enhanced result
        # Resize output to original size
        output_rgb_resized = cv2.resize(output_rgb, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        output_bgr_resized = cv2.cvtColor(output_rgb_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_bgr_resized)

        # --- Load Ground Truth (if provided)
        if gt_dir:
            gt_path = os.path.join(gt_dir, img_file)
            if os.path.exists(gt_path):
                gt_img = cv2.imread(gt_path)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                # Resize GT to match original size
                gt_img = cv2.resize(gt_img, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            else:
                print(f"⚠️ GT not found for {img_file}, skipping PSNR/SSIM")
                gt_img = None
        else:
            gt_img = None

        # --- Compute metrics (use output_rgb_resized which is at original size)
        img_metrics = {
            'psnr': metrics.calculate_psnr(gt_img, output_rgb_resized) if gt_img is not None else -1,
            'ssim': metrics.calculate_ssim(gt_img, output_rgb_resized) if gt_img is not None else -1,
            'uiqm': metrics.calculate_uiqm(output_rgb_resized),
            'uciqe': metrics.calculate_uciqe(output_rgb_resized),
            'niqe': metrics.calculate_niqe(output_rgb_resized)
        }

        all_metrics.append(img_metrics)

    # --- FPS Calculation
    fps_metrics = metrics.calculate_video_fps_metrics(all_inference_times)

    # --- Average metrics with standard deviation
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

    # --- Calculate performance metrics
    avg_inference_time = np.mean(all_inference_times) if all_inference_times else 0.0
    avg_total_time = np.mean(all_total_times) if all_total_times else 0.0
    avg_preprocess_time = np.mean(all_preprocess_times) if all_preprocess_times else 0.0
    avg_postprocess_time = np.mean(all_postprocess_times) if all_postprocess_times else 0.0
    avg_flops = np.mean(all_flops) if all_flops else 0.0

    # Memory: AVERAGE memory
    avg_memory = np.mean(all_memory_usage) if all_memory_usage else 0.0

    # RAM Usage: AVERAGE RAM per image (from memory_profiler)
    avg_ram_usage = np.mean(all_ram_usage) if all_ram_usage else 0.0
    ram_usage_std = np.std(all_ram_usage) if all_ram_usage else 0.0

    # Energy: AVERAGE energy per image
    avg_energy, avg_battery = metrics.calculate_energy_consumption(avg_total_time)

    avg_metrics['inference_time'] = avg_inference_time
    avg_metrics['total_time'] = avg_total_time
    avg_metrics['preprocess_time'] = avg_preprocess_time
    avg_metrics['postprocess_time'] = avg_postprocess_time
    avg_metrics['model_size_mb'] = 0.0  # No model file
    avg_metrics['memory_mb'] = avg_memory
    avg_metrics['ram_usage_mb'] = avg_ram_usage
    avg_metrics['ram_usage_std'] = ram_usage_std
    avg_metrics['energy_joules'] = avg_energy
    avg_metrics['battery_wh'] = avg_battery
    avg_metrics['flops_mflops'] = avg_flops

    # --- Display results
    print("\n" + "=" * 70)
    print("BATCH ENHANCEMENT RESULTS (ULAP)")
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
    print(f"  Avg Inference Time    : {avg_metrics['inference_time']:.4f} s  (ULAP algorithm)")
    print(f"  Avg Total Time        : {avg_metrics['total_time']:.4f} s  (full pipeline)")
    print(f"  Avg FPS (1/inference) : {fps_metrics['avg_fps']:.2f}")
    print("-" * 70)
    print("DEVICE PERFORMANCE METRICS (Per Image Average)")
    print(f"  Model Size            : N/A (traditional algorithm, no model file)")
    print(f"  Avg FLOPs             : {avg_metrics['flops_mflops']:.2f} MFLOPs  (estimated)")
    print(f"  Memory Usage          : {avg_metrics['memory_mb']:.2f} MB  (process memory)")
    print(f"  RAM Usage (Profiler)  : {avg_metrics['ram_usage_mb']:.2f} ± {avg_metrics['ram_usage_std']:.2f} MB  (image processing only)")
    print(f"  Energy Consumption    : {avg_metrics['energy_joules']:.2f} J ({avg_metrics['battery_wh']:.6f} Wh)")
    print(f"                          (avg per image: preprocess + inference + postprocess)")
    print("-" * 70)
    print("TIMING BREAKDOWN")
    print(f"  Preprocess  : {avg_metrics['preprocess_time']:.4f} s")
    print(f"  Inference   : {avg_metrics['inference_time']:.4f} s  (ULAP algorithm)")
    print(f"  Postprocess : {avg_metrics['postprocess_time']:.4f} s")
    print(f"  Total       : {avg_metrics['total_time']:.4f} s")
    print("=" * 70)


def process_video(processor, video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nEnhancing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("✗ Cannot open video.")
        return

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_times = []
    frame_idx = 0

    print(f"Input video FPS: {fps_in:.2f}, total frames: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        start = time.time()

        # Process frame with ULAP
        result = process_ulap(frame, blockSize=9, gimfiltR=50, eps=0.001)
        enhanced_frame = result['output']

        frame_times.append(time.time() - start)
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:06d}.png"), enhanced_frame)

    cap.release()

    fps_metrics = metrics.calculate_video_fps_metrics(frame_times)
    print("\n" + "=" * 70)
    print("VIDEO INFERENCE PERFORMANCE")
    print("=" * 70)
    print(f"  Total Frames  : {fps_metrics['total_frames']}")
    print(f"  Total Time    : {fps_metrics['total_time']:.2f} s")
    print(f"  Avg FPS       : {fps_metrics['avg_fps']:.2f}")
    print(f"  Min FPS       : {fps_metrics['min_fps']:.2f}")
    print(f"  Max FPS       : {fps_metrics['max_fps']:.2f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="ULAP CLI for Underwater Image Enhancement")
    parser.add_argument('--input', type=str, required=True, help='Input image directory or video file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--gt', type=str, default=None, help='Ground truth directory (optional)')
    parser.add_argument('--video', action='store_true', help='Process as video input')

    args = parser.parse_args()
    processor = ULAPProcessor()

    if args.video:
        process_video(processor, args.input, args.output)
    else:
        process_directory(processor, args.input, args.output, gt_dir=args.gt)


if __name__ == "__main__":
    main()
