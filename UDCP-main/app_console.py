import os
import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from memory_profiler import memory_usage

from udcp import process_udcp
import metrics


class UDCPProcessor:
    def __init__(self, block_size=15):
        """
        Initialize UDCP processor.

        Args:
            block_size: Size of local patch for dark channel computation (default: 15)
        """
        self.block_size = block_size
        print(f"UDCP Processor initialized with block_size={block_size}")
        print("Note: UDCP is a non-parametric algorithm (no neural network)")

    def _process_image_core(self, image):
        """Core image processing function for memory profiling"""
        pipeline_start = time.time()

        # Convert BGR to RGB
        preprocess_start = time.time()
        input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocess_time = time.time() - preprocess_start

        # UDCP Processing
        inference_start = time.time()
        result = process_udcp(input_rgb, blockSize=self.block_size)
        inference_time = time.time() - inference_start

        # Extract output
        postprocess_start = time.time()
        output_rgb = result['output']
        postprocess_time = time.time() - postprocess_start

        total_time = time.time() - pipeline_start

        return output_rgb, input_rgb, inference_time, result, preprocess_time, postprocess_time, total_time

    def process_image(self, image):
        """
        Process single image with UDCP algorithm.

        Args:
            image: Input BGR image (numpy array)

        Returns:
            Tuple of (output_rgb, input_rgb, processing_time, intermediate_results, ram_used_mb)
        """
        mem_before = memory_usage()[0]

        # Run core processing and track memory
        mem_usage_list = memory_usage((self._process_image_core, (image,)), interval=0.001, timeout=None, max_usage=True)
        mem_peak = mem_usage_list if isinstance(mem_usage_list, (int, float)) else max(mem_usage_list)

        # Calculate memory used only for image processing
        ram_used_mb = mem_peak - mem_before

        # Get the actual results by running the function again
        output_rgb, input_rgb, inference_time, result, preprocess_time, postprocess_time, total_time = self._process_image_core(image)

        return output_rgb, input_rgb, inference_time, result, preprocess_time, postprocess_time, total_time, ram_used_mb


def estimate_udcp_flops(image_shape, block_size=15):
    """
    Estimate FLOPs for UDCP (Underwater Dark Channel Prior) algorithm (corrected version).

    UDCP operations (Drews Jr. et al., 2013):
    1. Underwater Dark Channel: min over G,B channels (not RGB) + erosion
    2. Atmospheric Light Estimation
    3. Transmission Estimation
    4. Guided Filter for transmission refinement
    5. Scene Radiance Recovery

    Args:
        image_shape: Input image shape (H, W, C)
        block_size: Block size for dark channel (default: 15)

    Returns:
        float: Estimated FLOPs in MFLOPs (changed from GFLOPs for consistency)

    Note:
        UDCP differs from DCP by using only G,B channels (not R) for underwater scenes.
        Guided filter uses box filter optimization O(1) per pixel.
    """
    H, W, C = image_shape
    N = H * W

    # 1. Underwater Dark Channel computation
    # Min over G,B channels (exclude red channel for underwater)
    dark_channel_ops = N * 1  # 1 comparison (min of 2 values)
    # Erosion using minimum filter with separable optimization
    dark_channel_ops += N * (2 * block_size)  # separable filter ~2*block_size ops

    # 2. Atmospheric Light estimation
    # Find brightest pixels in dark channel, average their RGB values
    n_top = max(int(N * 0.001), 1)  # top 0.1% pixels
    atmospheric_light_ops = N * 2  # find top 0.1% bright pixels
    atmospheric_light_ops += n_top * 3  # average RGB

    # 3. Transmission estimation
    # Normalize by atmospheric light
    transmission_ops = N * 2 * 2  # divide by A for G,B channels
    # Dark channel on normalized image
    transmission_ops += N * 1  # min over G,B
    transmission_ops += N * (2 * block_size)  # erosion
    # t(x) = 1 - omega * dark(x)
    transmission_ops += N * 2  # multiply + subtract

    # 4. Guided Filter for transmission refinement (CORRECTED)
    # Box filter optimization: O(1) per pixel using integral image
    # 6 box filters needed: mean_I, mean_p, mean_Ip, mean_II, mean_a, mean_b
    guided_filter_ops = N * 4 * 6  # 6 box filters at ~4 ops/pixel
    # Per-pixel computations: variance, covariance, coefficients
    guided_filter_ops += N * 10

    # 5. Scene Radiance Recovery
    # J(x) = (I(x) - A) / max(t(x), t0) + A for each channel
    recovery_ops = N * C * 5  # subtract, max, divide, add per channel

    # Total FLOPs
    total_ops = (dark_channel_ops + atmospheric_light_ops +
                 transmission_ops + guided_filter_ops + recovery_ops)

    # Convert to MFLOPs (changed from GFLOPs for consistency with other implementations)
    mflops = total_ops / 1e6

    return mflops


def process_directory(processor, input_dir, output_dir, gt_dir=None):
    """
    Process all images in a directory with UDCP algorithm.

    Args:
        processor: UDCPProcessor instance
        input_dir: Input directory path
        output_dir: Output directory path
        gt_dir: Ground truth directory path (optional)
    """
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
    all_flops = []

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

        # --- UDCP processing on resized image
        output_rgb, input_rgb, inference_time, result, preprocess_time, postprocess_time, total_time, ram_used_mb = processor.process_image(image_resized)
        all_inference_times.append(inference_time)
        all_total_times.append(total_time)
        all_preprocess_times.append(preprocess_time)
        all_postprocess_times.append(postprocess_time)
        all_ram_usage.append(ram_used_mb)

        # Measure memory usage
        memory_per_image = metrics.get_memory_usage()
        all_memory_usage.append(memory_per_image)

        # Estimate FLOPs for 256x256 image (standardized)
        flops = estimate_udcp_flops(image_resized.shape, processor.block_size)
        all_flops.append(flops)

        # --- Save enhanced result (resize to original dimensions)
        output_rgb_resized = cv2.resize(output_rgb, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        output_bgr = cv2.cvtColor(output_rgb_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_bgr)

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

        # --- Compute image quality metrics (use output_rgb_resized which is at original size)
        img_metrics = {}

        # PSNR and SSIM (if GT available)
        if gt_img is not None:
            try:
                img_metrics['psnr'] = metrics.calculate_psnr(gt_img, output_rgb_resized)
            except:
                img_metrics['psnr'] = -1.0

            try:
                img_metrics['ssim'] = metrics.calculate_ssim(gt_img, output_rgb_resized)
            except:
                img_metrics['ssim'] = -1.0
        else:
            img_metrics['psnr'] = -1.0
            img_metrics['ssim'] = -1.0

        # No-reference metrics
        try:
            img_metrics['uiqm'] = metrics.calculate_uiqm(output_rgb_resized)
        except:
            img_metrics['uiqm'] = -1.0

        try:
            img_metrics['uciqe'] = metrics.calculate_uciqe(output_rgb_resized)
        except:
            img_metrics['uciqe'] = -1.0

        try:
            img_metrics['niqe'] = metrics.calculate_niqe(output_rgb_resized)
        except:
            img_metrics['niqe'] = -1.0

        all_metrics.append(img_metrics)

    # --- Calculate FPS metrics (using inference time only)
    fps_metrics = calculate_video_fps_metrics(all_inference_times)

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
    avg_memory = np.mean(all_memory_usage) if all_memory_usage else 0.0
    avg_flops = np.mean(all_flops) if all_flops else 0.0

    # RAM Usage: AVERAGE RAM per image (from memory_profiler)
    avg_ram_usage = np.mean(all_ram_usage) if all_ram_usage else 0.0
    ram_usage_std = np.std(all_ram_usage) if all_ram_usage else 0.0

    # Energy consumption
    avg_energy, avg_battery = metrics.calculate_energy_consumption(avg_total_time)

    avg_metrics['inference_time'] = avg_inference_time
    avg_metrics['total_time'] = avg_total_time
    avg_metrics['preprocess_time'] = avg_preprocess_time
    avg_metrics['postprocess_time'] = avg_postprocess_time
    avg_metrics['model_size_mb'] = 0.0  # No model file for UDCP
    avg_metrics['memory_mb'] = avg_memory
    avg_metrics['ram_usage_mb'] = avg_ram_usage
    avg_metrics['ram_usage_std'] = ram_usage_std
    avg_metrics['energy_joules'] = avg_energy
    avg_metrics['battery_wh'] = avg_battery
    avg_metrics['flops_mflops'] = avg_flops

    # --- Display results
    print("\n" + "=" * 70)
    print("BATCH ENHANCEMENT RESULTS")
    print("=" * 70)
    print(f"  Total Images   : {len(image_files)}")
    print(f"  Processed      : {len(all_inference_times)}")
    print("-" * 70)

    # Algorithm complexity
    print("ALGORITHM COMPLEXITY")
    print(f"  Avg FLOPs      : {avg_metrics['flops_mflops']:.2f} MFLOPs")
    print(f"  Block Size     : {processor.block_size}x{processor.block_size}")
    print(f"  Algorithm Type : Non-parametric (no neural network)")
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
    print(f"  Avg Inference Time    : {avg_metrics['inference_time']:.4f} s  (UDCP algorithm)")
    print(f"  Avg Total Time        : {avg_metrics['total_time']:.4f} s  (full pipeline)")
    print(f"  Avg FPS (1/inference) : {fps_metrics['avg_fps']:.2f}")
    print("-" * 70)
    print("DEVICE PERFORMANCE METRICS (Per Image Average)")
    print(f"  Model Size            : N/A (non-parametric algorithm)")
    print(f"  Memory Usage          : {avg_metrics['memory_mb']:.2f} MB  (avg per image)")
    print(f"  RAM Usage (Profiler)  : {avg_metrics['ram_usage_mb']:.2f} ± {avg_metrics['ram_usage_std']:.2f} MB  (image processing only)")
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
    """
    Calculate FPS metrics from list of frame processing times.
    Excludes first frame (warmup) from FPS calculation.

    Args:
        frame_times: List of processing times per frame

    Returns:
        dict: FPS metrics
    """
    if not frame_times:
        return {
            'total_frames': 0,
            'frames_used_for_fps': 0,
            'first_frame_time': 0.0,
            'total_time': 0.0,
            'avg_fps': 0.0,
            'min_fps': 0.0,
            'max_fps': 0.0
        }

    first_frame_time = frame_times[0]

    if len(frame_times) > 1:
        # Exclude first frame for FPS calculation
        fps_times = frame_times[1:]
        fps_values = [1.0 / t if t > 0 else 0.0 for t in fps_times]
        avg_fps = np.mean(fps_values)
        min_fps = np.min(fps_values)
        max_fps = np.max(fps_values)
        frames_for_fps = len(fps_times)
    else:
        avg_fps = 1.0 / first_frame_time if first_frame_time > 0 else 0.0
        min_fps = avg_fps
        max_fps = avg_fps
        frames_for_fps = 1

    return {
        'total_frames': len(frame_times),
        'frames_used_for_fps': frames_for_fps,
        'first_frame_time': first_frame_time,
        'total_time': sum(frame_times),
        'avg_fps': avg_fps,
        'min_fps': min_fps,
        'max_fps': max_fps
    }


def process_video(processor, video_path, output_dir):
    """
    Process video file with UDCP algorithm.

    Args:
        processor: UDCPProcessor instance
        video_path: Input video path
        output_dir: Output directory for frames
    """
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
        output_rgb, _, _, _, _, _, _ = processor.process_image(frame)
        frame_times.append(time.time() - start)

        enhanced_frame = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:06d}.png"), enhanced_frame)

    cap.release()

    fps_metrics = calculate_video_fps_metrics(frame_times)
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
    parser = argparse.ArgumentParser(description="UDCP CLI for Underwater Image Enhancement")
    parser.add_argument('--input', type=str, required=True, help='Input image directory or video file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--gt', type=str, default=None, help='Ground truth directory (optional)')
    parser.add_argument('--block_size', type=int, default=15, help='Block size for dark channel (default: 15)')
    parser.add_argument('--video', action='store_true', help='Process as video input')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("UDCP - Underwater Dark Channel Prior")
    print("=" * 70)
    print("Algorithm: Non-parametric image processing")
    print("Reference: Drews Jr. et al., 2013")
    print("=" * 70)

    processor = UDCPProcessor(block_size=args.block_size)

    if args.video:
        process_video(processor, args.input, args.output)
    else:
        process_directory(processor, args.input, args.output, gt_dir=args.gt)


if __name__ == "__main__":
    main()
