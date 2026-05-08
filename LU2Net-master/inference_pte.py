import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from executorch.extension.pybindings.portable_lib import _load_for_executorch

import metrics


class LU2NetPTEProcessor:
    def __init__(self, model_path='/home/ndpthao/eject/IMPLEMENTATION/LU2Net-master/checkpoints/LightUNet_200.pte', device=None):
        self.model_path = model_path
        self.device = device if device else 'cpu'
        self.model = None
        self.load_model()

    def load_model(self):
        print(f"Loading ExecuTorch model from {self.model_path}...")
        print(f"Using device: {self.device}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = _load_for_executorch(self.model_path)
        print("ExecuTorch model loaded successfully!")

        # Get model file size
        model_size = os.path.getsize(self.model_path) / (1024 * 1024)
        print(f"\nModel File Size: {model_size:.2f} MB\n")

    def preprocess_image(self, image, target_size=(256, 256)):
        resized = cv2.resize(image, target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0

        # CRITICAL: Create tensor in NCHW format directly from numpy to match torch.randn() memory layout
        # Convert HWC (256, 256, 3) to CHW (3, 256, 256) then add batch dimension
        chw = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        nchw = np.expand_dims(chw, axis=0)  # CHW -> NCHW (1, 3, 256, 256)

        # Create tensor from NCHW numpy array (same layout as torch.randn)
        tensor = torch.from_numpy(nchw.copy())  # .copy() ensures C-contiguous array

        return tensor, rgb

    def postprocess_image(self, output_tensor):
        output_np = output_tensor.squeeze(0).numpy()
        output_np = np.clip(output_np, 0, 1)
        output_np = np.transpose(output_np, (1, 2, 0))
        return (output_np * 255).astype(np.uint8)

    def _process_image_core(self, image):
        """Core image processing function for memory profiling"""
        pipeline_start = time.time()

        # Preprocessing
        preprocess_start = time.time()
        img_tensor, input_rgb = self.preprocess_image(image)
        preprocess_time = time.time() - preprocess_start

        # Inference (ExecuTorch)
        inference_start = time.time()
        output_tensor = self.model.forward([img_tensor])[0]
        inference_time = time.time() - inference_start

        # Postprocessing
        postprocess_start = time.time()
        output_rgb = self.postprocess_image(output_tensor)
        postprocess_time = time.time() - postprocess_start

        total_time = time.time() - pipeline_start

        return output_rgb, input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time

    def process_image(self, image):
        # OPTIMIZED: Run inference only ONCE, not twice
        # Memory profiling is disabled for accurate FPS measurement
        # (memory_profiler has significant overhead and runs the function twice)

        # Run inference once and get results
        output_rgb, input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time = self._process_image_core(image)

        # Estimate RAM usage instead of profiling (to avoid 2x slowdown)
        # Approximate RAM = model size + input tensor + output tensor
        model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        input_size_mb = (image.size * image.itemsize) / (1024 * 1024)
        output_size_mb = (output_rgb.size * output_rgb.itemsize) / (1024 * 1024)
        ram_used_mb = model_size_mb + input_size_mb + output_size_mb

        return output_rgb, input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time, ram_used_mb


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

        # Model inference
        output_rgb, input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time, ram_used_mb = processor.process_image(image)
        all_inference_times.append(inference_time)
        all_total_times.append(total_time)
        all_preprocess_times.append(preprocess_time)
        all_postprocess_times.append(postprocess_time)
        all_ram_usage.append(ram_used_mb)

        # Model memory (file size approximation for ExecuTorch)
        model_memory = os.path.getsize(processor.model_path) / (1024 * 1024)
        all_memory_usage.append(model_memory)

        # Resize output to original size
        output_rgb_resized = cv2.resize(output_rgb, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        output_bgr = cv2.cvtColor(output_rgb_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_bgr)

        # Load Ground Truth (if provided)
        if gt_dir:
            gt_path = os.path.join(gt_dir, img_file)
            if os.path.exists(gt_path):
                gt_img = cv2.imread(gt_path)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                gt_img = cv2.resize(gt_img, (output_rgb.shape[1], output_rgb.shape[0]))
            else:
                print(f"⚠️ GT not found for {img_file}, skipping PSNR/SSIM")
                gt_img = None
        else:
            gt_img = None

        # Compute metrics
        img_metrics = metrics.evaluate_all_image_metrics(
            input_img=gt_img,
            output_img=output_rgb,
            output_tensor=torch.from_numpy(output_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
            device='cpu'
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

    # Calculate performance metrics
    model_size_mb = os.path.getsize(processor.model_path) / (1024 * 1024)
    avg_inference_time = np.mean(all_inference_times) if all_inference_times else 0.0
    avg_total_time = np.mean(all_total_times) if all_total_times else 0.0
    avg_preprocess_time = np.mean(all_preprocess_times) if all_preprocess_times else 0.0
    avg_postprocess_time = np.mean(all_postprocess_times) if all_postprocess_times else 0.0

    avg_memory = np.mean(all_memory_usage) if all_memory_usage else 0.0
    avg_ram_usage = np.mean(all_ram_usage) if all_ram_usage else 0.0
    ram_usage_std = np.std(all_ram_usage) if all_ram_usage else 0.0

    avg_energy, avg_battery = metrics.calculate_energy_consumption(avg_total_time)

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

    # Display results
    print("\n" + "=" * 70)
    print("BATCH ENHANCEMENT RESULTS (ExecuTorch)")
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
    print(f"  Avg Inference Time    : {avg_metrics['inference_time']:.4f} s  (model only)")
    print(f"  Avg Total Time        : {avg_metrics['total_time']:.4f} s  (full pipeline)")
    print(f"  Avg FPS (1/inference) : {fps_metrics['avg_fps']:.2f}")
    print("-" * 70)
    print("DEVICE PERFORMANCE METRICS (Per Image Average)")
    print(f"  Model Size            : {avg_metrics['model_size_mb']:.2f} MB  (ExecuTorch .pte file)")
    print(f"  Memory Usage          : {avg_metrics['memory_mb']:.2f} MB  (avg: model file size)")
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
        img_tensor, _ = processor.preprocess_image(frame)
        output_tensor = processor.model.forward([img_tensor])[0]
        frame_times.append(time.time() - start)
        enhanced_frame = processor.postprocess_image(output_tensor)
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:06d}.png"),
                    cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR))

    cap.release()

    fps_metrics = metrics.calculate_video_fps_metrics(frame_times)
    print("\n" + "=" * 70)
    print("VIDEO INFERENCE PERFORMANCE (ExecuTorch)")
    print("=" * 70)
    print(f"  Total Frames  : {fps_metrics['total_frames']}")
    print(f"  Total Time    : {fps_metrics['total_time']:.2f} s")
    print(f"  Avg FPS       : {fps_metrics['avg_fps']:.2f}")
    print(f"  Min FPS       : {fps_metrics['min_fps']:.2f}")
    print(f"  Max FPS       : {fps_metrics['max_fps']:.2f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="LU2Net ExecuTorch CLI for Enhancement")
    parser.add_argument('--input', type=str, required=True, help='Input image directory or video file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--gt', type=str, default=None, help='Ground truth directory (optional)')
    parser.add_argument('--model', type=str, default='/home/ndpthao/eject/IMPLEMENTATION/LU2Net-master/checkpoints/LightUNet_200.pte', help='ExecuTorch model path (.pte)')
    parser.add_argument('--video', action='store_true', help='Process as video input')

    args = parser.parse_args()
    processor = LU2NetPTEProcessor(model_path=args.model)

    if args.video:
        process_video(processor, args.input, args.output)
    else:
        process_directory(processor, args.input, args.output, gt_dir=args.gt)


if __name__ == "__main__":
    main()