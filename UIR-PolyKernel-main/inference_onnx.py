#!/usr/bin/env python3
"""
UIR-PolyKernel ONNX Inference Script
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm
import torch
from memory_profiler import memory_usage

import metrics


class UIRPolyKernelOnnx:
    """UIR-PolyKernel ONNX Inference Engine"""

    def __init__(self, onnx_path='models/UIR_PolyKernel_epoch_37.onnx', device='cpu', warmup=True):
        self.onnx_path = onnx_path
        self.device = device

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        print(f"Loading UIR-PolyKernel ONNX model from {onnx_path}...")
        print(f"Using device: {device}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4

        available_providers = ort.get_available_providers()
        print(f"\nAvailable ONNX Runtime providers: {available_providers}")

        if device == 'cuda':
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print(f"✓ Using CUDAExecutionProvider (GPU)")
            else:
                providers = ['CPUExecutionProvider']
                print(f"\n{'='*70}")
                print(f"⚠️  WARNING: CUDA requested but CUDAExecutionProvider not available!")
                print(f"{'='*70}")
                print(f"Fallback: Using CPUExecutionProvider")
                print(f"{'='*70}\n")
        else:
            providers = ['CPUExecutionProvider']
            print(f"✓ Using CPUExecutionProvider (CPU)")

        self.session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"✓ Model loaded successfully!")
        print(f"  - Input name: {self.input_name}")
        print(f"  - Output name: {self.output_name}")
        print(f"  - Providers: {self.session.get_providers()}")

        model_size = os.path.getsize(onnx_path) / (1024 ** 2)
        print(f"\nModel Information:")
        print(f"  - Model Size: {model_size:.2f} MB")

        if warmup:
            print(f"\nRunning warm-up inference...")
            dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
            _ = self.session.run([self.output_name], {self.input_name: dummy_input})
            print(f"✓ Warm-up completed!\n")

    def preprocess(self, image, target_size=(256, 256)):
        resized = cv2.resize(image, target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        batch = np.expand_dims(chw, axis=0)
        return batch, rgb

    def postprocess(self, output_tensor):
        output = output_tensor.squeeze(0)
        hwc = np.transpose(output, (1, 2, 0))
        output_img = np.clip(hwc * 255, 0, 255).astype(np.uint8)
        return output_img

    def _enhance_core(self, image):
        """Core enhancement function for memory profiling"""
        pipeline_start = time.time()

        preprocess_start = time.time()
        input_tensor, input_rgb = self.preprocess(image)
        preprocess_time = time.time() - preprocess_start

        inference_start = time.time()
        ort_inputs = {self.input_name: input_tensor}
        ort_outputs = self.session.run([self.output_name], ort_inputs)
        inference_time = time.time() - inference_start

        postprocess_start = time.time()
        enhanced_rgb = self.postprocess(ort_outputs[0])
        postprocess_time = time.time() - postprocess_start

        total_time = time.time() - pipeline_start
        output_tensor = ort_outputs[0]

        return enhanced_rgb, input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time

    def enhance(self, image):
        """Enhance image with memory profiling"""
        mem_before = memory_usage()[0]
        mem_usage_list = memory_usage((self._enhance_core, (image,)), interval=0.001, timeout=None, max_usage=True)
        mem_peak = mem_usage_list if isinstance(mem_usage_list, (int, float)) else max(mem_usage_list)
        ram_used_mb = mem_peak - mem_before

        enhanced_rgb, input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time = self._enhance_core(image)
        return enhanced_rgb, input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time, ram_used_mb


def process_directory(model, input_dir, output_dir, gt_dir=None):
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

    torch_device = torch.device('cuda' if model.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    for idx, img_file in enumerate(tqdm(image_files, desc="Processing images")):
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, f"enhanced_{img_file}")

        image = cv2.imread(input_path)
        if image is None:
            print(f"⚠️ Skipping {img_file} (unreadable)")
            continue

        orig_h, orig_w = image.shape[:2]

        output_rgb, input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time, ram_used_mb = model.enhance(image)

        if idx > 0:
            all_inference_times.append(inference_time)
            all_total_times.append(total_time)
            all_preprocess_times.append(preprocess_time)
            all_postprocess_times.append(postprocess_time)
            all_ram_usage.append(ram_used_mb)

        if idx == 0:
            onnx_memory = metrics.get_model_size(model.onnx_path)
            all_memory_usage.append(onnx_memory)
        else:
            all_memory_usage.append(onnx_memory)

        output_rgb_resized = cv2.resize(output_rgb, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        output_bgr = cv2.cvtColor(output_rgb_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_bgr)

        gt_img = None
        if gt_dir:
            gt_path = os.path.join(gt_dir, img_file)
            if os.path.exists(gt_path):
                gt_img = cv2.imread(gt_path)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                gt_img = cv2.resize(gt_img, (output_rgb.shape[1], output_rgb.shape[0]))

        output_torch = torch.from_numpy(output_tensor).to(torch_device)

        img_metrics = metrics.evaluate_all_image_metrics(
            input_img=gt_img,
            output_img=output_rgb,
            output_tensor=output_torch,
            device=torch_device
        )

        all_metrics.append(img_metrics)

    fps_metrics = metrics.calculate_video_fps_metrics(all_inference_times)

    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            vals = [m.get(key, -1) for m in all_metrics if m.get(key, -1) > 0]
            if vals:
                avg_metrics[key] = float(np.mean(vals))
                avg_metrics[f"{key}_std"] = float(np.std(vals))
            else:
                avg_metrics[key] = -1.0
                avg_metrics[f"{key}_std"] = 0.0

    model_size_mb = metrics.get_model_size(model.onnx_path)
    avg_inference_time = np.mean(all_inference_times) if all_inference_times else 0.0
    std_inference_time = np.std(all_inference_times) if all_inference_times else 0.0
    avg_total_time = np.mean(all_total_times) if all_total_times else 0.0
    avg_preprocess_time = np.mean(all_preprocess_times) if all_preprocess_times else 0.0
    avg_postprocess_time = np.mean(all_postprocess_times) if all_postprocess_times else 0.0

    avg_memory = np.mean(all_memory_usage) if all_memory_usage else 0.0
    avg_ram_usage = np.mean(all_ram_usage) if all_ram_usage else 0.0
    ram_usage_std = np.std(all_ram_usage) if all_ram_usage else 0.0
    avg_energy, avg_battery = metrics.calculate_energy_consumption(avg_total_time)

    avg_metrics['inference_time'] = avg_inference_time
    avg_metrics['inference_time_std'] = std_inference_time
    avg_metrics['total_time'] = avg_total_time
    avg_metrics['preprocess_time'] = avg_preprocess_time
    avg_metrics['postprocess_time'] = avg_postprocess_time
    avg_metrics['model_size_mb'] = model_size_mb
    avg_metrics['memory_mb'] = avg_memory
    avg_metrics['energy_joules'] = avg_energy
    avg_metrics['battery_wh'] = avg_battery

    print("\n" + "=" * 70)
    print("BATCH ENHANCEMENT RESULTS (UIR-PolyKernel ONNX)")
    print("=" * 70)
    print(f"  Total Images   : {len(image_files)}")
    print(f"  Processed      : {len(image_files)}")
    print(f"  Used for Timing: {len(all_inference_times)} (excluded 1st frame warm-up)")
    print("-" * 70)

    if gt_dir and avg_metrics.get('psnr', -1) > 0:
        print("FULL-REFERENCE QUALITY METRICS (Average vs Ground Truth)")
        print(f"  PSNR           : {avg_metrics.get('psnr', -1):.9f} ± {avg_metrics.get('psnr_std', 0):.9f} dB")
        print(f"  SSIM           : {avg_metrics.get('ssim', -1):.9f} ± {avg_metrics.get('ssim_std', 0):.9f}")
        print("-" * 70)

    if avg_metrics.get('uciqe', -1) > 0 or avg_metrics.get('uiqm', -1) > 0 or avg_metrics.get('niqe', -1) > 0:
        print("NO-REFERENCE QUALITY METRICS (Average)")
        if avg_metrics.get('uciqe', -1) > 0:
            print(f"  UCIQE          : {avg_metrics.get('uciqe', -1):.9f} ± {avg_metrics.get('uciqe_std', 0):.9f}")
        if avg_metrics.get('uiqm', -1) > 0:
            print(f"  UIQM           : {avg_metrics.get('uiqm', -1):.9f} ± {avg_metrics.get('uiqm_std', 0):.9f}")
        if avg_metrics.get('niqe', -1) > 0:
            print(f"  NIQE           : {avg_metrics.get('niqe', -1):.9f} ± {avg_metrics.get('niqe_std', 0):.9f} (lower is better)")
        print("-" * 70)

    print("PERFORMANCE (Inference only - 1st frame excluded)")
    print("  Total Frames   : {}".format(fps_metrics['total_frames']))
    print("  Total Time     : {:.2f} s".format(fps_metrics['total_time']))
    print("  Avg FPS        : {:.2f}".format(fps_metrics['avg_fps']))
    print("  Min FPS        : {:.2f}".format(fps_metrics['min_fps']))
    print("  Max FPS        : {:.2f}".format(fps_metrics['max_fps']))
    print("-" * 70)
    print(f"  RAM Usage (Profiler)  : {avg_ram_usage:.2f} ± {ram_usage_std:.2f} MB (image processing only)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="UIR-PolyKernel ONNX Inference")
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--gt', type=str, default=None, help='Ground truth directory (optional)')
    parser.add_argument('--model', type=str, default='models/UIR_PolyKernel_epoch_37.onnx', help='Path to ONNX model')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device (cpu/cuda)')

    args = parser.parse_args()

    model = UIRPolyKernelOnnx(onnx_path=args.model, device=args.device)

    if os.path.isdir(args.input):
        process_directory(model, args.input, args.output, gt_dir=args.gt)
    else:
        print("Single image mode not implemented yet. Use directory mode.")


if __name__ == "__main__":
    main()
