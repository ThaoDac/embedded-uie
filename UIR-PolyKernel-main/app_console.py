import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import warnings
from memory_profiler import memory_usage

from accelerate import Accelerator
from config import Config
from models import UIR_PolyKernel
from utils import load_checkpoint, seed_everything
import metrics

warnings.filterwarnings('ignore')


class UIRPolyKernelProcessor:
    def __init__(self, model_path, config_path='config.yml', device=None):
        self.model_path = model_path
        self.config_path = config_path

        # Load config
        self.opt = Config(config_path)
        seed_everything(self.opt.OPTIM.SEED)

        # Setup device
        if device is None:
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
        else:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
            self.accelerator = None

        self.model = None
        self.load_model()

    def load_model(self):
        print(f"Loading model from {self.model_path}...")
        print(f"Using device: {self.device}")

        self.model = UIR_PolyKernel()

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load checkpoint
        load_checkpoint(self.model, self.model_path, device=self.device)

        if self.accelerator:
            self.model = self.accelerator.prepare(self.model)
        else:
            self.model = self.model.to(self.device)

        self.model.eval()
        print("Model loaded successfully!")

        # Calculate model complexity metrics
        print("\nModel Complexity Analysis:")
        flops_info = metrics.calculate_flops_accurate(self.model, input_shape=(1, 3, 256, 256), device=self.device)
        print(f"  FLOPs (GFLOPs)     : {flops_info['flops']:.3f}")
        print(f"  MACs (GMACs)       : {flops_info['macs']:.3f}")
        print(f"  Parameters (M)     : {flops_info['params']:.3f}")

        # Get model memory usage
        model_memory = metrics.get_memory_usage(self.model)
        print(f"  Model Memory (MB)  : {model_memory:.2f}\n")

    def preprocess_image(self, image, target_size=(256, 256)):
        """
        Preprocess image for UIR-PolyKernel model.

        Args:
            image: Input image (BGR format from cv2)
            target_size: Target size for resizing

        Returns:
            tensor: Preprocessed image tensor
            rgb: RGB image for visualization
        """
        resized = cv2.resize(image, target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor, rgb

    def postprocess_image(self, output_tensor):
        """
        Postprocess model output tensor to image.

        Args:
            output_tensor: Model output tensor

        Returns:
            numpy array: Output image (RGB, uint8)
        """
        output_np = output_tensor.squeeze(0).detach().cpu().numpy()
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

        # Inference (model only)
        inference_start = time.time()
        with torch.no_grad():
            output_tensor = self.model(img_tensor)
        inference_time = time.time() - inference_start

        # Postprocessing
        postprocess_start = time.time()
        output_rgb = self.postprocess_image(output_tensor)
        postprocess_time = time.time() - postprocess_start

        total_time = time.time() - pipeline_start

        return output_rgb, input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time

    def process_image(self, image):
        """
        Process a single image through the entire pipeline.

        Args:
            image: Input image (BGR format)

        Returns:
            tuple: (output_rgb, input_rgb, inference_time, output_tensor,
                   preprocess_time, postprocess_time, total_time, ram_used_mb)
        """
        mem_before = memory_usage()[0]

        # Run core processing and track memory
        mem_usage_list = memory_usage((self._process_image_core, (image,)), interval=0.001, timeout=None, max_usage=True)
        mem_peak = mem_usage_list if isinstance(mem_usage_list, (int, float)) else max(mem_usage_list)

        # Calculate memory used only for image processing
        ram_used_mb = mem_peak - mem_before

        # Get the actual results by running the function again
        output_rgb, input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time = self._process_image_core(image)

        return output_rgb, input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time, ram_used_mb


def process_directory(processor, input_dir, output_dir, gt_dir=None):
    """
    Process all images in a directory.

    Args:
        processor: UIRPolyKernelProcessor instance
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

    for img_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, f"enhanced_{img_file}")
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

        # Measure memory usage per image
        memory_per_image = metrics.get_memory_usage(model=processor.model)
        all_memory_usage.append(memory_per_image)

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
                # Resize GT to match enhanced output
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
            output_tensor=output_tensor,
            device=processor.device
        )

        all_metrics.append(img_metrics)

    # FPS Calculation (model inference only)
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
    model_size_mb = metrics.get_model_size(processor.model_path)
    avg_inference_time = np.mean(all_inference_times) if all_inference_times else 0.0
    avg_total_time = np.mean(all_total_times) if all_total_times else 0.0
    avg_preprocess_time = np.mean(all_preprocess_times) if all_preprocess_times else 0.0
    avg_postprocess_time = np.mean(all_postprocess_times) if all_postprocess_times else 0.0

    # Memory: AVERAGE memory per image
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
    avg_metrics['model_size_mb'] = model_size_mb
    avg_metrics['memory_mb'] = avg_memory
    avg_metrics['ram_usage_mb'] = avg_ram_usage
    avg_metrics['ram_usage_std'] = ram_usage_std
    avg_metrics['energy_joules'] = avg_energy
    avg_metrics['battery_wh'] = avg_battery

    # Display results
    print("\n" + "=" * 70)
    print("BATCH ENHANCEMENT RESULTS (UIR-PolyKernel)")
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
    print("  Total Time     : {:.2f} s".format(fps_metrics['total_time']))
    print("  Avg FPS        : {:.2f}".format(fps_metrics['avg_fps']))
    print("  Min FPS        : {:.2f}".format(fps_metrics['min_fps']))
    print("  Max FPS        : {:.2f}".format(fps_metrics['max_fps']))
    print("=" * 70)
    print("PERFORMANCE METRICS")
    print(f"  Avg Inference Time    : {avg_metrics['inference_time']:.4f} s  (model only)")
    print(f"  Avg Total Time        : {avg_metrics['total_time']:.4f} s  (full pipeline)")
    print(f"  Avg FPS (1/inference) : {fps_metrics['avg_fps']:.2f}")
    print("-" * 70)
    print("DEVICE PERFORMANCE METRICS (Per Image Average)")
    print(f"  Model Size            : {avg_metrics['model_size_mb']:.2f} MB  (pretrained model file)")
    print(f"  Memory Usage          : {avg_metrics['memory_mb']:.2f} MB  (avg: model + tensors per image)")
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


def main():
    parser = argparse.ArgumentParser(description="UIR-PolyKernel CLI for Underwater Image Enhancement")
    parser.add_argument('--input', type=str, required=True, help='Input image directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--gt', type=str, default=None, help='Ground truth directory (optional)')
    parser.add_argument('--model', type=str, default='models/UIR_PolyKernel_epoch_37.pth', help='Model path')
    parser.add_argument('--config', type=str, default='config.yml', help='Config file path')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu, default: auto)')

    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    else:
        device = None  # Will use Accelerator

    processor = UIRPolyKernelProcessor(model_path=args.model, config_path=args.config, device=device)
    process_directory(processor, args.input, args.output, gt_dir=args.gt)


if __name__ == "__main__":
    main()
