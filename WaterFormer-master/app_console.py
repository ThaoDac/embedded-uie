import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from runpy import run_path
from skimage import img_as_ubyte
from memory_profiler import memory_usage

# IMPORTANT: Import custom metrics BEFORE adding waterformer to path
# Otherwise it will import waterformer.metrics instead of root metrics.py
import metrics as custom_metrics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'waterformer'))
from waterformer.utils.options import parse


class WaterFormerProcessor:
    def __init__(self, config_path='./configs/uie_waterformer.yml',
                 model_path='/home/ndpthao/eject/IMPLEMENTATION/WaterFormer-master/work_dirs/UW_WaterFormer/models/net_g_best.pth',
                 device=None, resize_target=None):
        self.config_path = config_path
        self.model_path = model_path
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.img_multiple_of = 8
        self.resize_target = resize_target  # Target size for resizing (e.g., 256), None = keep original
        self.load_model()

    def load_model(self):
        print(f"Loading WaterFormer model...")
        print(f"Config: {self.config_path}")
        print(f"Weights: {self.model_path}")
        print(f"Using device: {self.device}")
        if self.resize_target:
            print(f"🔧 Resize mode enabled: {self.resize_target}x{self.resize_target} (like LU2Net)")

        # Load config
        opt = parse(self.config_path, is_train=False)

        # Get model parameters from config
        parameters = opt['network_g']
        arch_type = parameters.pop('type')

        # Load model architecture
        load_arch = run_path(f'./waterformer/models/archs/{arch_type.lower()}_arch.py')
        self.model = load_arch[arch_type](**parameters)

        # Load weights
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        weights = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(weights['params'], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully!")

        # Calculate model complexity metrics
        print("\nModel Complexity Analysis:")
        flops_info = custom_metrics.calculate_flops_accurate(self.model, input_shape=(1, 3, 256, 256), device=self.device)
        print(f"  FLOPs (GFLOPs)     : {flops_info['flops']:.3f}")
        print(f"  MACs (GMACs)       : {flops_info['macs']:.3f}")
        print(f"  Parameters (M)     : {flops_info['params']:.3f}")

        # Get model memory usage
        model_memory = custom_metrics.get_memory_usage(self.model)
        print(f"  Model Memory (MB)  : {model_memory:.2f}\n")

    def preprocess_image(self, image):
        """Preprocess image: BGR -> RGB -> (optional resize) -> normalize -> tensor with padding"""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Optional: Resize to target size (like LU2Net)
        if self.resize_target is not None:
            original_size = (rgb.shape[0], rgb.shape[1])  # (height, width)
            rgb = cv2.resize(rgb, (self.resize_target, self.resize_target))
        else:
            original_size = None

        # Normalize to [0, 1] and convert to tensor
        input_tensor = torch.from_numpy(rgb).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Pad to multiple of 8
        height, width = input_tensor.shape[2], input_tensor.shape[3]
        H = ((height + self.img_multiple_of) // self.img_multiple_of) * self.img_multiple_of
        W = ((width + self.img_multiple_of) // self.img_multiple_of) * self.img_multiple_of
        padh = H - height if height % self.img_multiple_of != 0 else 0
        padw = W - width if width % self.img_multiple_of != 0 else 0
        input_tensor = F.pad(input_tensor, (0, padw, 0, padh), 'reflect')

        return input_tensor, rgb, (height, width)

    def postprocess_image(self, output_tensor, original_size):
        """Postprocess output: unpad -> clamp -> denormalize -> numpy"""
        height, width = original_size

        # Unpad to original size
        restored = output_tensor[:, :, :height, :width]

        # Clamp to [0, 1]
        restored = torch.clamp(restored, 0, 1)

        # Convert to numpy
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        return restored

    def _process_image_core(self, image, tile_size=None, tile_overlap=32):
        """Core image processing function for memory profiling"""
        pipeline_start = time.time()

        # Preprocessing
        preprocess_start = time.time()
        input_tensor, input_rgb, original_size = self.preprocess_image(image)
        preprocess_time = time.time() - preprocess_start

        # Inference
        inference_start = time.time()
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

            if tile_size is None:
                # Process full image
                output_tensor = self.model(input_tensor)
            else:
                # Process with tiling
                b, c, h, w = input_tensor.shape
                tile = min(tile_size, h, w)
                assert tile % 8 == 0, "tile size should be multiple of 8"

                stride = tile - tile_overlap
                h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
                w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
                E = torch.zeros(b, c, h, w).type_as(input_tensor)
                W = torch.zeros_like(E)

                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = input_tensor[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                        out_patch = self.model(in_patch)
                        out_patch_mask = torch.ones_like(out_patch)

                        E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                        W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
                output_tensor = E.div_(W)

        inference_time = time.time() - inference_start

        # Postprocessing
        postprocess_start = time.time()
        output_rgb = self.postprocess_image(output_tensor, original_size)
        postprocess_time = time.time() - postprocess_start

        total_time = time.time() - pipeline_start

        # Return output for metrics calculation (before unpadding)
        output_tensor_metrics = torch.clamp(output_tensor[:, :, :original_size[0], :original_size[1]], 0, 1)

        return output_rgb, input_rgb, inference_time, output_tensor_metrics, preprocess_time, postprocess_time, total_time

    def process_image(self, image, tile_size=None, tile_overlap=32):
        """Process single image with optional tiling"""
        mem_before = memory_usage()[0]

        # Run core processing and track memory
        mem_usage_list = memory_usage((self._process_image_core, (image, tile_size, tile_overlap)), interval=0.001, timeout=None, max_usage=True)
        mem_peak = mem_usage_list if isinstance(mem_usage_list, (int, float)) else max(mem_usage_list)

        # Calculate memory used only for image processing
        ram_used_mb = mem_peak - mem_before

        # Get the actual results by running the function again
        output_rgb, input_rgb, inference_time, output_tensor_metrics, preprocess_time, postprocess_time, total_time = self._process_image_core(image, tile_size, tile_overlap)

        return output_rgb, input_rgb, inference_time, output_tensor_metrics, preprocess_time, postprocess_time, total_time, ram_used_mb


def process_directory(processor, input_dir, output_dir, gt_dir=None, tile_size=None, tile_overlap=32):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nEnhancing images from: {input_dir}")
    if gt_dir:
        print(f"Using Ground Truth folder: {gt_dir}")
    if processor.resize_target:
        print(f"🔧 Resize mode: All images will be resized to {processor.resize_target}x{processor.resize_target} (like LU2Net)")
    if tile_size:
        print(f"Using tile-based processing: tile_size={tile_size}, overlap={tile_overlap}")

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

        # Clear GPU cache before processing each image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Auto-enable tiling for large images to avoid GPU OOM
        # Skip auto-tiling if using --resize mode (images already resized to small size)
        auto_tile_size = tile_size
        if processor.resize_target is None and tile_size is None and (orig_h > 720 or orig_w > 720):
            auto_tile_size = 512
            if img_file == image_files[0]:  # Only print once
                print(f"\n⚠️ Large image detected ({orig_w}x{orig_h}), auto-enabling tiled inference (tile_size=512)")

        # Model inference
        output_rgb, input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time, ram_used_mb = \
            processor.process_image(image, tile_size=auto_tile_size, tile_overlap=tile_overlap)

        all_inference_times.append(inference_time)
        all_total_times.append(total_time)
        all_preprocess_times.append(preprocess_time)
        all_postprocess_times.append(postprocess_time)
        all_ram_usage.append(ram_used_mb)

        # Measure memory usage per image
        memory_per_image = custom_metrics.get_memory_usage(model=processor.model)
        all_memory_usage.append(memory_per_image)

        # Resize output to original size if needed
        if output_rgb.shape[0] != orig_h or output_rgb.shape[1] != orig_w:
            output_rgb_resized = cv2.resize(output_rgb, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        else:
            output_rgb_resized = output_rgb

        # Save enhanced result (RGB -> BGR)
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
        img_metrics = custom_metrics.evaluate_all_image_metrics(
            input_img=gt_img,
            output_img=output_rgb,
            output_tensor=output_tensor,
            device=processor.device
        )

        all_metrics.append(img_metrics)

        # Clear GPU memory after each image
        del output_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # FPS Calculation (model inference only)
    fps_metrics = custom_metrics.calculate_video_fps_metrics(all_inference_times)

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
    model_size_mb = custom_metrics.get_model_size(processor.model_path)
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
    avg_energy, avg_battery = custom_metrics.calculate_energy_consumption(avg_total_time)

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


def process_video(processor, video_path, output_dir, tile_size=None, tile_overlap=32):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nEnhancing video: {video_path}")
    if tile_size:
        print(f"Using tile-based processing: tile_size={tile_size}, overlap={tile_overlap}")

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

        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

            input_tensor, _, original_size = processor.preprocess_image(frame)

            if tile_size is None:
                output_tensor = processor.model(input_tensor)
            else:
                # Tile-based processing (same as process_image)
                b, c, h, w = input_tensor.shape
                tile = min(tile_size, h, w)
                assert tile % 8 == 0, "tile size should be multiple of 8"

                stride = tile - tile_overlap
                h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
                w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
                E = torch.zeros(b, c, h, w).type_as(input_tensor)
                W = torch.zeros_like(E)

                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = input_tensor[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                        out_patch = processor.model(in_patch)
                        out_patch_mask = torch.ones_like(out_patch)

                        E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                        W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
                output_tensor = E.div_(W)

        frame_times.append(time.time() - start)
        enhanced_frame = processor.postprocess_image(output_tensor, original_size)
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:06d}.png"),
                    cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR))

    cap.release()

    fps_metrics = custom_metrics.calculate_video_fps_metrics(frame_times)
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
    parser = argparse.ArgumentParser(description="WaterFormer CLI for Underwater Image Enhancement")
    parser.add_argument('--input', type=str, required=True, help='Input image directory or video file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--gt', type=str, default=None, help='Ground truth directory (optional)')
    parser.add_argument('--config', type=str, default='./configs/uie_waterformer.yml', help='Config file path')
    parser.add_argument('--model', type=str, default='/home/ndpthao/eject/IMPLEMENTATION/WaterFormer-master/work_dirs/UW_WaterFormer/models/net_g_best.pth', help='Model checkpoint path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--video', action='store_true', help='Process as video input')
    parser.add_argument('--tile', type=int, default=None, help='Tile size for large images (e.g., 720). None means full resolution')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--resize', type=int, default=None, help='Resize images to NxN before processing (e.g., 256). None = keep original size with auto-tiling')

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    processor = WaterFormerProcessor(config_path=args.config, model_path=args.model,
                                    device=device, resize_target=args.resize)

    if args.video:
        process_video(processor, args.input, args.output, tile_size=args.tile, tile_overlap=args.tile_overlap)
    else:
        process_directory(processor, args.input, args.output, gt_dir=args.gt,
                         tile_size=args.tile, tile_overlap=args.tile_overlap)


if __name__ == "__main__":
    main()
