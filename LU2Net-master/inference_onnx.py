"""
Script sử dụng ONNX model để enhance underwater images
Sử dụng ONNX Runtime thay vì PyTorch
Bao gồm đầy đủ metrics đánh giá chất lượng ảnh
"""

import os
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


class LU2NetONNX:
    """LU2Net ONNX Inference Engine"""

    def __init__(self, onnx_path='/home/ndpthao/eject/IMPLEMENTATION/LU2Net-master/checkpoints/LightUNet_200_gem_fp16.onnx', device='cpu', warmup=True):
        """
        Initialize ONNX Runtime session

        Args:
            onnx_path: Path to ONNX model file
            device: 'cpu' or 'cuda' (requires onnxruntime-gpu)
            warmup: Run warm-up inference to initialize session
        """
        self.onnx_path = onnx_path
        self.device = device

        # Kiểm tra file tồn tại
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Tạo ONNX Runtime session
        print(f"Loading ONNX model from {onnx_path}...")
        print(f"Using device: {device}")

        # Configure session options for ARM optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Optimize for Raspberry Pi 5 (4 cores Cortex-A76)
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1

        # Enable memory optimizations
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True

        # Configure providers based on device
        if device == 'cuda':
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                }),
                ('CPUExecutionProvider', {
                    'arena_extend_strategy': 'kSameAsRequested',
                })
            ]
        else:
            # ARM-optimized CPU provider options
            providers = [
                ('CPUExecutionProvider', {
                    'arena_extend_strategy': 'kSameAsRequested',
                })
            ]

        # Try to load model with error handling for quantized models
        try:
            self.session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
        except Exception as e:
            error_msg = str(e)
            if 'ConvInteger' in error_msg or 'NOT_IMPLEMENTED' in error_msg:
                print(f"\n{'='*70}")
                print("⚠️  QUANTIZED MODEL NOT SUPPORTED ON THIS DEVICE")
                print("="*70)
                print(f"\nLỗi: {error_msg[:200]}...")
                print("\nModel này là QUANTIZED (Int8) model sử dụng operators không được")
                print("hỗ trợ trên CPUExecutionProvider hiện tại.")
                print("\nGiải pháp:")
                print("  1. Sử dụng Float32 model thay thế:")
                print(f"     --model checkpoints/LightUNet_200.onnx")
                print("\n  2. HOẶC sử dụng Static Quantization (QDQ format) thay vì Dynamic:")
                print(f"     python onnx_PTQ_onnx.py --mode static \\")
                print(f"         --input checkpoints/LightUNet_200.onnx \\")
                print(f"         --output checkpoints/LightUNet_200_PTQ_static.onnx \\")
                print(f"         --calibration_data /path/to/calibration/images/")
                print("\n  3. HOẶC chạy trên device có int8 support (mobile, GPU với TensorRT)")
                print("="*70)
                raise RuntimeError("Quantized model không tương thích với device hiện tại. Vui lòng dùng Float32 model hoặc Static Quantization (QDQ).")
            else:
                # Re-raise other errors
                raise

        # Lấy thông tin input/output
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"✓ Model loaded successfully!")
        print(f"  - Input name: {self.input_name}")
        print(f"  - Output name: {self.output_name}")
        print(f"  - Providers: {self.session.get_providers()}")

        # Model info
        model_size = os.path.getsize(onnx_path) / (1024 ** 2)
        print(f"\nModel Information:")
        print(f"  - Model Size: {model_size:.2f} MB")

        # Try to calculate FLOPs by loading corresponding PyTorch model
        try:
            import LU2Net
            print(f"\nModel Complexity Analysis:")
            print(f"  (Loading PyTorch model to calculate FLOPs...)")

            # Try to find corresponding .pth file
            pth_path = onnx_path.replace('.onnx', '.pth')
            if not os.path.exists(pth_path):
                # Try default path
                pth_path = '/home/ndpthao/eject/IMPLEMENTATION/LU2Net-master/checkpoints/LightUNet_200.pth'

            if os.path.exists(pth_path):
                torch_model = LU2Net.LU2Net()
                torch_model.load_state_dict(torch.load(pth_path, map_location='cpu'))
                torch_model.eval()

                flops_info = metrics.calculate_flops(torch_model, input_shape=(1, 3, 256, 256), device='cpu')
                print(f"  FLOPs (GFLOPs)     : {flops_info['flops']:.3f}")
                print(f"  MACs (GMACs)       : {flops_info['macs']:.3f}")
                print(f"  Parameters (M)     : {flops_info['params']:.3f}")
            else:
                print(f"  Note: PyTorch model not found at {pth_path}")
                print(f"        FLOPs cannot be calculated for ONNX model")
        except Exception as e:
            print(f"  Note: Could not calculate FLOPs ({str(e)})")
            print(f"        This is normal for ONNX-only deployment")

        # Warm-up run để khởi tạo session
        if warmup:
            print(f"\nRunning warm-up inference...")
            dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
            _ = self.session.run([self.output_name], {self.input_name: dummy_input})
            print(f"✓ Warm-up completed!")
        print()

    def preprocess(self, image, target_size=(256, 256)):
        """
        Preprocess image cho ONNX model

        Args:
            image: OpenCV image (BGR)
            target_size: (width, height)

        Returns:
            Preprocessed tensor (1, 3, H, W), RGB image
        """
        # Resize
        resized = cv2.resize(image, target_size)

        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # HWC -> CHW
        chw = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        batch = np.expand_dims(chw, axis=0)

        return batch, rgb

    def postprocess(self, output_tensor):
        """
        Postprocess ONNX output tensor

        Args:
            output_tensor: ONNX output (1, 3, H, W)

        Returns:
            RGB image (H, W, 3) uint8
        """
        # Remove batch dimension
        output = output_tensor.squeeze(0)

        # CHW -> HWC
        hwc = np.transpose(output, (1, 2, 0))

        # Clip và convert về uint8
        output_img = np.clip(hwc * 255, 0, 255).astype(np.uint8)

        return output_img

    def _enhance_core(self, image):
        """Core enhancement function for memory profiling"""
        pipeline_start = time.time()

        # Preprocess
        preprocess_start = time.time()
        input_tensor, input_rgb = self.preprocess(image)
        preprocess_time = time.time() - preprocess_start

        # Inference
        inference_start = time.time()
        ort_inputs = {self.input_name: input_tensor}
        ort_outputs = self.session.run([self.output_name], ort_inputs)
        inference_time = time.time() - inference_start

        # Postprocess
        postprocess_start = time.time()
        enhanced_rgb = self.postprocess(ort_outputs[0])
        postprocess_time = time.time() - postprocess_start

        total_time = time.time() - pipeline_start

        # Convert output_tensor để tính metrics (giống PyTorch format)
        output_tensor = ort_outputs[0]  # Keep as numpy array for now

        return enhanced_rgb, input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time

    def enhance(self, image):
        """
        Enhance underwater image với timing chi tiết

        Args:
            image: OpenCV image (BGR)

        Returns:
            enhanced_image (RGB), input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time, ram_used_mb
        """
        mem_before = memory_usage()[0]

        # Run core processing and track memory
        mem_usage_list = memory_usage((self._enhance_core, (image,)), interval=0.001, timeout=None, max_usage=True)
        mem_peak = mem_usage_list if isinstance(mem_usage_list, (int, float)) else max(mem_usage_list)

        # Calculate memory used only for image processing
        ram_used_mb = mem_peak - mem_before

        # Get the actual results by running the function again
        enhanced_rgb, input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time = self._enhance_core(image)

        return enhanced_rgb, input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time, ram_used_mb


def process_single_image(model, input_path, output_path, gt_path=None):
    """Process single image với metrics đầy đủ"""
    print(f"\n{'='*70}")
    print(f"Processing: {input_path}")
    print(f"{'='*70}")

    # Load image
    image = cv2.imread(input_path)
    if image is None:
        print(f"✗ Cannot read image: {input_path}")
        return

    orig_h, orig_w = image.shape[:2]
    print(f"Original size: {orig_w}x{orig_h}")

    # Enhance
    output_rgb, input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time = model.enhance(image)

    # Load Ground Truth (if provided)
    gt_img = None
    if gt_path and os.path.exists(gt_path):
        gt_img = cv2.imread(gt_path)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        # Resize GT to match enhanced output
        gt_img = cv2.resize(gt_img, (output_rgb.shape[1], output_rgb.shape[0]))
        print(f"Ground Truth: {gt_path}")

    # Convert output_tensor to torch tensor for metrics
    torch_device = torch.device('cuda' if model.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    output_torch = torch.from_numpy(output_tensor).to(torch_device)

    # Compute metrics
    img_metrics = metrics.evaluate_all_image_metrics(
        input_img=gt_img,
        output_img=output_rgb,
        output_tensor=output_torch,
        device=torch_device
    )

    # Memory usage
    memory_mb = metrics.get_memory_usage(model=None)  # ONNX model không dùng torch

    # Energy consumption
    energy_joules, battery_wh = metrics.calculate_energy_consumption(total_time)

    # Model size
    model_size_mb = metrics.get_model_size(model.onnx_path)

    # Resize về kích thước gốc
    output_rgb_resized = cv2.resize(output_rgb, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Save output
    output_bgr = cv2.cvtColor(output_rgb_resized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output_bgr)
    print(f"✓ Saved: {output_path}")

    # Display metrics
    print(f"\n{'='*70}")
    print("IMAGE QUALITY METRICS")
    print(f"{'='*70}")

    # Full-reference metrics (if GT available)
    if gt_img is not None:
        print("Full-Reference Quality Metrics (vs Ground Truth):")
        print(f"  PSNR           : {img_metrics.get('psnr', -1):.2f} dB")
        print(f"  SSIM           : {img_metrics.get('ssim', -1):.4f}")
        print()

    # No-reference metrics (always available)
    print("No-Reference Quality Metrics:")
    print(f"  UCIQE          : {img_metrics.get('uciqe', -1):.4f}")
    print(f"  UIQM           : {img_metrics.get('uiqm', -1):.4f}")
    print(f"  NIQE           : {img_metrics.get('niqe', -1):.4f} (lower is better)")
    print(f"{'-'*70}")
    print("PERFORMANCE METRICS")
    print(f"  Inference Time : {inference_time*1000:.2f} ms  (ONNX model only)")
    print(f"  Total Time     : {total_time*1000:.2f} ms  (full pipeline)")
    print(f"  FPS            : {1.0/inference_time:.2f}")
    print(f"{'-'*70}")
    print("DEVICE METRICS")
    print(f"  Model Size     : {model_size_mb:.2f} MB")
    print(f"  Memory Usage   : {memory_mb:.2f} MB")
    print(f"  Energy         : {energy_joules:.2f} J ({battery_wh:.6f} Wh)")
    print(f"{'-'*70}")
    print("TIMING BREAKDOWN")
    print(f"  Preprocess     : {preprocess_time*1000:.2f} ms")
    print(f"  Inference      : {inference_time*1000:.2f} ms")
    print(f"  Postprocess    : {postprocess_time*1000:.2f} ms")
    print(f"  Total          : {total_time*1000:.2f} ms")
    print(f"{'='*70}")


def process_directory(model, input_dir, output_dir, gt_dir=None):
    """Process all images in directory với metrics đầy đủ"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"Processing directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if gt_dir:
        print(f"Ground Truth directory: {gt_dir}")
    print(f"{'='*70}")

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in os.listdir(input_dir)
                   if Path(f).suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images\n")

    # Storage for metrics
    all_inference_times = []
    all_total_times = []
    all_preprocess_times = []
    all_postprocess_times = []
    all_memory_usage = []
    all_ram_usage = []
    all_metrics = []

    # Process each image
    torch_device = torch.device('cuda' if model.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    for idx, img_file in enumerate(tqdm(image_files, desc="Processing images")):
        input_path = os.path.join(input_dir, img_file)
        # Giữ nguyên tên file gốc
        output_path = os.path.join(output_dir, img_file)

        # Load image
        image = cv2.imread(input_path)
        if image is None:
            print(f"⚠️ Skipping {img_file} (unreadable)")
            continue

        orig_h, orig_w = image.shape[:2]

        # Enhance
        output_rgb, input_rgb, inference_time, output_tensor, preprocess_time, postprocess_time, total_time, ram_used_mb = model.enhance(image)

        # Skip first frame timing (warm-up effect) - only for FPS calculation
        # But still save the image
        if idx > 0:  # Bỏ qua frame đầu tiên khi tính metrics timing
            all_inference_times.append(inference_time)
            all_total_times.append(total_time)
            all_preprocess_times.append(preprocess_time)
            all_postprocess_times.append(postprocess_time)
            all_ram_usage.append(ram_used_mb)

        # Memory usage for ONNX: use model file size as estimate
        # ONNX Runtime manages memory internally
        if idx == 0:  # Calculate only once
            onnx_memory = metrics.get_model_size(model.onnx_path)
            all_memory_usage.append(onnx_memory)
        else:
            all_memory_usage.append(onnx_memory)

        # Resize về kích thước gốc
        output_rgb_resized = cv2.resize(output_rgb, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        # Save
        output_bgr = cv2.cvtColor(output_rgb_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_bgr)

        # Load Ground Truth (if provided)
        gt_img = None
        if gt_dir:
            gt_path = os.path.join(gt_dir, img_file)
            if os.path.exists(gt_path):
                gt_img = cv2.imread(gt_path)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                # Resize GT to match enhanced output
                gt_img = cv2.resize(gt_img, (output_rgb.shape[1], output_rgb.shape[0]))

        # Convert output_tensor to torch tensor for metrics
        output_torch = torch.from_numpy(output_tensor).to(torch_device)

        # Compute metrics
        img_metrics = metrics.evaluate_all_image_metrics(
            input_img=gt_img,
            output_img=output_rgb,
            output_tensor=output_torch,
            device=torch_device
        )

        all_metrics.append(img_metrics)

    # Calculate FPS metrics
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

    # Performance metrics with standard deviation
    model_size_mb = metrics.get_model_size(model.onnx_path)
    avg_inference_time = np.mean(all_inference_times) if all_inference_times else 0.0
    std_inference_time = np.std(all_inference_times) if all_inference_times else 0.0
    avg_total_time = np.mean(all_total_times) if all_total_times else 0.0
    std_total_time = np.std(all_total_times) if all_total_times else 0.0
    avg_preprocess_time = np.mean(all_preprocess_times) if all_preprocess_times else 0.0
    std_preprocess_time = np.std(all_preprocess_times) if all_preprocess_times else 0.0
    avg_postprocess_time = np.mean(all_postprocess_times) if all_postprocess_times else 0.0
    std_postprocess_time = np.std(all_postprocess_times) if all_postprocess_times else 0.0

    # Memory: AVERAGE memory per image
    avg_memory = np.mean(all_memory_usage) if all_memory_usage else 0.0

    # RAM Usage: AVERAGE RAM per image (from memory_profiler)
    avg_ram_usage = np.mean(all_ram_usage) if all_ram_usage else 0.0
    ram_usage_std = np.std(all_ram_usage) if all_ram_usage else 0.0

    # Energy: AVERAGE energy per image
    avg_energy, avg_battery = metrics.calculate_energy_consumption(avg_total_time)

    avg_metrics['inference_time'] = avg_inference_time
    avg_metrics['inference_time_std'] = std_inference_time
    avg_metrics['total_time'] = avg_total_time
    avg_metrics['total_time_std'] = std_total_time
    avg_metrics['preprocess_time'] = avg_preprocess_time
    avg_metrics['preprocess_time_std'] = std_preprocess_time
    avg_metrics['postprocess_time'] = avg_postprocess_time
    avg_metrics['postprocess_time_std'] = std_postprocess_time
    avg_metrics['model_size_mb'] = model_size_mb
    avg_metrics['memory_mb'] = avg_memory
    avg_metrics['ram_usage_mb'] = avg_ram_usage
    avg_metrics['ram_usage_std'] = ram_usage_std
    avg_metrics['energy_joules'] = avg_energy
    avg_metrics['battery_wh'] = avg_battery

    # Try to get FLOPs info from PyTorch model
    flops_info = None
    try:
        import LU2Net
        pth_path = model.onnx_path.replace('.onnx', '.pth')
        if not os.path.exists(pth_path):
            pth_path = '/home/ndpthao/eject/IMPLEMENTATION/LU2Net-master/checkpoints/LightUNet_200.pth'

        if os.path.exists(pth_path):
            torch_model = LU2Net.LU2Net()
            torch_model.load_state_dict(torch.load(pth_path, map_location='cpu'))
            torch_model.eval()
            flops_info = metrics.calculate_flops(torch_model, input_shape=(1, 3, 256, 256), device='cpu')
    except:
        pass

    # Display results
    print("\n" + "=" * 70)
    print("BATCH ENHANCEMENT RESULTS (ONNX)")
    print("=" * 70)
    print(f"  Total Images   : {len(image_files)}")
    print(f"  Processed      : {len(image_files)}")
    print(f"  Used for Timing: {len(all_inference_times)} (excluded 1st frame warm-up)")
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
    print("PERFORMANCE (Inference only - 1st frame excluded)")
    print("  Total Frames   : {}".format(fps_metrics['total_frames']))
    print("  Total Time     : {:.2f} s".format(fps_metrics['total_time']))
    print("  Avg FPS        : {:.2f}".format(fps_metrics['avg_fps']))
    print("  Min FPS        : {:.2f}".format(fps_metrics['min_fps']))
    print("  Max FPS        : {:.2f}".format(fps_metrics['max_fps']))
    print("=" * 70)
    print("PERFORMANCE METRICS (Excluding 1st frame)")
    print(f"  Avg Inference Time    : {avg_metrics['inference_time']:.4f} s  (ONNX model only)")
    print(f"  Avg Total Time        : {avg_metrics['total_time']:.4f} s  (full pipeline)")
    print(f"  Avg FPS (1/inference) : {fps_metrics['avg_fps']:.2f}")
    print(f"  Note: First frame excluded from timing (warm-up effect)")
    print("-" * 70)
    if flops_info:
        print("MODEL COMPLEXITY")
        print(f"  FLOPs                 : {flops_info['flops']:.3f} GFLOPs")
        print(f"  MACs                  : {flops_info['macs']:.3f} GMACs")
        print(f"  Parameters            : {flops_info['params']:.3f} M")
        print("-" * 70)
    print("DEVICE PERFORMANCE METRICS (Per Image Average)")
    print(f"  Model Size            : {avg_metrics['model_size_mb']:.2f} MB  (ONNX model file)")
    print(f"  Memory Usage          : {avg_metrics['memory_mb']:.2f} MB  (avg: process memory)")
    print(f"  RAM Usage (Profiler)  : {avg_metrics['ram_usage_mb']:.2f} ± {avg_metrics['ram_usage_std']:.2f} MB  (image processing only)")
    print(f"  Energy Consumption    : {avg_metrics['energy_joules']:.2f} J ({avg_metrics['battery_wh']:.6f} Wh)")
    print(f"                          (avg per image: preprocess + inference + postprocess)")
    print("-" * 70)
    print("TIMING BREAKDOWN")
    print(f"  Preprocess  : {avg_metrics['preprocess_time']:.4f} ± {avg_metrics['preprocess_time_std']:.4f} s")
    print(f"  Inference   : {avg_metrics['inference_time']:.4f} ± {avg_metrics['inference_time_std']:.4f} s")
    print(f"  Postprocess : {avg_metrics['postprocess_time']:.4f} ± {avg_metrics['postprocess_time_std']:.4f} s")
    print(f"  Total       : {avg_metrics['total_time']:.4f} ± {avg_metrics['total_time_std']:.4f} s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="LU2Net ONNX Inference with Full Metrics")
    parser.add_argument('--input', type=str, required=True,
                        help='Input image file or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output image file or directory')
    parser.add_argument('--gt', type=str, default=None,
                        help='Ground truth image or directory (optional)')
    parser.add_argument('--model', type=str, default='/home/ndpthao/eject/IMPLEMENTATION/LU2Net-master/checkpoints/LightUNet_200.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use (cpu or cuda)')

    args = parser.parse_args()

    # Load ONNX model
    model = LU2NetONNX(onnx_path=args.model, device=args.device)

    # Process
    if os.path.isfile(args.input):
        # Single image
        process_single_image(model, args.input, args.output, gt_path=args.gt)
    elif os.path.isdir(args.input):
        # Directory
        process_directory(model, args.input, args.output, gt_dir=args.gt)
    else:
        print(f"✗ Invalid input: {args.input}")


if __name__ == "__main__":
    main()
