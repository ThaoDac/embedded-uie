"""
WaterFormer ONNX Inference Script
Chạy inference với WaterFormer ONNX model và tính toán các metrics
Dựa trên mẫu của LU2Net-master/inference_onnx.py

Usage:
    # Single image
    python inference_onnx.py --input path/to/image.jpg --output output/enhanced.jpg

    # Batch processing
    python inference_onnx.py --input input_dir/ --output output_dir/ --gt gt_dir/
"""

import os
import sys
import cv2
import numpy as np
import onnxruntime as ort
import torch
import argparse
from pathlib import Path
import time
from memory_profiler import memory_usage

# Add parent directory to path for metrics import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'waterformer'))
sys.path.insert(0, os.path.dirname(__file__))

import metrics


class WaterFormerONNX:
    """WaterFormer ONNX Model Inference Class"""

    def __init__(self, onnx_path, imgsz=256, warmup=True):
        """
        Initialize WaterFormer ONNX model

        Args:
            onnx_path: Path to ONNX model file
            imgsz: Input image size (will be resized to imgsz x imgsz)
            warmup: Run warm-up inference
        """
        self.imgsz = imgsz
        print("=" * 70)
        print("WATERFORMER ONNX MODEL INITIALIZATION")
        print("=" * 70)

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Model not found: {onnx_path}")

        # Create ONNX Runtime session with error handling for quantized models
        print(f"\nLoading ONNX model: {onnx_path}")

        # Configure session options for better performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Try to load model with different providers
        providers = ['CPUExecutionProvider']

        try:
            self.session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
            self.onnx_path = onnx_path
            print(f"✓ Model loaded successfully!")
        except Exception as e:
            error_msg = str(e)

            # Check if it's a quantized model issue
            if 'ConvInteger' in error_msg or 'NOT_IMPLEMENTED' in error_msg:
                print(f"\n{'='*70}")
                print("⚠️  QUANTIZED MODEL NOT SUPPORTED ON THIS DEVICE")
                print("="*70)
                print(f"\nLỗi: {error_msg[:200]}...")
                print("\nModel này là DYNAMIC QUANTIZED (Int8) model sử dụng ConvInteger")
                print("operators không được hỗ trợ trên CPUExecutionProvider.")
                print("\nGiải pháp:")
                print("  1. Sử dụng Float32 model thay thế:")

                # Try to find Float32 model
                float32_path = onnx_path.replace('_dynamic.onnx', '.onnx').replace('_qdq.onnx', '.onnx').replace('_qoperator.onnx', '.onnx').replace('_static.onnx', '.onnx')

                if os.path.exists(float32_path) and float32_path != onnx_path:
                    print(f"     {float32_path}")
                    print("\n  → Tự động chuyển sang Float32 model...")

                    try:
                        self.session = ort.InferenceSession(float32_path, sess_options=sess_options, providers=providers)
                        self.onnx_path = float32_path
                        print(f"  ✓ Loaded Float32 model successfully: {float32_path}")
                    except Exception as e2:
                        print(f"  ✗ Không thể load Float32 model: {e2}")
                        raise RuntimeError("Không thể load được model nào. Vui lòng kiểm tra lại model files.")
                else:
                    print(f"     --model {float32_path}")
                    print("\n  2. HOẶC sử dụng Static Quantization (QDQ format):")
                    print(f"     python quantize_onnx.py --mode static --format qdq \\")
                    print(f"         --input {float32_path} \\")
                    print(f"         --output {onnx_path.replace('_dynamic', '_qdq')} \\")
                    print(f"         --calibration_data /path/to/calibration/images/")
                    print("\n  3. HOẶC chạy trên device có int8 support (GPU với TensorRT, mobile)")
                    print("="*70)
                    raise RuntimeError("Quantized model không tương thích. Vui lòng dùng Float32 model hoặc Static Quantization (QDQ).")
            else:
                # Re-raise other errors
                raise

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"\nONNX Runtime Information:")
        print(f"  - Input name: {self.input_name}")
        print(f"  - Output name: {self.output_name}")
        print(f"  - Providers: {self.session.get_providers()}")

        # Model info (use self.onnx_path which may be different from input onnx_path)
        model_size = os.path.getsize(self.onnx_path) / (1024 ** 2)
        print(f"\nModel Information:")
        print(f"  - Model Size: {model_size:.2f} MB")
        print(f"  - Input Size: {self.imgsz}x{self.imgsz}")

        # Detect quantization type
        if '_dynamic' in self.onnx_path:
            print(f"  - Type: Dynamic Quantized (Int8)")
        elif '_qdq' in self.onnx_path or '_static' in self.onnx_path:
            print(f"  - Type: Static Quantized (Int8 - QDQ format)")
        elif '_qoperator' in self.onnx_path:
            print(f"  - Type: Static Quantized (Int8 - QOperator format)")
        else:
            print(f"  - Type: Float32")

        # Try to calculate FLOPs by loading corresponding PyTorch model
        try:
            from waterformer.models.archs.waterformer_arch import WaterFormer
            print(f"\nModel Complexity Analysis:")
            print(f"  (Loading PyTorch model to calculate FLOPs...)")

            # Try to find corresponding .pth file (use self.onnx_path which might be Float32)
            pth_path = self.onnx_path.replace('.onnx', '.pth')
            if not os.path.exists(pth_path):
                # Try work_dirs path
                pth_path = 'work_dirs/UW_WaterFormer/models/net_g_best.pth'
            if not os.path.exists(pth_path):
                pth_path = 'checkpoints/weights.pth'

            if os.path.exists(pth_path):
                # Create model with default WaterFormer architecture
                torch_model = WaterFormer(
                    inp_channels=3,
                    out_channels=3,
                    dim=36,
                    num_blocks=[2, 2, 2, 2],
                    heads=[2, 2, 2, 2],
                    bias=False,
                    window_size=8,
                    shift_size=3
                )

                # Load checkpoint
                checkpoint = torch.load(pth_path, map_location='cpu')
                if 'params' in checkpoint:
                    state_dict = checkpoint['params']
                elif 'params_ema' in checkpoint:
                    state_dict = checkpoint['params_ema']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                torch_model.load_state_dict(state_dict, strict=False)
                torch_model.eval()

                flops_info = metrics.calculate_flops(torch_model, input_shape=(1, 3, self.imgsz, self.imgsz), device='cpu')
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
            dummy_input = np.random.randn(1, 3, self.imgsz, self.imgsz).astype(np.float32)
            _ = self.session.run([self.output_name], {self.input_name: dummy_input})
            print(f"✓ Warm-up completed!")
        print()

    def preprocess(self, image, target_size=None):
        """
        Preprocess input image

        Args:
            image: Input BGR image (numpy array)
            target_size: Target size (height, width). If None, uses self.imgsz

        Returns:
            preprocessed: Preprocessed tensor (NCHW format)
            rgb: RGB image for visualization
        """
        # Use self.imgsz if target_size not specified
        if target_size is None:
            target_size = (self.imgsz, self.imgsz)

        # Resize
        resized = cv2.resize(image, target_size)

        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # HWC -> CHW and add batch dimension
        chw = np.transpose(normalized, (2, 0, 1))
        batch = np.expand_dims(chw, axis=0)

        return batch, rgb

    def postprocess(self, output_tensor):
        """
        Postprocess output tensor to image

        Args:
            output_tensor: Output from model (NCHW format)

        Returns:
            output_img: Output image (numpy array, uint8)
        """
        # Remove batch dimension and CHW -> HWC
        output = output_tensor.squeeze(0)
        hwc = np.transpose(output, (1, 2, 0))

        # Clip to [0, 1] and convert to uint8
        output_img = np.clip(hwc * 255, 0, 255).astype(np.uint8)

        return output_img

    def _enhance_core(self, image_path, output_path=None):
        """Core enhancement function for memory profiling"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        original_size = (image.shape[1], image.shape[0])  # (width, height)

        # Preprocess
        start_preprocess = time.time()
        input_tensor, rgb_input = self.preprocess(image)
        preprocess_time = time.time() - start_preprocess

        # Inference
        start_inference = time.time()
        ort_inputs = {self.input_name: input_tensor}
        ort_outputs = self.session.run([self.output_name], ort_inputs)
        inference_time = time.time() - start_inference

        # Postprocess
        start_postprocess = time.time()
        output_img = self.postprocess(ort_outputs[0])

        # Resize back to original size
        output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        output_resized = cv2.resize(output_img_bgr, original_size)
        postprocess_time = time.time() - start_postprocess

        # Save output if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            cv2.imwrite(output_path, output_resized)

        # Calculate metrics
        result = {
            'preprocess_time': preprocess_time,
            'inference_time': inference_time,
            'postprocess_time': postprocess_time,
            'total_time': preprocess_time + inference_time + postprocess_time,
            'fps': 1.0 / inference_time if inference_time > 0 else 0
        }

        return result

    def enhance(self, image_path, output_path=None):
        """
        Enhance a single image

        Args:
            image_path: Path to input image
            output_path: Path to save enhanced image (optional)

        Returns:
            result_dict: Dictionary containing metrics and timing info
        """
        mem_before = memory_usage()[0]

        # Run core processing and track memory
        mem_usage_list = memory_usage((self._enhance_core, (image_path, output_path)), interval=0.001, timeout=None, max_usage=True)
        mem_peak = mem_usage_list if isinstance(mem_usage_list, (int, float)) else max(mem_usage_list)

        # Calculate memory used only for image processing
        ram_used_mb = mem_peak - mem_before

        # Get the actual results by running the function again
        result = self._enhance_core(image_path, output_path)
        result['ram_used_mb'] = ram_used_mb

        return result


def process_single_image(model, input_path, output_path, gt_path=None):
    """Process a single image and display results"""
    print(f"\n{'='*70}")
    print(f"Processing: {os.path.basename(input_path)}")
    print(f"{'='*70}")

    # Enhance image
    result = model.enhance(input_path, output_path)

    # Display results
    print(f"\nTiming Information:")
    print(f"  Preprocess  : {result['preprocess_time']:.4f} s")
    print(f"  Inference   : {result['inference_time']:.4f} s")
    print(f"  Postprocess : {result['postprocess_time']:.4f} s")
    print(f"  Total       : {result['total_time']:.4f} s")
    print(f"  FPS         : {result['fps']:.2f}")

    # Calculate quality metrics
    enhanced_img = cv2.imread(output_path)

    if enhanced_img is not None:
        # Calculate no-reference metrics (always available)
        print(f"\nNo-Reference Quality Metrics:")

        try:
            # UCIQE - Underwater Color Image Quality Evaluation
            uciqe = metrics.calculate_uciqe(enhanced_img)
            print(f"  UCIQE       : {uciqe:.4f}")
        except Exception as e:
            print(f"  UCIQE       : Error ({str(e)[:50]})")

        try:
            # UIQM - Underwater Image Quality Measure
            uiqm = metrics.calculate_uiqm(enhanced_img)
            print(f"  UIQM        : {uiqm:.4f}")
        except Exception as e:
            print(f"  UIQM        : Error ({str(e)[:50]})")

        try:
            # NIQE - Natural Image Quality Evaluator
            # Convert to tensor for NIQE
            enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
            enhanced_tensor = torch.from_numpy(enhanced_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            niqe = metrics.calculate_niqe(enhanced_tensor)
            print(f"  NIQE        : {niqe:.4f} (lower is better)")
        except Exception as e:
            print(f"  NIQE        : Error ({str(e)[:50]})")

    # Calculate full-reference metrics if GT provided
    if gt_path and os.path.exists(gt_path):
        print(f"\nFull-Reference Quality Metrics (vs Ground Truth):")

        gt_img = cv2.imread(gt_path)

        if enhanced_img is not None and gt_img is not None:
            # Ensure same size
            if enhanced_img.shape != gt_img.shape:
                gt_img = cv2.resize(gt_img, (enhanced_img.shape[1], enhanced_img.shape[0]))

            # Calculate metrics
            psnr = metrics.calculate_psnr(enhanced_img, gt_img)
            ssim = metrics.calculate_ssim(enhanced_img, gt_img)

            print(f"  PSNR        : {psnr:.2f} dB")
            print(f"  SSIM        : {ssim:.4f}")

    print(f"\n✓ Enhanced image saved: {output_path}")
    print(f"{'='*70}")


def process_directory(model, input_dir, output_dir, gt_dir=None):
    """Process all images in a directory"""
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING")
    print(f"{'='*70}")
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    if gt_dir:
        print(f"GT directory    : {gt_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))

    image_files = sorted(image_files)

    if len(image_files) == 0:
        print(f"✗ No images found in {input_dir}")
        return

    print(f"\nFound {len(image_files)} images")
    print(f"{'='*70}\n")

    # Process images
    all_results = []
    all_psnr = []
    all_ssim = []
    all_uciqe = []
    all_uiqm = []
    all_niqe = []
    all_inference_times = []  # Exclude first frame for timing stats
    all_total_time = []
    all_ram_usage = []

    # Get model complexity info once
    model_size_mb = os.path.getsize(model.onnx_path) / (1024 * 1024)

    for idx, img_path in enumerate(image_files):
        img_name = img_path.stem
        output_path = os.path.join(output_dir, f"enhanced_{img_name}.jpg")

        print(f"[{idx+1}/{len(image_files)}] Processing: {img_path.name}")

        # Enhance
        result = model.enhance(str(img_path), output_path)


        # Skip first frame for timing statistics (warm-up effect)
        if idx > 0:
            all_inference_times.append(result['inference_time'])
            all_ram_usage.append(result['ram_used_mb'])

        # Calculate no-reference metrics (always)
        enhanced_img = cv2.imread(output_path)
        if enhanced_img is not None:
            try:
                uciqe = metrics.calculate_uciqe(enhanced_img)
                result['uciqe'] = uciqe
                all_uciqe.append(uciqe)
            except:
                pass

            try:
                uiqm = metrics.calculate_uiqm(enhanced_img)
                result['uiqm'] = uiqm
                all_uiqm.append(uiqm)
            except:
                pass

            try:
                enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
                enhanced_tensor = torch.from_numpy(enhanced_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                niqe = metrics.calculate_niqe(enhanced_tensor)
                result['niqe'] = niqe
                all_niqe.append(niqe)
            except:
                pass

        # Calculate full-reference metrics if GT available
        if gt_dir:
            gt_path = os.path.join(gt_dir, img_path.name)
            if not os.path.exists(gt_path):
                # Try with different extensions
                for ext in image_extensions:
                    gt_path_alt = os.path.join(gt_dir, f"{img_name}{ext}")
                    if os.path.exists(gt_path_alt):
                        gt_path = gt_path_alt
                        break

            if os.path.exists(gt_path):
                enhanced_img = cv2.imread(output_path)
                gt_img = cv2.imread(gt_path)

                if enhanced_img is not None and gt_img is not None:
                    if enhanced_img.shape != gt_img.shape:
                        gt_img = cv2.resize(gt_img, (enhanced_img.shape[1], enhanced_img.shape[0]))

                    psnr = metrics.calculate_psnr(enhanced_img, gt_img)
                    ssim = metrics.calculate_ssim(enhanced_img, gt_img)

                    result['psnr'] = psnr
                    result['ssim'] = ssim

                    all_psnr.append(psnr)
                    all_ssim.append(ssim)

                    print(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, Time: {result['inference_time']:.4f}s")
                else:
                    print(f"  Time: {result['inference_time']:.4f}s (GT read failed)")
            else:
                print(f"  Time: {result['inference_time']:.4f}s (GT not found)")
        else:
            print(f"  Time: {result['inference_time']:.4f}s")

        # Get device metrics
        result['memory_mb'] = metrics.get_memory_usage()
        result['energy_joules'], result['battery_wh'] = metrics.calculate_energy_consumption(result['total_time'])

        all_results.append(result)

    # Calculate average metrics (excluding first frame for timing)
    avg_metrics = {}

    if len(all_inference_times) > 0:
        avg_inference_time = np.mean(all_inference_times)
        std_inference_time = np.std(all_inference_times)
        min_inference_time = np.min(all_inference_times)
        max_inference_time = np.max(all_inference_times)
    else:
        avg_inference_time = all_results[0]['inference_time']
        std_inference_time = 0
        min_inference_time = avg_inference_time
        max_inference_time = avg_inference_time

    avg_total_time = np.mean([r['total_time'] for r in all_results[1:]] if len(all_results) > 1 else [all_results[0]['total_time']])
    avg_preprocess_time = np.mean([r['preprocess_time'] for r in all_results])
    avg_postprocess_time = np.mean([r['postprocess_time'] for r in all_results])

    avg_memory = np.mean([r['memory_mb'] for r in all_results])
    avg_energy = np.mean([r['energy_joules'] for r in all_results])
    avg_battery = np.mean([r['battery_wh'] for r in all_results])

    # FPS metrics
    fps_metrics = {
        'avg_fps': 1.0 / avg_inference_time if avg_inference_time > 0 else 0,
        'min_fps': 1.0 / max_inference_time if max_inference_time > 0 else 0,
        'max_fps': 1.0 / min_inference_time if min_inference_time > 0 else 0
    }

    if all_psnr:
        avg_metrics['psnr'] = np.mean(all_psnr)
        avg_metrics['psnr_std'] = np.std(all_psnr)

    if all_ssim:
        avg_metrics['ssim'] = np.mean(all_ssim)
        avg_metrics['ssim_std'] = np.std(all_ssim)

    # No-reference metrics
    if all_uciqe:
        avg_metrics['uciqe'] = np.mean(all_uciqe)
        avg_metrics['uciqe_std'] = np.std(all_uciqe)

    if all_uiqm:
        avg_metrics['uiqm'] = np.mean(all_uiqm)
        avg_metrics['uiqm_std'] = np.std(all_uiqm)

    if all_niqe:
        avg_metrics['niqe'] = np.mean(all_niqe)
        avg_metrics['niqe_std'] = np.std(all_niqe)

    # RAM Usage: AVERAGE RAM per image (from memory_profiler)
    avg_ram_usage = np.mean(all_ram_usage) if all_ram_usage else 0.0
    ram_usage_std = np.std(all_ram_usage) if all_ram_usage else 0.0

    avg_metrics['inference_time'] = avg_inference_time
    avg_metrics['inference_time_std'] = std_inference_time
    avg_metrics['total_time'] = avg_total_time
    avg_metrics['preprocess_time'] = avg_preprocess_time
    avg_metrics['postprocess_time'] = avg_postprocess_time
    avg_metrics['model_size_mb'] = model_size_mb
    avg_metrics['memory_mb'] = avg_memory
    avg_metrics['ram_usage_mb'] = avg_ram_usage
    avg_metrics['ram_usage_std'] = ram_usage_std
    avg_metrics['energy_joules'] = avg_energy
    avg_metrics['battery_wh'] = avg_battery

    # Try to get FLOPs info from PyTorch model
    flops_info = None
    try:
        from models.archs.waterformer_arch import WaterFormer
        pth_path = model.onnx_path.replace('.onnx', '.pth')
        if not os.path.exists(pth_path):
            pth_path = 'checkpoints/weights.pth'

        if os.path.exists(pth_path):
            torch_model = WaterFormer(
                inp_channels=3, out_channels=3, dim=36,
                num_blocks=[2, 2, 2, 2], heads=[2, 2, 2, 2],
                bias=False, window_size=8, shift_size=3
            )
            checkpoint = torch.load(pth_path, map_location='cpu')
            if 'params' in checkpoint:
                state_dict = checkpoint['params']
            elif 'params_ema' in checkpoint:
                state_dict = checkpoint['params_ema']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            torch_model.load_state_dict(state_dict, strict=False)
            torch_model.eval()
            flops_info = metrics.calculate_flops(torch_model, input_shape=(1, 3, model.imgsz, model.imgsz), device='cpu')
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

    if all_psnr:
        print("FULL-REFERENCE QUALITY METRICS (Average vs Ground Truth)")
        print(f"  PSNR           : {avg_metrics['psnr']:.9f} ± {avg_metrics['psnr_std']:.9f} dB")
        print(f"  SSIM           : {avg_metrics['ssim']:.9f} ± {avg_metrics['ssim_std']:.9f}")
        print("-" * 70)

    if all_uciqe or all_uiqm or all_niqe:
        print("NO-REFERENCE QUALITY METRICS (Average)")
        if all_uciqe:
            print(f"  UCIQE          : {avg_metrics['uciqe']:.9f} ± {avg_metrics['uciqe_std']:.9f}")
        if all_uiqm:
            print(f"  UIQM           : {avg_metrics['uiqm']:.9f} ± {avg_metrics['uiqm_std']:.9f}")
        if all_niqe:
            print(f"  NIQE           : {avg_metrics['niqe']:.9f} ± {avg_metrics['niqe_std']:.9f} (lower is better)")
        print("-" * 70)

    print("FPS STATISTICS (Excluding 1st frame)")
    print(f"  Avg FPS        : {fps_metrics['avg_fps']:.2f}")
    print(f"  Min FPS        : {fps_metrics['min_fps']:.2f}")
    print(f"  Max FPS        : {fps_metrics['max_fps']:.2f}")
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
    print(f"  Preprocess  : {avg_metrics['preprocess_time']:.4f} s")
    print(f"  Inference   : {avg_metrics['inference_time']:.4f} ± {avg_metrics['inference_time_std']:.4f} s")
    print(f"  Postprocess : {avg_metrics['postprocess_time']:.4f} s")
    print("=" * 70)

    print(f"\n✓ All enhanced images saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="WaterFormer ONNX Inference")
    parser.add_argument('--model', type=str, default='/home/ndpthao/eject/IMPLEMENTATION/WaterFormer-master/work_dirs/UW_WaterFormer/models/net_g_best.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output image or directory')
    parser.add_argument('--gt', type=str, default=None,
                        help='Ground truth directory (optional, for metrics)')
    parser.add_argument('--imgsz', type=int, default=256,
                        help='Input image size for model (default: 256)')

    args = parser.parse_args()

    # Load model
    model = WaterFormerONNX(args.model, imgsz=args.imgsz)

    # Process
    if os.path.isfile(args.input):
        # Single image
        process_single_image(model, args.input, args.output, args.gt)
    elif os.path.isdir(args.input):
        # Directory
        process_directory(model, args.input, args.output, gt_dir=args.gt)
    else:
        print(f"✗ Invalid input: {args.input}")


if __name__ == "__main__":
    main()
