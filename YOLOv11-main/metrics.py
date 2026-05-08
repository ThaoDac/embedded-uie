"""
Underwater Image Quality Metrics Module
========================================
This module provides comprehensive metrics for evaluating underwater image quality,
including both image quality assessment and real-time performance metrics.

Image Quality Metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- NIQE (Natural Image Quality Evaluator)
- UIQM (Underwater Image Quality Measure)
- UCIQE (Underwater Color Image Quality Evaluation)

Performance Metrics:
- FPS (Frames Per Second)
- Inference Latency
- Model Size
- Memory Usage
- Energy Consumption
"""

import os
import time
import psutil
import numpy as np
import cv2
import math
import torch
from scipy import ndimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import gaussian_filter
from skimage.util import view_as_windows


# ============================================================================
# Image Quality Metrics
# ============================================================================

def calculate_psnr(img1, img2, data_range=255):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1: Reference image (numpy array)
        img2: Test image (numpy array)
        data_range: The data range of the input image (default: 255)

    Returns:
        float: PSNR value in dB
    """
    return peak_signal_noise_ratio(img1, img2, data_range=data_range)


def calculate_ssim(img1, img2, multichannel=True, channel_axis=-1, data_range=255):
    """
    Calculate Structural Similarity Index (SSIM) between two images.

    Args:
        img1: Reference image (numpy array)
        img2: Test image (numpy array)
        multichannel: Whether to treat the image as multichannel (default: True)
        channel_axis: Channel axis for color images (default: -1)
        data_range: The data range of the input image (default: 255)

    Returns:
        float: SSIM value between -1 and 1
    """
    return structural_similarity(
        img1, img2,
        multichannel=multichannel,
        channel_axis=channel_axis,
        data_range=data_range
    )


def calculate_niqe(img_tensor, device=None):
    """
    Calculate Natural Image Quality Evaluator (NIQE).
    Lower NIQE scores indicate better perceptual quality.

    Args:
        img_tensor: Input image - can be:
                   - PyTorch tensor (N, 3, H, W) or (3, H, W), RGB, range [-1,1] or [0,1]
                   - Numpy array (H, W, 3) or (H, W), RGB/BGR, 0-255 range
        device: torch device (if None, will auto-detect)

    Returns:
        float: NIQE score (lower is better)
    """
    # Try using PyIQA first (more accurate)
    try:
        import pyiqa

        # Force CPU to avoid cuDNN issues with pyiqa
        if device is None:
            device = torch.device("cpu")

        # Convert input to tensor if needed
        if not torch.is_tensor(img_tensor):
            # Convert numpy array to tensor
            img_np = img_tensor.astype(np.float32)
            if img_np.max() > 1.0:
                img_np = img_np / 255.0

            # Convert (H, W, C) to (C, H, W)
            if img_np.ndim == 3:
                img_np = np.transpose(img_np, (2, 0, 1))

            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)
        else:
            # Normalize tensor range to [0, 1] if needed
            img_min = img_tensor.min().item()
            img_max = img_tensor.max().item()

            # If tensor is in range [-1, 1], convert to [0, 1]
            if img_min < 0:
                img_tensor = (img_tensor + 1.0) / 2.0

            # Clamp to [0, 1] range to ensure valid input for pyiqa
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

            # Ensure tensor is on correct device
            if img_tensor.device != device:
                img_tensor = img_tensor.to(device)

            # Ensure 4D tensor (N, C, H, W)
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)

        # Create NIQE metric
        iqa_metric = pyiqa.create_metric('niqe', device=device)

        # Calculate NIQE score
        niqe_score = iqa_metric(img_tensor)

        return float(niqe_score.item())

    except ImportError:
        print("Warning: pyiqa not installed. Using simplified NIQE estimation.")
        print("For better accuracy, install pyiqa: pip install pyiqa")
        # Fallback to simplified implementation
        return _calculate_niqe_simplified(img_tensor, device)

    except Exception as e:
        print(f"Error calculating NIQE with pyiqa: {str(e)}")
        print("Falling back to simplified NIQE estimation.")
        # Fallback to simplified implementation
        return _calculate_niqe_simplified(img_tensor, device)


def _calculate_niqe_simplified(image, device=None):
    """
    Simplified NIQE estimation based on patch statistics.
    This is a fallback method when PyIQA is not available.

    Args:
        image: Input image (tensor or numpy array)
        device: torch device (not used in this implementation)

    Returns:
        float: Simplified NIQE score
    """
    try:
        # Convert PyTorch tensor to numpy array if needed
        if torch.is_tensor(image):
            # Tensor format: (N, C, H, W) or (C, H, W)
            img_np = image.squeeze().detach().cpu().numpy()

            # If normalized to [-1, 1], convert to [0, 1]
            if img_np.min() < 0:
                img_np = (img_np + 1.0) / 2.0

            # Convert from (C, H, W) to (H, W, C)
            if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
                img_np = np.transpose(img_np, (1, 2, 0))

            # Scale to 0-255 range
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            image = img_np

        # Convert to grayscale if color image
        if image.ndim == 3:
            # Handle RGB or BGR
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                image = image[:, :, 0]

        # Simplified NIQE calculation
        image = image.astype(np.float32)

        # Normalize
        image = (image - np.mean(image)) / (np.std(image) + 1e-9)

        # Calculate local mean and variance
        mu = gaussian_filter(image, 7 / 6, truncate=3)
        sigma = np.sqrt(np.abs(gaussian_filter(image ** 2, 7 / 6, truncate=3) - mu ** 2))

        # Structural distortion
        structdis = (image - mu) / (sigma + 1)

        # Extract patches
        min_dim = min(structdis.shape)
        if min_dim < 8:
            # If image too small, return high NIQE score (poor quality indicator)
            return 100.0

        patches = view_as_windows(structdis, (8, 8)).reshape(-1, 64)

        # Calculate NIQE approximation
        niqe_score = float(
            np.mean(np.sqrt(np.var(patches, axis=1))) +
            np.mean(np.abs(np.mean(patches, axis=1)))
        )

        return niqe_score

    except Exception as e:
        print(f"Error in simplified NIQE calculation: {str(e)}")
        return -1.0

# ============================================================================
# UIQM (Underwater Image Quality Measure) Implementation
# Reference: https://github.com/mkartik/Shallow-UWnet/blob/main/uiqm_utils.py
# ============================================================================

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """Calculate the asymmetric alpha-trimmed mean"""
    x = sorted(x)
    K = len(x)
    T_a_L = math.ceil(alpha_L * K)
    T_a_R = math.floor(alpha_R * K)
    weight = (1 / (K - T_a_L - T_a_R))
    s = int(T_a_L + 1)
    e = int(K - T_a_R)
    val = sum(x[s:e])
    val = weight * val
    return val


def s_a(x, mu):
    """Calculate variance for alpha-trimmed mean"""
    val = 0
    for pixel in x:
        val += math.pow((pixel - mu), 2)
    return val / len(x)


def _uicm(x):
    """Underwater Image Colorfulness Measure"""
    R = x[:, :, 0].flatten()
    G = x[:, :, 1].flatten()
    B = x[:, :, 2].flatten()
    RG = R - G
    YB = ((R + G) / 2) - B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt((math.pow(mu_a_RG, 2) + math.pow(mu_a_YB, 2)))
    r = math.sqrt(s_a_RG + s_a_YB)
    return (-0.0268 * l) + (0.1586 * r)


def sobel(x):
    """Apply Sobel edge detection"""
    dx = ndimage.sobel(x, 0)
    dy = ndimage.sobel(x, 1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    return mag


def eme(x, window_size):
    """Enhancement Measure Estimation"""
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size
    w = 2. / (k1 * k2)
    blocksize_x = window_size
    blocksize_y = window_size
    x = x[:blocksize_y * k2, :blocksize_x * k1]
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1)]
            max_ = np.max(block)
            min_ = np.min(block)
            if min_ == 0.0:
                val += 0
            elif max_ == 0.0:
                val += 0
            else:
                val += math.log(max_ / min_)
    return w * val


def _uism(x):
    """Underwater Image Sharpness Measure"""
    R = x[:, :, 0]
    G = x[:, :, 1]
    B = x[:, :, 2]
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    r_eme = eme(R_edge_map, 8)
    g_eme = eme(G_edge_map, 8)
    b_eme = eme(B_edge_map, 8)
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r * r_eme) + (lambda_g * g_eme) + (lambda_b * b_eme)


def _uiconm(x, window_size):
    """Underwater Image Contrast Measure"""
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size
    w = -1. / (k1 * k2)
    blocksize_x = window_size
    blocksize_y = window_size
    x = x[:blocksize_y * k2, :blocksize_x * k1]
    alpha = 1
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_ - min_
            bot = max_ + min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0:
                val += 0.0
            else:
                val += alpha * math.pow((top / bot), alpha) * math.log(top / bot)
    return w * val


def calculate_uiqm(img):
    """
    Calculate Underwater Image Quality Measure (UIQM).

    Args:
        img: Input image as numpy array (RGB format, uint8)

    Returns:
        float: UIQM score (higher is better)
    """
    x = img.astype(np.float32)
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753
    uicm = _uicm(x)
    uism = _uism(x)
    uiconm = _uiconm(x, 8)
    uiqm = (c1 * uicm) + (c2 * uism) + (c3 * uiconm)
    return uiqm


# ============================================================================
# UCIQE (Underwater Color Image Quality Evaluation) Implementation
# Reference: https://github.com/TongJiayan/UCIQE-python
# ============================================================================

def calculate_uciqe(img):
    """
    Calculate Underwater Color Image Quality Evaluation (UCIQE).

    Args:
        img: Input image as numpy array (BGR format, uint8) or image path

    Returns:
        float: UCIQE score (higher is better)
    """
    # Handle both numpy array and file path inputs
    if isinstance(img, str):
        img_BGR = cv2.imread(img)
    else:
        # If input is RGB numpy array, convert to BGR for cv2
        if img.shape[2] == 3:
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_BGR = img

    # Convert to LAB color space
    img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
    img_LAB = np.array(img_LAB, dtype=np.float64)

    # Coefficients for UCIQE metric
    coe_Metric = [0.4680, 0.2745, 0.2576]

    # Extract L, A, B channels and normalize
    img_lum = img_LAB[:, :, 0] / 255.0
    img_a = img_LAB[:, :, 1] / 255.0
    img_b = img_LAB[:, :, 2] / 255.0

    # 1. Chroma calculation
    chroma = np.sqrt(np.square(img_a) + np.square(img_b))
    sigma_c = np.std(chroma)

    # 2. Luminance contrast
    img_lum_flat = img_lum.flatten()
    sorted_index = np.argsort(img_lum_flat)
    top_index = sorted_index[int(len(img_lum_flat) * 0.99)]
    bottom_index = sorted_index[int(len(img_lum_flat) * 0.01)]
    con_lum = img_lum_flat[top_index] - img_lum_flat[bottom_index]

    # 3. Saturation calculation
    chroma_flat = chroma.flatten()
    sat = np.divide(
        chroma_flat,
        img_lum_flat,
        out=np.zeros_like(chroma_flat, dtype=np.float64),
        where=img_lum_flat != 0
    )
    avg_sat = np.mean(sat)

    # Calculate UCIQE
    uciqe = sigma_c * coe_Metric[0] + con_lum * coe_Metric[1] + avg_sat * coe_Metric[2]

    return uciqe


# ============================================================================
# Real-Time Performance Metrics
# ============================================================================

def calculate_fps(latency):
    """
    Calculate Frames Per Second (FPS) from latency.

    Args:
        latency: Inference latency in seconds

    Returns:
        float: FPS value
    """
    if latency > 0:
        return 1.0 / latency
    return 0.0


def get_model_size(model_path):
    """
    Calculate model size in MB.

    Args:
        model_path: Path to model file or directory containing model files

    Returns:
        float: Model size in MB
    """
    try:
        if os.path.isfile(model_path):
            size = os.path.getsize(model_path) / (1024 ** 2)
        elif os.path.isdir(model_path):
            # Support PyTorch (.pth, .pt) and Keras/TensorFlow (.h5, .json) models
            size = sum(
                os.path.getsize(os.path.join(model_path, f))
                for f in os.listdir(model_path)
                if f.endswith(('.pth', '.pt', '.h5', '.json'))
            ) / (1024 ** 2)
        else:
            size = 0.0
        return size
    except Exception as e:
        print(f"Error calculating model size: {str(e)}")
        return 0.0


def get_memory_usage(model=None, device='cpu'):
    """
    Get model memory usage in MB (GPU or CPU).

    This measures ONLY the model weights and buffers memory,
    NOT the entire process memory.

    Args:
        model: PyTorch/Keras model instance (optional)
        device: 'cpu' or 'cuda'

    Returns:
        float: Model memory usage in MB
    """
    try:
        # If model is provided, calculate its actual memory usage
        if model is not None:
            total_bytes = 0

            # PyTorch model
            if hasattr(model, 'parameters'):
                # Count parameters memory
                for param in model.parameters():
                    total_bytes += param.numel() * param.element_size()

                # Count buffers memory (BatchNorm, etc.)
                for buffer in model.buffers():
                    total_bytes += buffer.numel() * buffer.element_size()

                return total_bytes / (1024 ** 2)  # Convert to MB

            # Keras/TensorFlow model
            elif hasattr(model, 'count_params'):
                total_params = model.count_params()
                total_bytes = total_params * 4  # Assume float32 (4 bytes)
                return total_bytes / (1024 ** 2)

        # If no model provided (e.g., ONNX), return minimal memory estimate
        # ONNX Runtime manages memory internally, we can't directly measure it
        # Return a small constant to indicate model-only memory (not process memory)
        return 0.0

    except Exception as e:
        print(f"Error getting memory usage: {str(e)}")
        return 0.0


def get_model_parameters_count(model):
    """
    Get total number of parameters in the model.

    Args:
        model: PyTorch model instance

    Returns:
        dict: Dictionary with 'total', 'trainable', 'non_trainable' parameters
    """
    try:
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_trainable_params = total_params - trainable_params

            return {
                'total': total_params,
                'trainable': trainable_params,
                'non_trainable': non_trainable_params,
                'total_mb': (total_params * 4) / (1024 ** 2)  # Assume float32
            }
        return {'total': 0, 'trainable': 0, 'non_trainable': 0, 'total_mb': 0.0}
    except Exception as e:
        print(f"Error counting parameters: {str(e)}")
        return {'total': 0, 'trainable': 0, 'non_trainable': 0, 'total_mb': 0.0}


def calculate_flops(model, input_shape=(1, 3, 256, 256), device='cpu'):
    """
    Calculate FLOPs (Floating Point Operations) using thop library.

    Args:
        model: PyTorch model instance
        input_shape: Input tensor shape (batch, channels, height, width)
        device: 'cpu' or 'cuda'

    Returns:
        dict: Dictionary containing FLOPs, MACs, and parameters
              {
                  'flops': FLOPs in GFLOPs,
                  'macs': MACs (Multiply-Accumulate Operations) in GMACs,
                  'params': Number of parameters in millions,
                  'flops_str': Human-readable FLOPs string,
                  'macs_str': Human-readable MACs string
              }
    """
    try:
        # Try using thop (PyTorch-OpCounter) for accurate FLOPs calculation
        try:
            from thop import profile, clever_format

            # Create dummy input
            dummy_input = torch.randn(input_shape).to(device)

            # Calculate FLOPs and parameters
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)

            # MACs = FLOPs / 2 (approximately, for most operations)
            macs = flops / 2

            # Convert to human-readable format
            flops_str, params_str = clever_format([flops, params], "%.3f")
            macs_str = f"{macs/1e9:.3f}G"

            return {
                'flops': flops / 1e9,  # GFLOPs
                'macs': macs / 1e9,    # GMACs
                'params': params / 1e6,  # Million parameters
                'flops_str': flops_str,
                'macs_str': macs_str,
                'params_str': params_str
            }

        except ImportError:
            print("Warning: 'thop' library not found. Install with: pip install thop")
            print("Falling back to estimation method...")

            # Fallback: Simple estimation
            # Rough estimation: FLOPs ≈ 2 * params * input_pixels
            total_params = sum(p.numel() for p in model.parameters())
            input_pixels = np.prod(input_shape[1:])  # C * H * W
            estimated_flops = 2 * total_params * input_pixels / 1e9  # GFLOPs

            return {
                'flops': estimated_flops,
                'macs': estimated_flops / 2,
                'params': total_params / 1e6,
                'flops_str': f"{estimated_flops:.3f}G",
                'macs_str': f"{estimated_flops/2:.3f}G",
                'params_str': f"{total_params/1e6:.3f}M"
            }

    except Exception as e:
        print(f"Error calculating FLOPs: {str(e)}")
        return {
            'flops': 0.0,
            'macs': 0.0,
            'params': 0.0,
            'flops_str': "N/A",
            'macs_str': "N/A",
            'params_str': "N/A"
        }


def calculate_flops_simple(image_shape, model_complexity_factor=30):
    """
    Simple FLOPs estimation based on image size (legacy method).

    Args:
        image_shape: Shape of the input image (H, W, C)
        model_complexity_factor: Multiplier for model complexity (default: 30)

    Returns:
        float: Estimated FLOPs in MFLOPs
    """
    flops = np.prod(image_shape) * model_complexity_factor / 1e6
    return flops


def calculate_energy_consumption(latency, power_watts=150):
    """
    Calculate energy consumption in Joules and Watt-hours.

    Args:
        latency: Inference time in seconds
        power_watts: Estimated power consumption in Watts (default: 150W for GPU)

    Returns:
        tuple: (energy_joules, battery_watt_hours)
    """
    energy_joules = latency * power_watts
    battery_wh = energy_joules / 3600  # Convert Joules to Watt-hours
    return energy_joules, battery_wh


def calculate_performance_metrics(start_time, end_time, image_shape, model_path='PyTorch/models', model=None,
                                 preprocess_time=0, inference_time=0, postprocess_time=0):
    """
    Calculate comprehensive real-time performance metrics.

    Args:
        start_time: Start timestamp for entire pipeline (from time.time())
        end_time: End timestamp for entire pipeline (from time.time())
        image_shape: Shape of the processed image
        model_path: Path to model file or directory
        model: Model instance for memory calculation (optional)
        preprocess_time: Time spent on preprocessing (optional)
        inference_time: Time spent on model inference only (optional)
        postprocess_time: Time spent on postprocessing (optional)

    Returns:
        dict: Dictionary containing all performance metrics
    """
    try:
        # Calculate total pipeline latency (load input -> model -> output)
        total_latency = end_time - start_time

        # If individual times not provided, use total latency
        if inference_time == 0:
            inference_time = total_latency

        # FPS based on inference time only (standard metric)
        fps = calculate_fps(inference_time)

        # Memory and model metrics (model-only memory)
        memory_mb = get_memory_usage(model=model)
        model_size_mb = get_model_size(model_path)

        # FLOPs estimation
        flops_mflops = calculate_flops(image_shape)

        # Energy consumption for ENTIRE pipeline (preprocess + inference + postprocess)
        # This represents total energy from loading input to producing output
        energy_j, battery_wh = calculate_energy_consumption(total_latency)

        return {
            'latency': inference_time,  # Inference time only for FPS calculation
            'total_latency': total_latency,  # Total pipeline time
            'fps': fps,
            'memory_mb': memory_mb,
            'model_size_mb': model_size_mb,
            'flops_mflops': flops_mflops,
            'energy_joules': energy_j,
            'battery_wh': battery_wh,
            'preprocess_time': preprocess_time,
            'postprocess_time': postprocess_time
        }

    except Exception as e:
        print(f"Error calculating performance metrics: {str(e)}")
        return {
            'latency': 0.0,
            'total_latency': 0.0,
            'fps': 0.0,
            'memory_mb': 0.0,
            'model_size_mb': 0.0,
            'flops_mflops': 0.0,
            'energy_joules': 0.0,
            'battery_wh': 0.0,
            'preprocess_time': 0.0,
            'postprocess_time': 0.0
        }


def calculate_video_fps_metrics(frame_times):
    """
    Calculate FPS metrics from video processing frame times.

    Args:
        frame_times: List of frame processing times (in seconds)

    Returns:
        dict: Dictionary containing FPS metrics
            - total_frames: Total number of frames processed
            - total_time: Total processing time in seconds
            - avg_fps: Average frames per second
            - min_fps: Minimum FPS (worst case)
            - max_fps: Maximum FPS (best case)
            - avg_frame_time: Average time per frame in seconds
            - min_frame_time: Minimum time per frame in seconds
            - max_frame_time: Maximum time per frame in seconds
            - std_frame_time: Standard deviation of frame times
            - median_frame_time: Median frame processing time
    """
    if not frame_times or len(frame_times) == 0:
        return {
            'total_frames': 0,
            'total_time': 0.0,
            'avg_fps': 0.0,
            'min_fps': 0.0,
            'max_fps': 0.0,
            'avg_frame_time': 0.0,
            'min_frame_time': 0.0,
            'max_frame_time': 0.0,
            'std_frame_time': 0.0,
            'median_frame_time': 0.0
        }

    frame_times_array = np.array(frame_times)

    total_frames = len(frame_times)
    total_time = np.sum(frame_times_array)
    avg_frame_time = np.mean(frame_times_array)
    min_frame_time = np.min(frame_times_array)
    max_frame_time = np.max(frame_times_array)
    std_frame_time = np.std(frame_times_array)
    median_frame_time = np.median(frame_times_array)

    # Calculate FPS metrics
    avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    min_fps = 1.0 / max_frame_time if max_frame_time > 0 else 0.0  # Min FPS = slowest frame
    max_fps = 1.0 / min_frame_time if min_frame_time > 0 else 0.0  # Max FPS = fastest frame

    return {
        'total_frames': total_frames,
        'total_time': total_time,
        'avg_fps': avg_fps,
        'min_fps': min_fps,
        'max_fps': max_fps,
        'avg_frame_time': avg_frame_time,
        'min_frame_time': min_frame_time,
        'max_frame_time': max_frame_time,
        'std_frame_time': std_frame_time,
        'median_frame_time': median_frame_time
    }


# ============================================================================
# Convenience Functions
# ============================================================================

def evaluate_all_image_metrics(input_img, output_img, output_tensor=None, device=None):
    """
    Evaluate all image quality metrics at once.

    Args:
        input_img: Input/reference image as numpy array (RGB, uint8)
        output_img: Output/enhanced image as numpy array (RGB, uint8)
        output_tensor: Optional output image as PyTorch tensor for NIQE calculation
        device: Optional torch device for NIQE

    Returns:
        dict: Dictionary containing all image quality metrics
    """
    metrics = {}

    # PSNR and SSIM (reference-based metrics)
    try:
        metrics['psnr'] = calculate_psnr(input_img, output_img)
    except Exception as e:
        print(f"Error calculating PSNR: {str(e)}")
        metrics['psnr'] = -1.0

    try:
        metrics['ssim'] = calculate_ssim(input_img, output_img)
    except Exception as e:
        print(f"Error calculating SSIM: {str(e)}")
        metrics['ssim'] = -1.0

    # No-reference metrics (only need output image)
    try:
        metrics['uiqm'] = calculate_uiqm(output_img)
    except Exception as e:
        print(f"Error calculating UIQM: {str(e)}")
        metrics['uiqm'] = -1.0

    try:
        metrics['uciqe'] = calculate_uciqe(output_img)
    except Exception as e:
        print(f"Error calculating UCIQE: {str(e)}")
        metrics['uciqe'] = -1.0

    # NIQE (can use tensor or numpy array)
    try:
        # Use tensor if provided, otherwise use numpy array
        niqe_input = output_tensor if output_tensor is not None else output_img
        metrics['niqe'] = calculate_niqe(niqe_input, device)
    except Exception as e:
        print(f"Error calculating NIQE: {str(e)}")
        metrics['niqe'] = -1.0

    return metrics


# ============================================================================
# Object Detection Metrics (UOD - Underwater Object Detection)
# ============================================================================

def calculate_iou(box1, box2, format='xyxy'):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box [x1, y1, x2, y2] or [x, y, w, h]
        box2: Second bounding box [x1, y1, x2, y2] or [x, y, w, h]
        format: 'xyxy' for (x1, y1, x2, y2) or 'xywh' for (x, y, w, h)

    Returns:
        float: IoU value between 0 and 1
    """
    # Convert to xyxy format if needed
    if format == 'xywh':
        box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
        box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0

    return iou


def calculate_batch_iou(pred_boxes, gt_boxes, format='xyxy'):
    """
    Calculate IoU for batches of bounding boxes.

    Args:
        pred_boxes: List of predicted boxes [[x1, y1, x2, y2], ...]
        gt_boxes: List of ground truth boxes [[x1, y1, x2, y2], ...]
        format: 'xyxy' or 'xywh'

    Returns:
        np.ndarray: IoU matrix of shape (len(pred_boxes), len(gt_boxes))
    """
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))

    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_iou(pred_box, gt_box, format)

    return iou_matrix


def calculate_precision_recall_f1(tp, fp, fn):
    """
    Calculate Precision, Recall, and F1-Score.

    Args:
        tp: True Positives (int)
        fp: False Positives (int)
        fn: False Negatives (int)

    Returns:
        dict: {'precision': float, 'recall': float, 'f1': float}
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_ap(precisions, recalls):
    """
    Calculate Average Precision (AP) using 11-point interpolation.

    Args:
        precisions: List of precision values
        recalls: List of recall values

    Returns:
        float: Average Precision
    """
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = np.array(recalls)[sorted_indices]
    precisions = np.array(precisions)[sorted_indices]

    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0

    return ap


def calculate_map(ap_per_class):
    """
    Calculate mean Average Precision (mAP) across all classes.

    Args:
        ap_per_class: List or dict of AP values per class

    Returns:
        float: mAP value
    """
    if isinstance(ap_per_class, dict):
        ap_values = list(ap_per_class.values())
    else:
        ap_values = ap_per_class

    if len(ap_values) == 0:
        return 0.0

    return np.mean(ap_values)


def calculate_detection_metrics(predictions, ground_truths, iou_threshold=0.5, num_classes=None):
    """
    Calculate comprehensive object detection metrics.

    Args:
        predictions: List of predictions per image
                    [{'boxes': [[x1,y1,x2,y2], ...], 'scores': [...], 'labels': [...]}, ...]
        ground_truths: List of ground truths per image
                      [{'boxes': [[x1,y1,x2,y2], ...], 'labels': [...]}, ...]
        iou_threshold: IoU threshold for considering a detection as correct (default: 0.5)
        num_classes: Number of classes (if None, will be inferred)

    Returns:
        dict: Comprehensive detection metrics including:
            - precision: Overall precision
            - recall: Overall recall
            - f1: Overall F1-score
            - ap_per_class: AP for each class
            - mAP: mean Average Precision
            - tp: Total true positives
            - fp: Total false positives
            - fn: Total false negatives
    """
    # Infer number of classes if not provided
    if num_classes is None:
        all_labels = []
        for gt in ground_truths:
            all_labels.extend(gt['labels'])
        for pred in predictions:
            all_labels.extend(pred['labels'])
        num_classes = max(all_labels) + 1 if all_labels else 1

    # Initialize counters per class
    tp_per_class = [0] * num_classes
    fp_per_class = [0] * num_classes
    fn_per_class = [0] * num_classes
    precisions_per_class = {i: [] for i in range(num_classes)}
    recalls_per_class = {i: [] for i in range(num_classes)}

    # Process each image
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = np.array(pred['boxes']) if len(pred['boxes']) > 0 else np.array([])
        pred_scores = np.array(pred['scores']) if len(pred['scores']) > 0 else np.array([])
        pred_labels = np.array(pred['labels']) if len(pred['labels']) > 0 else np.array([])

        gt_boxes = np.array(gt['boxes']) if len(gt['boxes']) > 0 else np.array([])
        gt_labels = np.array(gt['labels']) if len(gt['labels']) > 0 else np.array([])

        # Sort predictions by confidence score (descending)
        if len(pred_scores) > 0:
            sorted_indices = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[sorted_indices]
            pred_labels = pred_labels[sorted_indices]
            pred_scores = pred_scores[sorted_indices]

        # Match predictions to ground truths
        matched_gt = set()

        for pred_idx in range(len(pred_boxes)):
            pred_box = pred_boxes[pred_idx]
            pred_label = pred_labels[pred_idx]

            best_iou = 0.0
            best_gt_idx = -1

            # Find best matching ground truth
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue

                gt_box = gt_boxes[gt_idx]
                gt_label = gt_labels[gt_idx]

                # Only match same class
                if pred_label != gt_label:
                    continue

                iou = calculate_iou(pred_box, gt_box, format='xyxy')

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Check if match is valid
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # True positive
                tp_per_class[pred_label] += 1
                matched_gt.add(best_gt_idx)
            else:
                # False positive
                fp_per_class[pred_label] += 1

        # Count false negatives (unmatched ground truths)
        for gt_idx in range(len(gt_boxes)):
            if gt_idx not in matched_gt:
                gt_label = gt_labels[gt_idx]
                fn_per_class[gt_label] += 1

    # Calculate metrics per class
    ap_per_class = {}
    for cls in range(num_classes):
        tp = tp_per_class[cls]
        fp = fp_per_class[cls]
        fn = fn_per_class[cls]

        if tp + fp + fn == 0:
            # No predictions or ground truths for this class
            ap_per_class[cls] = 0.0
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions_per_class[cls].append(precision)
        recalls_per_class[cls].append(recall)

        # Calculate AP (simplified)
        ap_per_class[cls] = precision  # Simplified AP = precision at this threshold

    # Calculate overall metrics
    total_tp = sum(tp_per_class)
    total_fp = sum(fp_per_class)
    total_fn = sum(fn_per_class)

    overall_metrics = calculate_precision_recall_f1(total_tp, total_fp, total_fn)

    # Calculate mAP
    mAP = calculate_map(ap_per_class)

    return {
        'precision': overall_metrics['precision'],
        'recall': overall_metrics['recall'],
        'f1': overall_metrics['f1'],
        'ap_per_class': ap_per_class,
        'mAP': mAP,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'tp_per_class': tp_per_class,
        'fp_per_class': fp_per_class,
        'fn_per_class': fn_per_class
    }


def extract_yolo_metrics(results):
    """
    Extract and format metrics from YOLO validation results.

    Args:
        results: YOLO validation results object from model.val()

    Returns:
        dict: Formatted metrics dictionary containing:
            - Detection metrics: mAP@0.5, mAP@0.5:0.95, precision, recall, F1
            - Performance metrics: FPS, inference time, etc.
            - Per-class metrics: AP per class
    """
    metrics = {}

    # Extract box metrics
    box = getattr(results, 'box', None)
    if box:
        # mAP metrics
        metrics['mAP50-95'] = float(getattr(box, 'map', 0.0))      # mAP@0.5:0.95
        metrics['mAP50'] = float(getattr(box, 'map50', 0.0))        # mAP@0.5
        metrics['mAP75'] = float(getattr(box, 'map75', 0.0))        # mAP@0.75

        # Precision and Recall
        mp = getattr(box, 'mp', None)  # mean precision
        mr = getattr(box, 'mr', None)  # mean recall

        if mp is not None:
            metrics['precision'] = float(mp)
        if mr is not None:
            metrics['recall'] = float(mr)

        # F1 Score
        if mp is not None and mr is not None and (mp + mr) > 0:
            metrics['f1'] = float(2 * mp * mr / (mp + mr))
        else:
            metrics['f1'] = 0.0

        # Per-class mAP
        if hasattr(box, 'maps'):
            maps = getattr(box, 'maps', None)
            if maps is not None:
                if hasattr(maps, 'tolist'):
                    metrics['per_class_mAP'] = maps.tolist()
                else:
                    metrics['per_class_mAP'] = list(maps)

        # Per-class precision and recall
        if hasattr(box, 'p'):
            p = getattr(box, 'p', None)
            if p is not None and hasattr(p, 'tolist'):
                metrics['per_class_precision'] = p.tolist()

        if hasattr(box, 'r'):
            r = getattr(box, 'r', None)
            if r is not None and hasattr(r, 'tolist'):
                metrics['per_class_recall'] = r.tolist()

    # Extract speed metrics
    if hasattr(results, 'speed'):
        speed = results.speed
        metrics['speed'] = {}
        for k, v in speed.items():
            metrics['speed'][k] = float(v)

        # Calculate FPS from inference time
        if 'inference' in speed and speed['inference'] > 0:
            metrics['fps'] = 1000.0 / speed['inference']  # Convert ms to FPS

    # Extract confusion matrix info
    if hasattr(results, 'confusion_matrix'):
        cm = results.confusion_matrix
        if cm is not None and hasattr(cm, 'matrix'):
            metrics['confusion_matrix'] = cm.matrix.tolist()

    return metrics


def print_detection_metrics(metrics, class_names=None):
    """
    Pretty print detection metrics.

    Args:
        metrics: Dictionary of metrics from extract_yolo_metrics()
        class_names: Optional list of class names for per-class metrics
    """
    print("\n" + "="*70)
    print("OBJECT DETECTION METRICS (UOD)")
    print("="*70)

    # Overall metrics
    print("\nOverall Performance:")
    print(f"  mAP@0.5:0.95 : {metrics.get('mAP50-95', 0.0):.4f}")
    print(f"  mAP@0.5      : {metrics.get('mAP50', 0.0):.4f}")
    print(f"  mAP@0.75     : {metrics.get('mAP75', 0.0):.4f}")
    print(f"  Precision    : {metrics.get('precision', 0.0):.4f}")
    print(f"  Recall       : {metrics.get('recall', 0.0):.4f}")
    print(f"  F1-Score     : {metrics.get('f1', 0.0):.4f}")

    # Per-class metrics
    if 'per_class_mAP' in metrics:
        print("\nPer-Class mAP:")
        per_class_map = metrics['per_class_mAP']
        for i, ap in enumerate(per_class_map):
            class_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
            print(f"  {class_name:20s}: {ap:.4f}")

    # Speed metrics
    if 'speed' in metrics:
        print("\nInference Performance:")
        speed = metrics['speed']
        for k, v in speed.items():
            print(f"  {k.capitalize():15s}: {v:.2f} ms")
        if 'fps' in metrics:
            print(f"  {'FPS':15s}: {metrics['fps']:.2f}")

    # TP/FP/FN if available
    if 'tp' in metrics:
        print("\nDetection Statistics:")
        print(f"  True Positives  : {metrics.get('tp', 0)}")
        print(f"  False Positives : {metrics.get('fp', 0)}")
        print(f"  False Negatives : {metrics.get('fn', 0)}")

    print("="*70 + "\n")


# ============================================================================
# Main function for testing
# ============================================================================

if __name__ == '__main__':
    print("Underwater Image Quality & Object Detection Metrics Module")
    print("=" * 60)
    print("\nImage Quality Metrics:")
    print("- calculate_psnr(img1, img2)")
    print("- calculate_ssim(img1, img2)")
    print("- calculate_niqe(img_tensor)")
    print("- calculate_uiqm(img)")
    print("- calculate_uciqe(img)")
    print("- calculate_performance_metrics(start_time, end_time, image_shape)")
    print("- evaluate_all_image_metrics(input_img, output_img, output_tensor)")
    print("\nObject Detection Metrics (UOD):")
    print("- calculate_iou(box1, box2)")
    print("- calculate_precision_recall_f1(tp, fp, fn)")
    print("- calculate_ap(precisions, recalls)")
    print("- calculate_map(ap_per_class)")
    print("- calculate_detection_metrics(predictions, ground_truths)")
    print("- extract_yolo_metrics(results)")
    print("- print_detection_metrics(metrics)")
    print("\nFor usage examples, please refer to test.py")