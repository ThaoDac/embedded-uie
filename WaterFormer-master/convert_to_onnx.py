"""
Chuyển đổi WaterFormer model từ PyTorch (.pth) sang ONNX format
Dựa trên mẫu của LU2Net-master/convert_to_onnx.py

Usage:
    python convert_to_onnx.py --input checkpoints/weights.pth \
                               --output checkpoints/waterformer.onnx
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
import numpy as np

# Add waterformer directory to path to import model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'waterformer'))

from models.archs.waterformer_arch import WaterFormer


# ONNX-compatible AdaptiveAvgPool2d replacement
class AdaptiveAvgPool2dONNX(nn.Module):
    """
    Replace AdaptiveAvgPool2d with ONNX-compatible implementation using avg_pool2d.

    For AdaptiveAvgPool2d(output_size) where H % output_size == 0:
        kernel_size = H // output_size
        result = F.avg_pool2d(x, kernel_size=kernel_size, stride=kernel_size)

    This is mathematically IDENTICAL to AdaptiveAvgPool2d when dimensions divide evenly.
    """
    def __init__(self, output_size, input_size=256):
        super(AdaptiveAvgPool2dONNX, self).__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
            self.output_size = (output_size[0], output_size[0])
        else:
            self.output_size = tuple(output_size)

        # Pre-calculate kernel_size for non-(1,1) cases
        # Assuming input_size (default 256) divides evenly by output_size
        self.input_size = input_size
        if self.output_size != (1, 1):
            self.kernel_size = input_size // self.output_size[0]
        else:
            self.kernel_size = None

    def forward(self, x):
        # For (1,1) output, use mean which is faster and more ONNX-friendly
        if self.output_size == (1, 1):
            return x.mean(dim=(2, 3), keepdim=True)

        # AdaptiveAvgPool2d behavior:
        # - If H >= output_size: downsample using avg_pool2d
        # - If H < output_size: upsample using interpolate(area)
        H, W = x.shape[2], x.shape[3]

        if H >= self.output_size[0]:
            # Downsample case: use avg_pool2d (mathematically identical to AdaptiveAvgPool2d)
            kernel_h = H // self.output_size[0]
            kernel_w = W // self.output_size[1]
            return F.avg_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
        else:
            # Upsample case: use interpolate with mode='area' (equivalent to AdaptiveAvgPool2d)
            # This happens at latent level where H=32 but output_size=64
            return F.interpolate(x, size=self.output_size, mode='area')


def replace_adaptive_avgpool(model):
    """
    Recursively replace all AdaptiveAvgPool2d layers with ONNX-compatible version.

    For AdaptiveAvgPool2d(output_size) where H % output_size == 0:
        - Use F.avg_pool2d with kernel_size = H // output_size
        - This is mathematically IDENTICAL to AdaptiveAvgPool2d

    For (1,1) output: use mean() which is equivalent and faster.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            output_size = module.output_size
            # Normalize output_size to tuple
            if isinstance(output_size, int):
                output_size = (output_size, output_size)
            elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
                output_size = (output_size[0], output_size[0])

            # Replace with ONNX-compatible version using avg_pool2d
            print(f"  Replacing {name}: AdaptiveAvgPool2d({output_size}) -> AdaptiveAvgPool2dONNX (avg_pool2d)")
            setattr(model, name, AdaptiveAvgPool2dONNX(output_size))

        # Check if this is a GLTransBlock with adap_pool - replace it properly
        if module.__class__.__name__ == 'GLTransBlock' and hasattr(module, 'adap_pool'):
            if module.adap_pool is not None and isinstance(module.adap_pool, nn.AdaptiveAvgPool2d):
                output_size = module.adap_pool.output_size
                if isinstance(output_size, int):
                    output_size = (output_size, output_size)
                print(f"  Replacing adap_pool in GLTransBlock {name}: AdaptiveAvgPool2d({output_size}) -> AdaptiveAvgPool2dONNX")
                module.adap_pool = AdaptiveAvgPool2dONNX(output_size)

        # IMPORTANT: Always recurse into child modules, regardless of current module type
        replace_adaptive_avgpool(module)


def convert_pth_to_onnx(
    pth_model_path='/home/ndpthao/eject/IMPLEMENTATION/WaterFormer-master/work_dirs/UW_WaterFormer/models/net_g_best.pth',
    onnx_model_path='/home/ndpthao/eject/IMPLEMENTATION/WaterFormer-master/work_dirs/UW_WaterFormer/models/net_g_best.onnx',
    input_size=(1, 3, 256, 256),
    opset_version=17,
    verify=True,
    # WaterFormer architecture parameters
    inp_channels=3,
    out_channels=3,
    dim=36,
    num_blocks=[2, 2, 2, 2],
    heads=[2, 2, 2, 2],
    bias=False,
    window_size=8,
    shift_size=3
):
    """
    Chuyển đổi WaterFormer model PyTorch (.pth) sang ONNX format

    Args:
        pth_model_path: Đường dẫn đến file .pth
        onnx_model_path: Đường dẫn output file .onnx
        input_size: Kích thước input (batch, channels, height, width)
        opset_version: ONNX opset version (khuyến nghị: 13 hoặc cao hơn)
        verify: Kiểm tra model ONNX sau khi convert
        inp_channels, out_channels, dim, etc.: WaterFormer architecture params
    """

    print("=" * 70)
    print("CHUYỂN ĐỔI WATERFORMER MODEL SANG ONNX")
    print("=" * 70)

    # 1. Kiểm tra file .pth có tồn tại không
    if not os.path.exists(pth_model_path):
        raise FileNotFoundError(f"Không tìm thấy file model: {pth_model_path}")

    print(f"\n✓ File model PyTorch: {pth_model_path}")
    print(f"✓ Output ONNX: {onnx_model_path}")
    print(f"✓ Input size: {input_size}")
    print(f"✓ Opset version: {opset_version}")

    # 2. Load model PyTorch
    print("\n[1/5] Đang load WaterFormer model...")
    device = torch.device('cpu')  # Sử dụng CPU cho việc export

    # Create WaterFormer model with specified architecture
    model = WaterFormer(
        inp_channels=inp_channels,
        out_channels=out_channels,
        dim=dim,
        num_blocks=num_blocks,
        heads=heads,
        bias=bias,
        window_size=window_size,
        shift_size=shift_size
    )

    # Load pretrained weights
    print(f"  Loading checkpoint from: {pth_model_path}")
    checkpoint = torch.load(pth_model_path, map_location=device)

    # Handle different checkpoint formats
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
        print("  ✓ Loaded from 'params' key")
    elif 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
        print("  ✓ Loaded from 'params_ema' key (EMA weights)")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("  ✓ Loaded from 'state_dict' key")
    else:
        # Assume checkpoint is the state_dict itself
        state_dict = checkpoint
        print("  ✓ Loaded direct state_dict")

    try:
        model.load_state_dict(state_dict, strict=True)
        print("✓ WaterFormer model loaded thành công!")
    except RuntimeError as e:
        print(f"✗ Lỗi khi load state_dict: {str(e)}")
        print("  Thử load với strict=False...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {missing[:5]}...")  # Show first 5
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}...")
        print("✓ Model loaded với strict=False")

    model.to(device)
    model.eval()

    # 2.5. Replace AdaptiveAvgPool2d with ONNX-compatible version
    print("\n[2.5/5] Replacing AdaptiveAvgPool2d for ONNX compatibility...")
    replace_adaptive_avgpool(model)
    print("✓ Model adapted for ONNX export")

    # 3. Tạo dummy input
    print("\n[3/5] Tạo dummy input tensor...")
    dummy_input = torch.randn(input_size, device=device)
    print(f"✓ Dummy input shape: {dummy_input.shape}")

    # 4. Test forward pass trước khi export
    print("\n[3/5] Test forward pass...")
    with torch.no_grad():
        pytorch_output = model(dummy_input)
    print(f"✓ PyTorch output shape: {pytorch_output.shape}")

    # 5. Export sang ONNX
    print("\n[4/5] Đang export sang ONNX...")
    print("   (Quá trình này có thể mất vài phút do model phức tạp...)")

    # Create output directory if not exists
    os.makedirs(os.path.dirname(onnx_model_path) or '.', exist_ok=True)

    torch.onnx.export(
        model,                          # Model PyTorch
        dummy_input,                    # Dummy input
        onnx_model_path,                # Output file path
        export_params=True,             # Lưu parameters đã train
        opset_version=opset_version,    # ONNX opset version
        do_constant_folding=True,       # Tối ưu hóa constant folding
        input_names=['input'],          # Tên input
        output_names=['output'],        # Tên output
        dynamic_axes={
            'input': {0: 'batch_size'},    # Dynamic batch size
            'output': {0: 'batch_size'}
        }
    )
    print(f"✓ Export ONNX thành công: {onnx_model_path}")

    # 6. Kiểm tra file ONNX
    if verify:
        print("\n[5/5] Kiểm tra model ONNX...")
        try:
            # Load và validate ONNX model
            onnx_model = onnx.load(onnx_model_path)
            onnx.checker.check_model(onnx_model)
            print("✓ Model ONNX hợp lệ!")

            # Hiển thị thông tin model
            print("\nThông tin ONNX model:")
            print(f"   - IR version: {onnx_model.ir_version}")
            print(f"   - Opset version: {onnx_model.opset_import[0].version}")
            print(f"   - Producer: {onnx_model.producer_name}")

            # Test inference với ONNX Runtime
            print("\n✓ Kiểm tra inference với ONNX Runtime...")
            ort_session = ort.InferenceSession(onnx_model_path)

            # Test với dummy input
            dummy_input_np = dummy_input.cpu().numpy()
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_np}
            ort_outputs = ort_session.run(None, ort_inputs)

            print(f"   - Input shape: {dummy_input_np.shape}")
            print(f"   - Output shape: {ort_outputs[0].shape}")

            # So sánh output PyTorch vs ONNX
            pytorch_output_np = pytorch_output.cpu().numpy()
            max_diff = np.abs(pytorch_output_np - ort_outputs[0]).max()
            mean_diff = np.abs(pytorch_output_np - ort_outputs[0]).mean()

            print(f"   - Max difference (PyTorch vs ONNX): {max_diff:.8f}")
            print(f"   - Mean difference: {mean_diff:.8f}")

            if max_diff < 1e-5:
                print("   ✓ Output khớp hoàn toàn!")
            elif max_diff < 1e-3:
                print(f"   ✓ Output gần khớp (chênh lệch nhỏ do floating point)")
            else:
                print(f"   ⚠ Có chênh lệch đáng kể - cần kiểm tra lại!")

        except Exception as e:
            print(f"✗ Lỗi khi kiểm tra ONNX model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    # 7. Hiển thị kích thước file
    pth_size = os.path.getsize(pth_model_path) / (1024 * 1024)
    onnx_size = os.path.getsize(onnx_model_path) / (1024 * 1024)

    print("\n" + "=" * 70)
    print("KẾT QUẢ CHUYỂN ĐỔI")
    print("=" * 70)
    print(f"✓ PyTorch model size: {pth_size:.2f} MB")
    print(f"✓ ONNX model size: {onnx_size:.2f} MB")
    print(f"✓ File ONNX đã được lưu: {onnx_model_path}")
    print("=" * 70)

    return True


def test_onnx_inference(onnx_model_path, input_size=(1, 3, 256, 256)):
    """
    Test inference với ONNX Runtime
    """
    print("\n" + "=" * 70)
    print("TEST ONNX INFERENCE")
    print("=" * 70)

    # Create ONNX Runtime session
    print("\n[1/3] Tạo ONNX Runtime session...")
    ort_session = ort.InferenceSession(onnx_model_path)
    print("✓ Session created!")

    # Hiển thị thông tin inputs/outputs
    print("\n[2/3] Thông tin Model:")
    print("Inputs:")
    for input in ort_session.get_inputs():
        print(f"   - Name: {input.name}, Shape: {input.shape}, Type: {input.type}")

    print("Outputs:")
    for output in ort_session.get_outputs():
        print(f"   - Name: {output.name}, Shape: {output.shape}, Type: {output.type}")

    # Test inference
    print("\n[3/3] Test inference...")
    dummy_input = np.random.randn(*input_size).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}

    import time
    start_time = time.time()
    ort_outputs = ort_session.run(None, ort_inputs)
    inference_time = time.time() - start_time

    print(f"✓ Inference thành công!")
    print(f"   - Input shape: {dummy_input.shape}")
    print(f"   - Output shape: {ort_outputs[0].shape}")
    print(f"   - Inference time: {inference_time*1000:.2f} ms")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert WaterFormer PyTorch model to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert default model
  python convert_to_onnx.py

  # Convert specific model
  python convert_to_onnx.py --input checkpoints/weights.pth \\
                             --output checkpoints/waterformer.onnx

  # Convert and test
  python convert_to_onnx.py --test

  # Custom architecture (if different from default)
  python convert_to_onnx.py --dim 48 --num-blocks 3,3,3,3
        """
    )

    parser.add_argument('--input', type=str,
                        default='/home/ndpthao/eject/IMPLEMENTATION/WaterFormer-master/work_dirs/UW_WaterFormer/models/net_g_best.pth',
                        help='Path to input .pth model file')
    parser.add_argument('--output', type=str,
                        default='/home/ndpthao/eject/IMPLEMENTATION/WaterFormer-master/work_dirs/UW_WaterFormer/models/net_g_best.onnx',
                        help='Path to output .onnx model file')
    parser.add_argument('--height', type=int, default=256,
                        help='Input image height')
    parser.add_argument('--width', type=int, default=256,
                        help='Input image width')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version (use 17+ for better adaptive pooling support)')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip verification after conversion')
    parser.add_argument('--test', action='store_true',
                        help='Run test inference after conversion')

    # WaterFormer architecture parameters
    parser.add_argument('--dim', type=int, default=36,
                        help='Model dimension')
    parser.add_argument('--num-blocks', type=str, default='2,2,2,2',
                        help='Number of blocks per level (comma-separated)')
    parser.add_argument('--heads', type=str, default='2,2,2,2',
                        help='Number of attention heads per level (comma-separated)')
    parser.add_argument('--window-size', type=int, default=8,
                        help='Window size for attention')
    parser.add_argument('--shift-size', type=int, default=3,
                        help='Shift size for shifted window attention')

    args = parser.parse_args()

    # Parse list arguments
    num_blocks = [int(x) for x in args.num_blocks.split(',')]
    heads = [int(x) for x in args.heads.split(',')]

    # Convert to ONNX
    input_size = (1, 3, args.height, args.width)
    success = convert_pth_to_onnx(
        pth_model_path=args.input,
        onnx_model_path=args.output,
        input_size=input_size,
        opset_version=args.opset,
        verify=not args.no_verify,
        dim=args.dim,
        num_blocks=num_blocks,
        heads=heads,
        window_size=args.window_size,
        shift_size=args.shift_size
    )

    # Test inference if requested
    if success and args.test:
        test_onnx_inference(args.output, input_size)

    if success:
        print("\n✓ Hoàn thành!")
    else:
        print("\n✗ Conversion failed!")
        sys.exit(1)
