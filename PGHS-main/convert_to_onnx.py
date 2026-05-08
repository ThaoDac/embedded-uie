#!/usr/bin/env python3
"""
PGHS PyTorch to ONNX Conversion Script
=======================================

Convert PGHS (Prior-Guided Heuristic System) PyTorch model to ONNX format
for optimized inference and deployment.

Features:
- Supports standard PyTorch checkpoints
- Automatic model architecture loading
- ONNX verification and validation
- Dynamic batch size support

Usage:
    python convert_to_onnx.py \
        --pth_model checkpoints/uw_epoch_79.pth \
        --onnx_model checkpoints/uw_epoch_79.onnx \
        --input_size 1 3 256 256

Author: Claude Code
Date: 2025-12-15
"""

import os
import argparse
import torch
import onnx
import onnxruntime as ort
import numpy as np
from models import Model


def convert_pth_to_onnx(
    pth_model_path='models/uw_epoch_79.pth',
    onnx_model_path='models/uw_epoch_79.onnx',
    input_size=(1, 3, 256, 256),
    opset_version=13,
    verify=True
):
    """
    Convert PGHS PyTorch model (.pth) to ONNX format.

    Args:
        pth_model_path: Path to .pth checkpoint file
        onnx_model_path: Output .onnx file path
        input_size: Input tensor shape (batch, channels, height, width)
        opset_version: ONNX opset version (recommended: 11 or higher)
        verify: Verify ONNX model after conversion

    Returns:
        bool: True if successful, False otherwise
    """

    print("=" * 70)
    print("PGHS PYTORCH TO ONNX CONVERSION")
    print("=" * 70)

    # 1. Check if PyTorch model file exists
    if not os.path.exists(pth_model_path):
        raise FileNotFoundError(f"Model file not found: {pth_model_path}")

    print(f"\n✓ PyTorch Model: {pth_model_path}")
    print(f"✓ Output ONNX: {onnx_model_path}")
    print(f"✓ Input size: {input_size}")
    print(f"✓ Opset version: {opset_version}")

    # 2. Load PyTorch model
    print("\n[1/5] Loading PyTorch model...")
    device = torch.device('cpu')  # Use CPU for export
    model = Model()

    # Load pretrained weights
    checkpoint = torch.load(pth_model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("✓ Loaded state_dict from checkpoint!")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Loaded checkpoint directly!")

    model.to(device)
    model.eval()
    print("✓ PyTorch model loaded successfully!")

    # 3. Create dummy input
    print("\n[2/5] Creating dummy input tensor...")
    dummy_input = torch.randn(input_size, device=device)
    print(f"✓ Dummy input shape: {dummy_input.shape}")

    # Test forward pass
    print("\n[3/5] Testing forward pass...")
    with torch.no_grad():
        try:
            output = model(dummy_input)
            print(f"✓ Forward pass successful!")
            print(f"✓ Output shape: {output.shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {str(e)}")
            return False

    # 4. Export to ONNX
    print("\n[4/5] Exporting to ONNX...")
    print("   (This may take a few seconds...)")

    try:
        torch.onnx.export(
            model,                          # PyTorch model
            dummy_input,                    # Dummy input
            onnx_model_path,                # Output file path
            export_params=True,             # Save trained parameters
            opset_version=opset_version,    # ONNX opset version
            do_constant_folding=True,       # Optimize constant folding
            input_names=['input'],          # Input names
            output_names=['output'],        # Output names
            dynamic_axes={
                'input': {0: 'batch_size'},     # Dynamic batch size
                'output': {0: 'batch_size'}     # Dynamic batch size
            },
            verbose=False
        )
        print(f"✓ Model exported successfully to {onnx_model_path}")
    except Exception as e:
        print(f"✗ ONNX export failed: {str(e)}")
        return False

    # 5. Verify ONNX model
    if verify:
        print("\n[5/5] Verifying ONNX model...")

        try:
            # Load ONNX model
            onnx_model = onnx.load(onnx_model_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model structure is valid!")

            # Test with ONNX Runtime
            print("\n   Testing with ONNX Runtime...")
            ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

            # Get input/output names
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            print(f"   - Input name: {input_name}")
            print(f"   - Output name: {output_name}")

            # Run inference
            ort_inputs = {input_name: dummy_input.numpy()}
            ort_outputs = ort_session.run([output_name], ort_inputs)

            # Compare outputs
            pytorch_output = output.detach().cpu().numpy()
            onnx_output = ort_outputs[0]

            max_diff = np.abs(pytorch_output - onnx_output).max()
            mean_diff = np.abs(pytorch_output - onnx_output).mean()

            print(f"\n   Output comparison:")
            print(f"   - Max difference: {max_diff:.6f}")
            print(f"   - Mean difference: {mean_diff:.6f}")

            if max_diff < 1e-4:
                print("   ✓ Outputs match! (difference < 1e-4)")
            elif max_diff < 1e-3:
                print("   ⚠ Outputs mostly match (difference < 1e-3)")
            else:
                print(f"   ⚠ Warning: Large output difference ({max_diff:.6f})")

        except Exception as e:
            print(f"✗ Verification failed: {str(e)}")
            return False

    # 6. Model size comparison
    print("\n" + "=" * 70)
    print("MODEL SIZE COMPARISON")
    print("=" * 70)
    pth_size = os.path.getsize(pth_model_path) / (1024 ** 2)
    onnx_size = os.path.getsize(onnx_model_path) / (1024 ** 2)
    print(f"  PyTorch (.pth) : {pth_size:.2f} MB")
    print(f"  ONNX (.onnx)   : {onnx_size:.2f} MB")
    print(f"  Size change    : {((onnx_size / pth_size - 1) * 100):+.1f}%")
    print("=" * 70)

    print("\n✓ Conversion completed successfully!")
    print(f"\nONNX model saved to: {onnx_model_path}")
    print("\nNext steps:")
    print("  1. Test ONNX model: python inference_onnx.py --input test.jpg --output output.jpg")
    print("  2. Quantize model: python quantize_onnx.py (if needed)")
    print("  3. Deploy with ONNX Runtime for optimized inference")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert PGHS PyTorch model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion
    python convert_to_onnx.py

    # Custom paths
    python convert_to_onnx.py \\
        --pth_model models/uw_epoch_79.pth \\
        --onnx_model models/uw_epoch_79.onnx

    # Custom input size
    python convert_to_onnx.py \\
        --input_size 1 3 512 512

    # Different opset version
    python convert_to_onnx.py --opset_version 14
        """
    )

    parser.add_argument(
        '--pth_model',
        type=str,
        default='models/uw_epoch_79.pth',
        help='Path to PyTorch model (.pth file)'
    )

    parser.add_argument(
        '--onnx_model',
        type=str,
        default='models/uw_epoch_79.onnx',
        help='Output path for ONNX model (.onnx file)'
    )

    parser.add_argument(
        '--input_size',
        type=int,
        nargs=4,
        default=[1, 3, 256, 256],
        metavar=('BATCH', 'CHANNELS', 'HEIGHT', 'WIDTH'),
        help='Input tensor size (default: 1 3 256 256)'
    )

    parser.add_argument(
        '--opset_version',
        type=int,
        default=13,
        help='ONNX opset version (default: 13)'
    )

    parser.add_argument(
        '--no_verify',
        action='store_true',
        help='Skip ONNX model verification'
    )

    args = parser.parse_args()

    # Convert
    success = convert_pth_to_onnx(
        pth_model_path=args.pth_model,
        onnx_model_path=args.onnx_model,
        input_size=tuple(args.input_size),
        opset_version=args.opset_version,
        verify=not args.no_verify
    )

    if success:
        print("\n✓ All done!")
        return 0
    else:
        print("\n✗ Conversion failed!")
        return 1


if __name__ == '__main__':
    exit(main())
