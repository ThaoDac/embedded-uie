#!/usr/bin/env python3
"""
UIR-PolyKernel PyTorch to ONNX Conversion Script
"""

import os
import argparse
import torch
import onnx
import onnxruntime as ort
import numpy as np
from models import UIR_PolyKernel


def convert_pth_to_onnx(
    pth_model_path='models/UIR_PolyKernel_epoch_37.pth',
    onnx_model_path='models/UIR_PolyKernel_epoch_37.onnx',
    input_size=(1, 3, 256, 256),
    opset_version=17,
    verify=True
):
    print("=" * 70)
    print("UIR-PolyKernel PYTORCH TO ONNX CONVERSION")
    print("=" * 70)

    if not os.path.exists(pth_model_path):
        raise FileNotFoundError(f"Model file not found: {pth_model_path}")

    print(f"\n✓ PyTorch Model: {pth_model_path}")
    print(f"✓ Output ONNX: {onnx_model_path}")
    print(f"✓ Input size: {input_size}")
    print(f"✓ Opset version: {opset_version}")

    print("\n[1/5] Loading PyTorch model...")
    device = torch.device('cpu')
    model = UIR_PolyKernel()

    checkpoint = torch.load(pth_model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print("✓ Loaded 'model' from checkpoint!")
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("✓ Loaded 'state_dict' from checkpoint!")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Loaded checkpoint directly!")

    model.to(device)
    model.eval()
    print("✓ PyTorch model loaded successfully!")

    print("\n[2/5] Creating dummy input tensor...")
    dummy_input = torch.randn(input_size, device=device)
    print(f"✓ Dummy input shape: {dummy_input.shape}")

    print("\n[3/5] Testing forward pass...")
    with torch.no_grad():
        try:
            output = model(dummy_input)
            print(f"✓ Forward pass successful!")
            print(f"✓ Output shape: {output.shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {str(e)}")
            return False

    print("\n[4/5] Exporting to ONNX...")
    print("   (This may take a few seconds...)")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_model_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        print(f"✓ Model exported successfully to {onnx_model_path}")
    except Exception as e:
        print(f"✗ ONNX export failed: {str(e)}")
        return False

    if verify:
        print("\n[5/5] Verifying ONNX model...")

        try:
            onnx_model = onnx.load(onnx_model_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model structure is valid!")

            print("\n   Testing with ONNX Runtime...")
            ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            print(f"   - Input name: {input_name}")
            print(f"   - Output name: {output_name}")

            ort_inputs = {input_name: dummy_input.numpy()}
            ort_outputs = ort_session.run([output_name], ort_inputs)

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
        description="Convert UIR-PolyKernel PyTorch model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--pth_model', type=str, default='models/UIR_PolyKernel_epoch_37.pth',
                       help='Path to PyTorch model (.pth file)')
    parser.add_argument('--onnx_model', type=str, default='models/UIR_PolyKernel_epoch_37.onnx',
                       help='Output path for ONNX model (.onnx file)')
    parser.add_argument('--input_size', type=int, nargs=4, default=[1, 3, 256, 256],
                       metavar=('BATCH', 'CHANNELS', 'HEIGHT', 'WIDTH'),
                       help='Input tensor size (default: 1 3 256 256)')
    parser.add_argument('--opset_version', type=int, default=17,
                       help='ONNX opset version (default: 17, required for FFT ops)')
    parser.add_argument('--no_verify', action='store_true',
                       help='Skip ONNX model verification')

    args = parser.parse_args()

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
