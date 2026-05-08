"""
Convert HCLR-Net PyTorch Lightning checkpoint to ONNX format
"""

import torch
import torch.onnx
import os
import sys
from argparse import ArgumentParser
from train import CoolSystem
from argparse import Namespace

def convert_checkpoint_to_onnx(
    checkpoint_path,
    output_path=None,
    input_size=(1, 3, 256, 256),
    opset_version=11,
    simplify=True
):
    """
    Convert PyTorch Lightning checkpoint to ONNX format

    Args:
        checkpoint_path: Path to .ckpt file
        output_path: Path to save .onnx file (optional)
        input_size: Input tensor size (batch, channels, height, width)
        opset_version: ONNX opset version
        simplify: Whether to simplify the ONNX model using onnx-simplifier
    """

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Create model configuration (same as in train.py)
    hparams_dict = {
        'train_datasets': '',
        'test_datasets': None,
        'val_datasets': '',
        'train_bs': 16,
        'test_bs': 1,
        'val_bs': 8,
        'initlr': 0.0001,
        'weight_decay': 0.001,
        'crop_size': 256,
        'num_workers': 16,
        'model_blocks': 5,
        'chns': 64
    }
    hparams = Namespace(**hparams_dict)

    # Load model from checkpoint
    model = CoolSystem.load_from_checkpoint(
        checkpoint_path,
        hparams=hparams,
        map_location='cpu'
    )

    # Extract only the network part (not the whole lightning module)
    network = model.model
    network.eval()

    # Set output path
    if output_path is None:
        base_name = os.path.splitext(checkpoint_path)[0]
        output_path = base_name + '.onnx'

    # Create dummy input
    dummy_input = torch.randn(*input_size)

    print(f"Converting to ONNX with input shape: {input_size}")
    print(f"Output path: {output_path}")

    # Export to ONNX
    try:
        torch.onnx.export(
            network,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            },
            verbose=False
        )
        print(f"✓ Successfully exported to ONNX: {output_path}")

        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Model size: {file_size:.2f} MB")

    except Exception as e:
        print(f"✗ Error during ONNX export: {e}")
        return False

    # Simplify ONNX model (optional)
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            print("Simplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            model_simplified, check = onnx_simplify(onnx_model)

            if check:
                onnx.save(model_simplified, output_path)
                print("✓ ONNX model simplified successfully")

                # Get new file size
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  Simplified model size: {file_size:.2f} MB")
            else:
                print("! Simplification check failed, using original model")

        except ImportError:
            print("! onnx-simplifier not installed. Skipping simplification.")
            print("  Install with: pip install onnx-simplifier")
        except Exception as e:
            print(f"! Error during simplification: {e}")
            print("  Using original ONNX model")

    # Verify the ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed")
    except ImportError:
        print("! onnx package not installed for verification")
    except Exception as e:
        print(f"! ONNX model verification failed: {e}")

    return True


def test_onnx_inference(onnx_path, checkpoint_path, test_image_size=(1, 3, 256, 256)):
    """
    Test ONNX model inference and compare with PyTorch model

    Args:
        onnx_path: Path to ONNX model
        checkpoint_path: Path to original checkpoint
        test_image_size: Size of test input
    """
    print("\n" + "="*60)
    print("Testing ONNX inference...")
    print("="*60)

    try:
        import onnxruntime as ort
        import numpy as np

        # Load ONNX model
        ort_session = ort.InferenceSession(onnx_path)

        # Create test input
        test_input = torch.randn(*test_image_size)

        # Run ONNX inference
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_output = ort_outputs[0]

        print(f"✓ ONNX inference successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {onnx_output.shape}")

        # Compare with PyTorch model
        print("\nComparing with PyTorch model...")
        hparams_dict = {
            'train_datasets': '',
            'test_datasets': None,
            'val_datasets': '',
            'train_bs': 16,
            'test_bs': 1,
            'val_bs': 8,
            'initlr': 0.0001,
            'weight_decay': 0.001,
            'crop_size': 256,
            'num_workers': 16,
            'model_blocks': 5,
            'chns': 64
        }
        hparams = Namespace(**hparams_dict)

        model = CoolSystem.load_from_checkpoint(checkpoint_path, hparams=hparams, map_location='cpu')
        network = model.model
        network.eval()

        with torch.no_grad():
            pytorch_output = network(test_input).numpy()

        # Calculate difference
        diff = np.abs(onnx_output - pytorch_output)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")

        if max_diff < 1e-4:
            print("✓ ONNX and PyTorch outputs match closely!")
        elif max_diff < 1e-3:
            print("! Small difference detected (acceptable)")
        else:
            print("✗ Large difference detected - please verify")

    except ImportError as e:
        print(f"! Cannot test inference: {e}")
        print("  Install with: pip install onnxruntime")
    except Exception as e:
        print(f"✗ Error during inference test: {e}")


def main():
    parser = ArgumentParser(description='Convert HCLR-Net checkpoint to ONNX')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/home/ndpthao/eject/IMPLEMENTATION/HCLR-Net-main/checkpoints/last-epoch46.ckpt',
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/ndpthao/eject/IMPLEMENTATION/HCLR-Net-main/checkpoints/last-epoch46.onnx',
        help='Output ONNX file path (default: same as checkpoint with .onnx extension)'
    )
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=4,
        default=[1, 3, 256, 256],
        help='Input size as: batch channels height width (default: 1 3 256 256)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=11,
        help='ONNX opset version (default: 11)'
    )
    parser.add_argument(
        '--no-simplify',
        action='store_true',
        help='Skip ONNX model simplification'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test ONNX inference after conversion'
    )

    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    # Convert to ONNX
    success = convert_checkpoint_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_size=tuple(args.input_size),
        opset_version=args.opset,
        simplify=not args.no_simplify
    )

    if not success:
        sys.exit(1)

    # Test inference if requested
    if args.test:
        output_path = args.output if args.output else os.path.splitext(args.checkpoint)[0] + '.onnx'
        test_onnx_inference(output_path, args.checkpoint, tuple(args.input_size))

    print("\n" + "="*60)
    print("Conversion completed!")
    print("="*60)


if __name__ == '__main__':
    main()
