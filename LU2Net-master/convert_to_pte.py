import torch
import LU2Net

def convert_to_executorch(
    checkpoint_path='/home/ndpthao/eject/IMPLEMENTATION/LU2Net-master/checkpoints/LightUNet_200.pth',
    output_path='/home/ndpthao/eject/IMPLEMENTATION/LU2Net-master/checkpoints/LightUNet_200.pte',
    input_size=(1, 3, 256, 256)
):
    """
    Convert LU2Net PyTorch model to ExecuTorch format with XNNPACK delegation

    This creates an optimized .pte file for x86_64 and ARM (Raspberry Pi)
    using XNNPACK delegate for CPU performance optimization.

    Args:
        checkpoint_path: Path to the .pth checkpoint file
        output_path: Path to save the .pte file
        input_size: Input tensor size (batch, channels, height, width)
    """
    print(f"Loading model from {checkpoint_path}...")

    # Load model
    model = LU2Net.LU2Net()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Input size: {input_size}")

    # Create sample input
    sample_inputs = (torch.randn(input_size),)

    # Validate model output with sample input
    print("Validating model before export...")
    with torch.no_grad():
        original_output = model(*sample_inputs)
    print(f"  Original output shape: {original_output.shape}")
    print(f"  Original output range: [{original_output.min():.4f}, {original_output.max():.4f}]")

    # Export to ExecuTorch with XNNPACK delegation
    print("\nExporting to ExecuTorch format with XNNPACK delegation...")

    # Step 1: torch.export with strict=False to handle dynamic operations
    try:
        exported_program = torch.export.export(
            model,
            sample_inputs,
            strict=False
        )
        print("  ✓ torch.export completed")
    except Exception as e:
        print(f"  ✗ torch.export failed: {e}")
        print("  Trying with strict=True...")
        exported_program = torch.export.export(model, sample_inputs)
        print("  ✓ torch.export completed (strict mode)")

    # Step 2: Convert to Edge dialect with XNNPACK delegation (RECOMMENDED WAY)
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    print("  Applying XNNPACK delegation and lowering to Edge IR...")

    # Use the RECOMMENDED workflow: to_edge_transform_and_lower
    edge_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()]
    )
    print("  ✓ XNNPACK delegation and Edge lowering completed")

    # Step 3: Convert to ExecuTorch
    et_program = edge_program.to_executorch()
    print("  ✓ to_executorch completed")

    # Save the .pte file
    print(f"\nSaving to {output_path}...")
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    print(f"✓ Conversion successful! Model saved to {output_path}")

    # Get file size
    import os
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size:.2f} MB")

    # Validation with ExecuTorch runtime
    print("\n" + "="*70)
    print("VALIDATING EXECUTORCH MODEL")
    print("="*70)
    try:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch

        et_model = _load_for_executorch(output_path)
        et_output = et_model.forward(list(sample_inputs))[0]

        print(f"  ExecuTorch output shape: {et_output.shape}")
        print(f"  ExecuTorch output range: [{et_output.min():.4f}, {et_output.max():.4f}]")

        # Compare outputs
        diff = torch.abs(original_output - et_output).max()
        mean_diff = torch.abs(original_output - et_output).mean()

        print(f"\n  Max difference: {diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")

        if diff < 1e-3:
            print("  ✓ VALIDATION PASSED: Outputs match closely!")
        else:
            print(f"  ⚠ WARNING: Large difference detected ({diff:.6f})")
            print("  This may indicate numerical precision issues in conversion.")
    except Exception as e:
        print(f"  ⚠ Could not validate: {e}")
    print("="*70)

if __name__ == "__main__":
    convert_to_executorch()