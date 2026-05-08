import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
from LU2Net_onnx import LU2Net


def _clean_qat_state_dict(state_dict):
    """
    Remove fake-quant/observer entries and remap fused BN keys so a plain
    LU2Net_onnx model can load QAT checkpoints.
    """
    cleaned = {}
    for k, v in state_dict.items():
        if ("activation_post_process" in k or
            "weight_fake_quant" in k or
            "fake_quant_enabled" in k or
            "observer_enabled" in k):
            continue
        # QAT fuse_fx may nest BN under pw conv (e.g., d4.pw.bn.*); map back
        if ".pw.bn." in k:
            k = k.replace(".pw.bn.", ".bn.")
        cleaned[k] = v
    return cleaned


def convert_pth_to_onnx(
    pth_model_path='/home/ndpthao/eject/IMPLEMENTATION/LU2Net-master/checkpoints/LightUNet_200.pth',
    onnx_model_path='/home/ndpthao/eject/IMPLEMENTATION/LU2Net-master/checkpoints/LightUNet_200.onnx',
    input_size=(1, 3, 256, 256),
    opset_version=13,  # Changed from 11 to 13 for QDQ quantization support
    verify=True
):
    """
    Chuyển đổi model PyTorch (.pth) sang ONNX format

    Args:
        pth_model_path: Đường dẫn đến file .pth
        onnx_model_path: Đường dẫn output file .onnx
        input_size: Kích thước input (batch, channels, height, width)
        opset_version: ONNX opset version (khuyến nghị: 11 hoặc cao hơn)
        verify: Kiểm tra model ONNX sau khi convert
    """

    print("=" * 70)
    print("CHUYỂN ĐỔI PYTORCH MODEL SANG ONNX")
    print("=" * 70)

    # 1. Kiểm tra file .pth có tồn tại không
    if not os.path.exists(pth_model_path):
        raise FileNotFoundError(f"Không tìm thấy file model: {pth_model_path}")

    print(f"\n✓ File model PyTorch: {pth_model_path}")
    print(f"✓ Output ONNX: {onnx_model_path}")
    print(f"✓ Input size: {input_size}")
    print(f"✓ Opset version: {opset_version}")

    # 2. Load model PyTorch
    print("\n[1/5] Đang load model PyTorch...")
    device = torch.device('cpu')  # Sử dụng CPU cho việc export
    model = LU2Net()

    # Load pretrained weights (support both float and QAT checkpoints)
    state_dict = torch.load(pth_model_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
        print("✓ Model PyTorch loaded thành công!")
    except RuntimeError as e:
        print("⚠ Load trực tiếp thất bại, thử loại bỏ tham số QAT/fake-quant...")
        cleaned = _clean_qat_state_dict(state_dict)
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print("✓ Model PyTorch loaded sau khi làm sạch state_dict!")
        if missing:
            print(f"   Thiếu key (đã dùng strict=False): {missing}")
        if unexpected:
            print(f"   Key thừa (đã bỏ qua): {unexpected}")
    model.to(device)
    model.eval()

    # 3. Tạo dummy input
    print("\n[2/5] Tạo dummy input tensor...")
    dummy_input = torch.randn(input_size, device=device)
    print(f"✓ Dummy input shape: {dummy_input.shape}")

    # 4. Export sang ONNX
    print("\n[3/5] Đang export sang ONNX...")
    print("   (Quá trình này có thể mất vài giây...)")

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

    # 5. Kiểm tra file ONNX
    if verify:
        print("\n[4/5] Kiểm tra model ONNX...")
        try:
            # Load và validate ONNX model
            onnx_model = onnx.load(onnx_model_path)
            onnx.checker.check_model(onnx_model)
            print("✓ Model ONNX hợp lệ!")

            # Hiển thị thông tin model
            print("\n[5/5] Thông tin ONNX model:")
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
            with torch.no_grad():
                pytorch_output = model(dummy_input).cpu().numpy()

            max_diff = np.abs(pytorch_output - ort_outputs[0]).max()
            print(f"   - Max difference (PyTorch vs ONNX): {max_diff:.8f}")

            if max_diff < 1e-5:
                print("   ✓ Output khớp hoàn toàn!")
            else:
                print(f"   ⚠ Có chênh lệch nhỏ (có thể do floating point precision)")

        except Exception as e:
            print(f"✗ Lỗi khi kiểm tra ONNX model: {str(e)}")
            return False

    # 6. Hiển thị kích thước file
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


def test_onnx_inference(onnx_model_path='./LightUNet_170.onnx', input_size=(1, 3, 256, 256)):
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

    parser = argparse.ArgumentParser(description="Convert LU2Net PyTorch model to ONNX")
    parser.add_argument('--input', type=str, default='/home/ndpthao/eject/IMPLEMENTATION/LU2Net-master/checkpoints/LightUNet_200.pth',
                        help='Path to input .pth model file')
    parser.add_argument('--output', type=str, default='/home/ndpthao/eject/IMPLEMENTATION/LU2Net-master/checkpoints/LightUNet_200.onnx',
                        help='Path to output .onnx model file')
    parser.add_argument('--height', type=int, default=256,
                        help='Input image height')
    parser.add_argument('--width', type=int, default=256,
                        help='Input image width')
    parser.add_argument('--opset', type=int, default=13,
                        help='ONNX opset version (13+ required for QDQ quantization)')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip verification after conversion')
    parser.add_argument('--test', action='store_true',
                        help='Run test inference after conversion')

    args = parser.parse_args()

    # Convert to ONNX
    input_size = (1, 3, args.height, args.width)
    success = convert_pth_to_onnx(
        pth_model_path=args.input,
        onnx_model_path=args.output,
        input_size=input_size,
        opset_version=args.opset,
        verify=not args.no_verify
    )

    # Test inference if requested
    if success and args.test:
        test_onnx_inference(args.output, input_size)

    print("\n✓ Hoàn thành!")
