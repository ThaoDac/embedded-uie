# ONNX Conversion Limitation for UIR-PolyKernel

## Issue

UIR-PolyKernel model **cannot be converted to ONNX** due to the following limitation:

- The model uses **Frequency-Domain Pixel Attention (FDPA)** which relies on `torch.fft.fft2` operation
- PyTorch's FFT operations (`torch.fft.fft2`, `torch.fft.ifft2`) are **NOT supported** in ONNX export
- This is a known PyTorch limitation - see: https://github.com/pytorch/pytorch/issues

## Affected Components

The FDPA block in `models/model.py` (lines 217-240):
```python
class FDPA(nn.Module):
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        x2_fft = torch.fft.fft2(x2)  # ❌ Not supported in ONNX
        out = x1 * x2_fft
        out = torch.fft.ifft2(out, dim=(-2,-1))  # ❌ Not supported in ONNX
        out = torch.abs(out)

        return out * self.alpha + x * self.beta
```

## Recommendations

1. **Use PyTorch model directly** - The PyTorch `.pth` model works perfectly
2. **Use TorchScript instead** - Convert to TorchScript for deployment:
   ```python
   scripted_model = torch.jit.script(model)
   scripted_model.save('model.pt')
   ```
3. **Wait for PyTorch updates** - Future PyTorch versions may add FFT support to ONNX

## Files Affected

The following files are **NOT functional** for UIR-PolyKernel:
- ❌ `convert_to_onnx.py` - Cannot convert model
- ❌ `inference_onnx.py` - No ONNX model available
- ❌ `benchmark_pytorch_vs_onnx.py` - Cannot compare without ONNX model

## Working Files

These files work correctly:
- ✅ `app_console.py` - PyTorch inference with full metrics
- ✅ `app.py` - Flask web interface
- ✅ `test.py` - Original test script

## Alternative: Modify Model Architecture

To enable ONNX export, you would need to:
1. Remove or replace FDPA blocks
2. Use spatial-domain operations instead of frequency-domain
3. Retrain the model

This would change the model architecture significantly and require retraining.
