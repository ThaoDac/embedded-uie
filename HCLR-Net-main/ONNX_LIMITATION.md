# ONNX Export Limitation for HCLR-Net

## Issue

HCLR-Net model **CANNOT be exported to ONNX format** due to the following incompatibilities:

### 1. Adaptive Pooling with Dynamic Dimensions

In `utils.py`, the `Attention` class uses:

```python
self.avg = nn.AdaptiveAvgPool2d((None, 1))  # Line 51
self.max = nn.AdaptiveMaxPool2d((1, None))  # Line 52
```

**Problem**: ONNX does not support adaptive pooling operations where the output size contains `None` (dynamic dimensions). ONNX requires constant output sizes.

### 2. Dynamic Interpolation

In `network.py`, the forward pass uses:

```python
x_up1 = F.interpolate(x_up1, x_down2.size()[2:], mode='bilinear', align_corners=True)  # Line 60
x_up2 = F.interpolate(x_up2, x_down1.size()[2:], mode='bilinear', align_corners=True)  # Line 64
x_up3 = F.interpolate(x_up3, x.size()[2:], mode='bilinear', align_corners=True)        # Line 68
```

**Problem**: The target size for interpolation is computed dynamically from other tensor sizes during forward pass. ONNX export requires these sizes to be known at export time.

## Error Message

```
✗ Error during ONNX export: Unsupported: ONNX export of operator adaptive pooling,
since output_size is not constant.
```

## Workarounds

### Option 1: Use PyTorch Model Directly (Recommended)

Continue using the `.ckpt` (PyTorch Lightning checkpoint) format for inference:

```python
from train import CoolSystem
from argparse import Namespace

# Load checkpoint
hparams = Namespace(**hparams_dict)
model = CoolSystem.load_from_checkpoint(checkpoint_path, hparams=hparams)
network = model.model
network.eval()

# Inference
with torch.no_grad():
    output = network(input_tensor)
```

This is the approach used in `app_console.py`.

### Option 2: Export to TorchScript (JIT)

TorchScript supports dynamic operations better than ONNX:

```python
import torch

# Load model
network = load_your_model()
network.eval()

# Create example input
example_input = torch.randn(1, 3, 256, 256)

# Export to TorchScript
traced_model = torch.jit.trace(network, example_input)
traced_model.save("hclr_net.pt")

# Load and use
loaded_model = torch.jit.load("hclr_net.pt")
output = loaded_model(input_tensor)
```

### Option 3: Modify Model Architecture (Not Recommended)

To make the model ONNX-compatible, you would need to:

1. **Replace adaptive pooling** in `Attention` class with fixed-size pooling or global pooling
2. **Replace dynamic interpolation** with fixed-size upsampling or remove skip connections
3. **Test thoroughly** as these changes will affect model accuracy

**Warning**: This requires retraining the model and will likely change the model's performance.

## Deployment Options

### For Production Deployment:

1. **PyTorch Serving**: Use TorchServe or similar PyTorch-based serving solutions
2. **TorchScript**: Export to `.pt` format (supports dynamic operations)
3. **Python API**: Directly use PyTorch model in Python environments
4. **NVIDIA Triton**: Supports PyTorch models natively

### Mobile/Edge Deployment:

1. **PyTorch Mobile**: Use TorchScript `.pt` files
2. **NVIDIA TensorRT**: Can optimize PyTorch models directly (with some limitations)

## Conclusion

**ONNX export is NOT supported for HCLR-Net** due to architectural limitations. Use PyTorch checkpoint (.ckpt) or TorchScript (.pt) instead.

For inference, use the provided `app_console.py` script which loads the checkpoint directly.
