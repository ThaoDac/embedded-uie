import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
import os

##########################################################################
# WaterFormer architecture WITHOUT einops (for ExecuTorch compatibility)
# All einops.rearrange operations replaced with native PyTorch operations
##########################################################################

def to_3d(x):
    # rearrange(x, 'b c h w -> b (h w) c')
    b, c, h, w = x.shape
    return x.view(b, c, h * w).permute(0, 2, 1)  # b, h*w, c

def to_4d(x, h, w):
    # rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
    b, hw, c = x.shape
    return x.permute(0, 2, 1).view(b, c, h, w)

def window_partition(x, window_size: int, h, w):
    pad_l = pad_t = 0
    pad_r = (window_size - w % window_size) % window_size
    pad_b = (window_size - h % window_size) % window_size
    x = F.pad(x, [pad_l, pad_r, pad_t, pad_b])
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    pad_l = pad_t = 0
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    H = H + pad_b
    W = W + pad_r
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, -1, H // window_size, W // window_size, window_size, window_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
    windows = F.pad(x, [pad_l, -pad_r, pad_t, -pad_b])
    return windows

class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * 3)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.relu(x1) * x2
        x = self.project_out(x)
        return x

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(GlobalAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        # rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.view(b, c, h, w)

        out = self.project_out(out)
        return out

class LocalAttention(nn.Module):
    def __init__(self, dim, window_size, shift_size, bias):
        super(LocalAttention, self).__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)
        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)

        self.qkv_conv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv1 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

    def window_partitions(self, x, window_size: int):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def create_mask(self, x):
        n, c, H, W = x.shape
        Hp = int(math.ceil(H / self.window_size)) * self.window_size
        Wp = int(math.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1

        mask_windows = self.window_partitions(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        shortcut = x
        b, c, h, w = x.shape

        x = window_partition(x, self.window_size, h, w)

        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # rearrange(q, 'b c h w -> b c (h w)')
        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q.transpose(-2, -1) @ k) / self.window_size
        attn = attn.softmax(dim=-1)
        out = (v @ attn)

        # rearrange(out, 'b c (h w) -> b c h w', h=window_size, w=window_size)
        out = out.view(out.shape[0], out.shape[1], self.window_size, self.window_size)
        out = self.project_out(out)
        out = window_reverse(out, self.window_size, h, w)

        shift = torch.roll(out, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        shift_window = window_partition(shift, self.window_size, h, w)
        qkv = self.qkv_dwconv1(self.qkv_conv1(shift_window))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q.transpose(-2, -1) @ k) / self.window_size
        mask = self.create_mask(shortcut)
        attn = attn.view(b, -1, self.window_size * self.window_size,
                         self.window_size * self.window_size) + mask.unsqueeze(0)
        attn = attn.view(-1, self.window_size * self.window_size, self.window_size * self.window_size)
        attn = attn.softmax(dim=-1)

        out = (v @ attn)

        out = out.view(out.shape[0], out.shape[1], self.window_size, self.window_size)

        out = self.project_out1(out)
        out = window_reverse(out, self.window_size, h, w)
        out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        return out

class GLTransBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, bias, adaptor=False):
        super(GLTransBlock, self).__init__()

        self.window_size = window_size
        self.has_adaptor = adaptor

        if adaptor:
            # NOTE: AdaptiveAvgPool2d not supported in ExecuTorch
            # We'll handle this in forward() using mean() instead
            self.adaptor = nn.Embedding(10, dim)
            self.adap_ffn = nn.Linear(window_size**4 * dim, 10)
        else:
            self.adaptor = None
            self.adap_ffn = None

        self.norm1 = LayerNorm(dim)
        self.glob_attn = GlobalAttention(dim, num_heads, bias)
        self.local_attn = LocalAttention(dim, window_size, shift_size, bias)

        # NOTE: Replace AdaptiveAvgPool2d(1) with global mean (ExecuTorch compatible)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=True)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, bias)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x

        if self.has_adaptor and self.adaptor is not None:
            # Replace AdaptiveAvgPool2d(window_size^2) with equivalent operations
            # AdaptiveAvgPool2d behavior:
            # - If H >= target_size: downsample using avg_pool2d
            # - If H < target_size: upsample using interpolate(area)
            target_size = self.window_size ** 2  # 8^2 = 64

            if H >= target_size:
                # Downsample case: use avg_pool2d (mathematically identical to AdaptiveAvgPool2d)
                kernel_size = H // target_size  # e.g., 256 // 64 = 4
                x_adap = F.avg_pool2d(x, kernel_size=kernel_size, stride=kernel_size)
            else:
                # Upsample case: use interpolate with mode='area' (equivalent to AdaptiveAvgPool2d)
                # This happens at latent level where H=32 but target=64
                x_adap = F.interpolate(x, size=(target_size, target_size), mode='area')

            x_adap = x_adap.reshape(B, -1)
            water_type = torch.argmax(F.softmax(self.adap_ffn(x_adap), dim=1), dim=1)
            type_embed = self.adaptor(water_type)
            x = x + type_embed[..., None, None]

        x_norm = self.norm1(x)

        y1 = self.glob_attn(x_norm)
        y1 = shortcut + y1

        y2 = self.local_attn(x_norm)
        y2 = shortcut + y2

        # Replace AdaptiveAvgPool2d(1) with mean over spatial dims (ExecuTorch compatible)
        alpha = torch.sigmoid(y1.mean(dim=(2, 3), keepdim=True))

        y = alpha * y1 + (1 - alpha) * y2

        y = y + self.ffn(self.norm2(y))

        return y

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat, n_out):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_feat, n_out * 4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)

def cat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)

    return x

class WaterFormerExport(nn.Module):
    """WaterFormer without einops for ExecuTorch export"""
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim=36,
        num_blocks=[2,2,2,2],
        heads=[2,2,2,2],
        bias=False,
        window_size=8,
        shift_size=3
    ):
        super(WaterFormerExport, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            GLTransBlock(dim=dim, num_heads=heads[0], bias=bias, window_size=window_size, shift_size=shift_size) for
            i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            GLTransBlock(dim=dim * 2 ** 1, num_heads=heads[1], bias=bias, window_size=window_size,
                             shift_size=shift_size) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[
            GLTransBlock(dim=dim * 2 ** 2, num_heads=heads[2], bias=bias, window_size=window_size,
                             shift_size=shift_size) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[
            GLTransBlock(dim=dim * 2 ** 3, num_heads=heads[3], bias=bias, window_size=window_size,
                             shift_size=shift_size, adaptor=(i+1)//2) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3), int(dim * 2**2))
        self.skip_connect3 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            GLTransBlock(dim=int(dim * 2 ** 3), num_heads=heads[2], bias=bias, window_size=window_size,
                             shift_size=shift_size) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 3), int(dim * 2))
        self.skip_connect2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            GLTransBlock(dim=int(dim * 2 ** 2), num_heads=heads[1], bias=bias, window_size=window_size,
                             shift_size=shift_size) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 **2), int(dim))
        self.skip_connect1 = nn.Conv2d(int(dim), int(dim), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            GLTransBlock(dim=int(dim * 2), num_heads=heads[0], bias=bias, window_size=window_size,
                             shift_size=shift_size) for i in range(num_blocks[0])])

        self.output = nn.Conv2d(int(dim*2), out_channels, kernel_size=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)

        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = cat(inp_dec_level3, self.skip_connect3(out_enc_level3))
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = cat(inp_dec_level2, self.skip_connect2(out_enc_level2))
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = cat(inp_dec_level1, self.skip_connect1(out_enc_level1))
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        ref_out = out_dec_level1

        out = self.output(ref_out) + inp_img

        return out


def convert_to_executorch(
    checkpoint_path='/home/ndpthao/eject/IMPLEMENTATION/WaterFormer-master/work_dirs/UW_WaterFormer/models/net_g_best.pth',
    output_path='/home/ndpthao/eject/IMPLEMENTATION/WaterFormer-master/work_dirs/UW_WaterFormer/models/net_g_best.pte',
    input_size=(1, 3, 256, 256)
):
    """
    Convert WaterFormer PyTorch model to ExecuTorch format with XNNPACK delegation

    This creates an optimized .pte file for x86_64 and ARM (Raspberry Pi)
    using XNNPACK delegate for CPU performance optimization.

    Args:
        checkpoint_path: Path to the .pth checkpoint file
        output_path: Path to save the .pte file
        input_size: Input tensor size (batch, channels, height, width)
    """
    print(f"Loading model from {checkpoint_path}...")

    # Load model (using export-compatible architecture without einops)
    model = WaterFormerExport(
        inp_channels=3,
        out_channels=3,
        dim=36,
        num_blocks=[2, 2, 2, 2],
        heads=[2, 2, 2, 2],
        bias=False,
        window_size=8,
        shift_size=3
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['params'], strict=False)
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