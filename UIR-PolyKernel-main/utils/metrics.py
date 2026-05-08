import torch
import math
from skimage.metrics import structural_similarity as ssim_func
import numpy as np
from PIL import Image

def calc_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

def calc_ssim(img1, img2):
    a = img1.squeeze().cpu().numpy().transpose(1,2,0)
    b = img2.squeeze().cpu().numpy().transpose(1,2,0)
    return ssim_func(a, b, channel_axis=2, data_range=1.0)

def calc_uiqm(path):
    # giả lập đơn giản, có thể thay bằng thư viện UIQM thật
    return np.random.uniform(1, 5)
