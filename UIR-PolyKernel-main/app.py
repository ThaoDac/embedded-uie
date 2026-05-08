import os
import time
import cv2
import torch
import numpy as np
import psutil
from flask import Flask, render_template, request
from PIL import Image
from torchvision import transforms, utils as vutils
from scipy.ndimage import gaussian_filter
from skimage.util import view_as_windows
from accelerate import Accelerator

from config import Config
from data import get_data
from models import *
from utils import load_checkpoint, seed_everything
from metrics import (
    calculate_performance_metrics,
    evaluate_all_image_metrics
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

opt = Config('config.yml')
seed_everything(opt.OPTIM.SEED)

accelerator = Accelerator()
device = accelerator.device

# Tạo model
model = UIR_PolyKernel()
load_checkpoint(model, opt.TESTING.WEIGHT)
model = accelerator.prepare(model)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

# Helper: Tensor → Image
def tensor_to_img(tensor):
    img = tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

# Flask 
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return render_template('index.html', error='Vui lòng chọn ảnh!')

        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        output_path = os.path.join(app.config['RESULT_FOLDER'], 'enhanced_' + file.filename)
        file.save(input_path)

        # Load img
        img = Image.open(input_path).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)

        start = time.time()
        with torch.no_grad():
            y = model(x)
        end = time.time()

        # Save output image
        tensor_to_img(y).save(output_path)

        # Convert tensors to numpy arrays for metrics calculation
        input_np = np.array(img)  # Already RGB, uint8
        output_np = np.array(tensor_to_img(y))  # Convert tensor to PIL then numpy

        # Calculate all image quality metrics using the new metrics module
        image_metrics = evaluate_all_image_metrics(
            input_img=input_np,
            output_img=output_np,
            output_tensor=y,  # Pass tensor for NIQE calculation
            device=device
        )

        # Calculate performance metrics
        perf_metrics = calculate_performance_metrics(
            start_time=start,
            end_time=end,
            image_shape=y.shape,
            model_path='PyTorch/models'
        )

        # Extract metrics for display
        psnr = round(image_metrics.get('psnr', 0.0), 3)
        ssim = round(image_metrics.get('ssim', 0.0), 3)
        uiqm = round(image_metrics.get('uiqm', 0.0), 3)
        uciqe = round(image_metrics.get('uciqe', 0.0), 3)
        niqe = round(image_metrics.get('niqe', 0.0), 3)

        latency = round(perf_metrics.get('latency', 0.0), 3)
        fps = round(perf_metrics.get('fps', 0.0), 2)
        flops = round(perf_metrics.get('flops_mflops', 0.0), 2)
        model_size = round(perf_metrics.get('model_size_mb', 0.0), 3)
        mem = round(perf_metrics.get('memory_mb', 0.0), 2)
        energy = round(perf_metrics.get('energy_joules', 0.0), 5)
        battery = round(perf_metrics.get('battery_wh', 0.0), 5)

        info = {
            "psnr": psnr, "ssim": ssim, "uiqm": uiqm, "uciqe": uciqe, "niqe": niqe,
            "latency": latency, "fps": fps,
            "flops": flops, "model_size": model_size,
            "mem_used": mem, "energy_joule": energy,
            "battery_wh": battery
        }

        pixel_in = np.array(img)[:3, :5, :].tolist()
        pixel_out = np.array(tensor_to_img(y))[:3, :5, :].tolist()

        return render_template(
            'index.html',
            input_image=f'uploads/{file.filename}',
            output_image=f'results/enhanced_{file.filename}',
            pixel_in=pixel_in,
            pixel_out=pixel_out,
            info=info
        )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
