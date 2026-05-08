"""
CLAHE Flask Web Application
============================
Web interface for underwater image enhancement using CLAHE.
"""

import os
import time
import numpy as np
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from clahe import process_clahe, compare_histograms
from metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_uiqm,
    calculate_uciqe,
    calculate_niqe,
    calculate_performance_metrics
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs('static', exist_ok=True)
np.seterr(over='ignore')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    print("\n" + "=" * 80)
    print("NEW IMAGE UPLOAD AND PROCESSING")
    print("=" * 80)

    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    # Get CLAHE parameters
    try:
        clip_limit = float(request.form.get('clipLimit', 2.0))
        tile_size = int(request.form.get('tileSize', 4))
    except:
        clip_limit = 2.0
        tile_size = 4

    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
        file.save(input_path)
        print(f"✓ File saved: {input_path}")

        img_bgr = cv2.imread(input_path)
        if img_bgr is None:
            return 'Error reading image', 400

        print(f"✓ Image loaded: shape={img_bgr.shape}, dtype={img_bgr.dtype}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Get pixel matrices (3x5)
        pixel_in = []
        for i in range(min(3, img_rgb.shape[0])):
            row = []
            for j in range(min(5, img_rgb.shape[1])):
                pixel_val = f"({img_rgb[i,j,0]},{img_rgb[i,j,1]},{img_rgb[i,j,2]})"
                row.append(pixel_val)
            pixel_in.append(row)

        start_time = time.time()

        # CLAHE Processing
        result = process_clahe(img_bgr, clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

        end_time = time.time()

        output_bgr = result['output']
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

        # Get output pixel matrices
        pixel_out = []
        for i in range(min(3, output_rgb.shape[0])):
            row = []
            for j in range(min(5, output_rgb.shape[1])):
                pixel_val = f"({output_rgb[i,j,0]},{output_rgb[i,j,1]},{output_rgb[i,j,2]})"
                row.append(pixel_val)
            pixel_out.append(row)

        # Histogram statistics
        print("\n" + "-" * 80)
        print("COMPUTING HISTOGRAM STATISTICS")
        print("-" * 80)

        hist_stats = compare_histograms(img_bgr, output_bgr)

        for channel, stats in hist_stats.items():
            print(f"\n{channel} Channel:")
            print(f"  Before: mean={stats['mean_before']:.2f}, std={stats['std_before']:.2f}")
            print(f"  After:  mean={stats['mean_after']:.2f}, std={stats['std_after']:.2f}")

        # Save results
        print("\n" + "-" * 80)
        print("SAVING RESULTS")
        print("-" * 80)

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
        cv2.imwrite(output_path, output_bgr)
        print(f"✓ Output image saved: {output_path}")

        # Calculate Metrics
        print("\n" + "-" * 80)
        print("CALCULATING QUALITY METRICS")
        print("-" * 80)

        try:
            psnr = calculate_psnr(img_rgb, output_rgb)
            print(f"✓ PSNR: {psnr:.2f} dB")
        except Exception as e:
            print(f"✗ PSNR failed: {e}")
            psnr = 0.0

        try:
            ssim = calculate_ssim(img_rgb, output_rgb)
            print(f"✓ SSIM: {ssim:.4f}")
        except Exception as e:
            print(f"✗ SSIM failed: {e}")
            ssim = 0.0

        try:
            uiqm = calculate_uiqm(output_rgb)
            print(f"✓ UIQM: {uiqm:.4f}")
        except Exception as e:
            print(f"✗ UIQM failed: {e}")
            uiqm = 0.0

        try:
            uciqe = calculate_uciqe(output_rgb)
            print(f"✓ UCIQE: {uciqe:.4f}")
        except Exception as e:
            print(f"✗ UCIQE failed: {e}")
            uciqe = 0.0

        try:
            niqe = calculate_niqe(output_rgb)
            print(f"✓ NIQE: {niqe:.4f}")
        except Exception as e:
            print(f"✗ NIQE failed: {e}")
            niqe = 0.0

        # Performance metrics
        print("\n" + "-" * 80)
        print("CALCULATING PERFORMANCE METRICS")
        print("-" * 80)

        perf_metrics = calculate_performance_metrics(
            start_time, end_time, img_rgb.shape, model_path=''
        )

        latency = perf_metrics['latency']
        fps = perf_metrics['fps']
        flops = perf_metrics['flops_mflops']
        mem_used = perf_metrics['memory_mb']
        energy_joule = perf_metrics['energy_joules']
        battery_wh = perf_metrics['battery_wh']

        print(f"✓ Latency: {latency:.4f} s")
        print(f"✓ FPS: {fps:.2f}")
        print(f"✓ FLOPs: {flops:.2f} MFLOPs")
        print(f"✓ Memory: {mem_used:.2f} MB")
        print(f"✓ Energy: {energy_joule:.4f} J")
        print(f"✓ Battery: {battery_wh:.6f} Wh")

        # Prepare info for template
        info = {
            'clip_limit': f"{clip_limit}",
            'tile_size': f"{tile_size}x{tile_size}",
            'blue_mean_before': f"{hist_stats['Blue']['mean_before']:.2f}",
            'blue_mean_after': f"{hist_stats['Blue']['mean_after']:.2f}",
            'green_mean_before': f"{hist_stats['Green']['mean_before']:.2f}",
            'green_mean_after': f"{hist_stats['Green']['mean_after']:.2f}",
            'red_mean_before': f"{hist_stats['Red']['mean_before']:.2f}",
            'red_mean_after': f"{hist_stats['Red']['mean_after']:.2f}",
            'psnr': f"{psnr:.2f}",
            'ssim': f"{ssim:.4f}",
            'uiqm': f"{uiqm:.4f}",
            'uciqe': f"{uciqe:.4f}",
            'niqe': f"{niqe:.4f}",
            'latency': f"{latency:.4f}",
            'fps': f"{fps:.2f}",
            'flops': f"{flops:.2f}",
            'mem_used': f"{mem_used:.2f}",
            'model_size': "0",
            'energy_joule': f"{energy_joule:.4f}",
            'battery_wh': f"{battery_wh:.6f}"
        }

        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80 + "\n")

        return render_template(
            'index.html',
            input_image='input.jpg',
            output_image='output.jpg',
            pixel_in=pixel_in,
            pixel_out=pixel_out,
            info=info
        )


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("CLAHE - Contrast Limited Adaptive Histogram Equalization")
    print("=" * 80)
    print("Starting Flask server...")
    print("Access: http://localhost:5000")
    print("=" * 80 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)