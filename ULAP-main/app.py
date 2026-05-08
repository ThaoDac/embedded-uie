"""
ULAP Flask Web Application
===========================
Web interface for underwater image enhancement using ULAP algorithm.
"""

import os
import time
import numpy as np
import cv2
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename

# Import ULAP processing function
from ulap import process_ulap

# Import metrics
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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure static directory exists
os.makedirs('static', exist_ok=True)

# Suppress numpy overflow warnings
np.seterr(over='ignore')


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    """
    Handle image upload and ULAP processing
    """
    print("\n" + "=" * 80)
    print("NEW IMAGE UPLOAD AND PROCESSING")
    print("=" * 80)

    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    if file:
        # Save input file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
        file.save(input_path)
        print(f"✓ File saved: {input_path}")

        # Read image
        img_bgr = cv2.imread(input_path)
        if img_bgr is None:
            return 'Error reading image', 400

        print(f"✓ Image loaded: shape={img_bgr.shape}, dtype={img_bgr.dtype}")

        # Convert to RGB for metrics
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Get pixel matrices (3x5 top-left corner) for input
        pixel_in = []
        for i in range(min(3, img_rgb.shape[0])):
            row = []
            for j in range(min(5, img_rgb.shape[1])):
                pixel_val = f"({img_rgb[i,j,0]},{img_rgb[i,j,1]},{img_rgb[i,j,2]})"
                row.append(pixel_val)
            pixel_in.append(row)

        # Start timing
        start_time = time.time()

        # ====================================================================
        # ULAP Processing
        # ====================================================================
        result = process_ulap(img_bgr, blockSize=9, gimfiltR=50, eps=0.001)

        # End timing
        end_time = time.time()

        # Extract results
        output_bgr = result['output']
        depth_map = result['depth_map']
        depth_map_refined = result['depth_map_refined']
        background_light = result['background_light']
        min_depth = result['min_depth']
        transmissionB, transmissionG, transmissionR = result['transmission_raw']
        transmission_refined = result['transmission_refined']

        # Convert output to RGB for display
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

        # Get pixel matrices for output
        pixel_out = []
        for i in range(min(3, output_rgb.shape[0])):
            row = []
            for j in range(min(5, output_rgb.shape[1])):
                pixel_val = f"({output_rgb[i,j,0]},{output_rgb[i,j,1]},{output_rgb[i,j,2]})"
                row.append(pixel_val)
            pixel_out.append(row)

        # ====================================================================
        # Save intermediate results
        # ====================================================================
        print("\n" + "-" * 80)
        print("SAVING INTERMEDIATE RESULTS")
        print("-" * 80)

        # Save output image
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
        cv2.imwrite(output_path, output_bgr)
        print(f"✓ Output image saved: {output_path}")

        # Save depth map
        depth_map_path = os.path.join(app.config['UPLOAD_FOLDER'], 'depth_map.jpg')
        cv2.imwrite(depth_map_path, np.uint8(depth_map_refined * 255))
        print(f"✓ Depth map saved: {depth_map_path}")

        # Save transmission map (using red channel for visualization)
        trans_map_path = os.path.join(app.config['UPLOAD_FOLDER'], 'transmission_map.jpg')
        cv2.imwrite(trans_map_path, np.uint8(transmission_refined[:, :, 2] * 255))
        print(f"✓ Transmission map saved: {trans_map_path}")

        # Create heatmap for transmission
        trans_heatmap = cv2.applyColorMap(
            np.uint8(transmission_refined[:, :, 2] * 255),
            cv2.COLORMAP_JET
        )
        trans_heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], 'transmission_heatmap.jpg')
        cv2.imwrite(trans_heatmap_path, trans_heatmap)
        print(f"✓ Transmission heatmap saved: {trans_heatmap_path}")

        # ====================================================================
        # Calculate Metrics
        # ====================================================================
        print("\n" + "-" * 80)
        print("CALCULATING QUALITY METRICS")
        print("-" * 80)

        # Image quality metrics
        try:
            psnr = calculate_psnr(img_rgb, output_rgb)
            print(f"✓ PSNR: {psnr:.2f} dB")
        except Exception as e:
            print(f"✗ PSNR calculation failed: {e}")
            psnr = 0.0

        try:
            ssim = calculate_ssim(img_rgb, output_rgb)
            print(f"✓ SSIM: {ssim:.4f}")
        except Exception as e:
            print(f"✗ SSIM calculation failed: {e}")
            ssim = 0.0

        try:
            uiqm = calculate_uiqm(output_rgb)
            print(f"✓ UIQM: {uiqm:.4f}")
        except Exception as e:
            print(f"✗ UIQM calculation failed: {e}")
            uiqm = 0.0

        try:
            uciqe = calculate_uciqe(output_rgb)
            print(f"✓ UCIQE: {uciqe:.4f}")
        except Exception as e:
            print(f"✗ UCIQE calculation failed: {e}")
            uciqe = 0.0

        try:
            niqe = calculate_niqe(output_rgb)
            print(f"✓ NIQE: {niqe:.4f}")
        except Exception as e:
            print(f"✗ NIQE calculation failed: {e}")
            niqe = 0.0

        # Performance metrics
        print("\n" + "-" * 80)
        print("CALCULATING PERFORMANCE METRICS")
        print("-" * 80)

        perf_metrics = calculate_performance_metrics(
            start_time,
            end_time,
            img_rgb.shape,
            model_path=''  # ULAP is traditional algorithm, no model file
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

        # ====================================================================
        # Prepare info for template
        # ====================================================================
        info = {
            # Background light
            'A': f"[B={background_light[0]:.2f}, G={background_light[1]:.2f}, R={background_light[2]:.2f}]",
            'min_depth': f"{min_depth:.4f}",

            # Transmission statistics
            'trans_mean': f"{transmission_refined.mean():.4f}",
            'trans_min': f"{transmission_refined.min():.4f}",
            'trans_max': f"{transmission_refined.max():.4f}",

            # Image quality metrics
            'psnr': f"{psnr:.2f}",
            'ssim': f"{ssim:.4f}",
            'uiqm': f"{uiqm:.4f}",
            'uciqe': f"{uciqe:.4f}",
            'niqe': f"{niqe:.4f}",

            # Performance metrics
            'latency': f"{latency:.4f}",
            'fps': f"{fps:.2f}",
            'flops': f"{flops:.2f}",
            'mem_used': f"{mem_used:.2f}",
            'model_size': "0",  # No model file for traditional algorithm
            'energy_joule': f"{energy_joule:.4f}",
            'battery_wh': f"{battery_wh:.6f}"
        }

        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80 + "\n")

        # Render template with results
        return render_template(
            'index.html',
            input_image='input.jpg',
            output_image='output.jpg',
            depth_image='depth_map.jpg',
            trans_image='transmission_map.jpg',
            trans_heatmap='transmission_heatmap.jpg',
            pixel_in=pixel_in,
            pixel_out=pixel_out,
            info=info
        )


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🌊 ULAP - Underwater Light Attenuation Prior")
    print("=" * 80)
    print("Starting Flask server...")
    print("Access the application at: http://localhost:5001")
    print("=" * 80 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)