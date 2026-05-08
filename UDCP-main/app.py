"""
UDCP Web Application
====================
Flask web application for Underwater Dark Channel Prior (UDCP) image enhancement.
Provides a web interface for uploading underwater images and viewing enhancement results.
"""

import os
import time
import numpy as np
import cv2
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

# Import UDCP processing functions
from udcp import process_udcp

# Import metrics functions
from metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_uiqm,
    calculate_uciqe,
    calculate_niqe,
    calculate_performance_metrics
)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload

# Ensure static folder exists
os.makedirs('static', exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}


def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def format_pixel_matrix(img, rows=3, cols=5):
    """
    Format pixel matrix for display in HTML table.
    Returns a 2D list of formatted pixel strings.

    Args:
        img: Input image (numpy array)
        rows: Number of rows to display
        cols: Number of columns to display

    Returns:
        List of lists containing formatted pixel strings
    """
    pixel_matrix = []
    for i in range(min(rows, img.shape[0])):
        row = []
        for j in range(min(cols, img.shape[1])):
            if len(img.shape) == 3:
                # RGB image
                r, g, b = img[i, j, 0], img[i, j, 1], img[i, j, 2]
                pixel_str = f"({r},{g},{b})"
            else:
                # Grayscale image
                pixel_str = f"{img[i, j]}"
            row.append(pixel_str)
        pixel_matrix.append(row)
    return pixel_matrix


@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route for image upload and processing"""

    if request.method == 'POST':
        # Check if file is present
        if 'file' not in request.files:
            return render_template('index.html', error='Không có file được upload')

        file = request.files['file']

        # Check if file is selected
        if file.filename == '':
            return render_template('index.html', error='Không có file được chọn')

        # Check if file is allowed
        if file and allowed_file(file.filename):
            print("\n" + "="*80)
            print("🚀 BẮT ĐẦU XỬ LÝ ẢNH UDCP")
            print("="*80)

            # Save uploaded file
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
            file.save(input_path)

            print(f"✓ Đã lưu file: {input_path}")

            # Read image
            img_bgr = cv2.imread(input_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            print(f"✓ Đã đọc ảnh: shape={img_rgb.shape}, dtype={img_rgb.dtype}")
            print(f"  - Chiều cao (Height): {img_rgb.shape[0]} pixels")
            print(f"  - Chiều rộng (Width): {img_rgb.shape[1]} pixels")
            print(f"  - Số kênh (Channels): {img_rgb.shape[2]}")
            print(f"  - Giá trị pixel range: [{img_rgb.min()}, {img_rgb.max()}]")

            # Format input pixel matrix for display
            print("\n📊 Ma trận pixel đầu vào (3x5 góc trên bên trái):")
            pixel_in = format_pixel_matrix(img_rgb, rows=3, cols=5)
            for i, row in enumerate(pixel_in):
                print(f"  Row {i}: {row}")

            # Start timing
            start_time = time.time()

            # Process with UDCP
            print("\n" + "="*80)
            print("🔄 BẮT ĐẦU XỬ LÝ UDCP")
            print("="*80)

            blockSize = 15
            result = process_udcp(img_rgb, blockSize=blockSize)

            # End timing
            end_time = time.time()

            print("\n" + "="*80)
            print("💾 LƯU KẾT QUẢ")
            print("="*80)

            # Extract results
            output_img = result['output']
            dark_channel = result['dark_channel']
            atmospheric_light = result['atmospheric_light']
            transmission_raw = result['transmission_raw']
            transmission_refined = result['transmission_refined']

            # Save output images
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
            cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
            print(f"✓ Lưu ảnh output: {output_path}")

            # Save dark channel
            dark_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dark_channel.jpg')
            cv2.imwrite(dark_path, dark_channel)
            print(f"✓ Lưu dark channel: {dark_path}")

            # Save transmission maps
            trans_raw_path = os.path.join(app.config['UPLOAD_FOLDER'], 'transmission_raw.jpg')
            cv2.imwrite(trans_raw_path, np.uint8(transmission_raw * 255))
            print(f"✓ Lưu transmission raw: {trans_raw_path}")

            trans_refined_path = os.path.join(app.config['UPLOAD_FOLDER'], 'transmission_refined.jpg')
            cv2.imwrite(trans_refined_path, np.uint8(transmission_refined * 255))
            print(f"✓ Lưu transmission refined: {trans_refined_path}")

            # Create heatmap visualization of transmission
            heatmap = cv2.applyColorMap(np.uint8(transmission_refined * 255), cv2.COLORMAP_JET)
            heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], 'transmission_heatmap.jpg')
            cv2.imwrite(heatmap_path, heatmap)
            print(f"✓ Lưu transmission heatmap: {heatmap_path}")

            # Format output pixel matrix for display
            print("\n📊 Ma trận pixel đầu ra (3x5 góc trên bên trái):")
            pixel_out = format_pixel_matrix(output_img, rows=3, cols=5)
            for i, row in enumerate(pixel_out):
                print(f"  Row {i}: {row}")

            # Calculate metrics
            print("\n" + "="*80)
            print("📈 TÍNH TOÁN METRICS")
            print("="*80)

            try:
                psnr = calculate_psnr(img_rgb, output_img)
                print(f"✓ PSNR: {psnr:.2f} dB")
            except Exception as e:
                print(f"✗ PSNR error: {e}")
                psnr = 0.0

            try:
                ssim = calculate_ssim(img_rgb, output_img)
                print(f"✓ SSIM: {ssim:.4f}")
            except Exception as e:
                print(f"✗ SSIM error: {e}")
                ssim = 0.0

            try:
                uiqm = calculate_uiqm(output_img)
                print(f"✓ UIQM: {uiqm:.4f}")
            except Exception as e:
                print(f"✗ UIQM error: {e}")
                uiqm = 0.0

            try:
                uciqe = calculate_uciqe(output_img)
                print(f"✓ UCIQE: {uciqe:.4f}")
            except Exception as e:
                print(f"✗ UCIQE error: {e}")
                uciqe = 0.0

            try:
                niqe = calculate_niqe(output_img)
                print(f"✓ NIQE: {niqe:.4f}")
            except Exception as e:
                print(f"✗ NIQE error: {e}")
                niqe = 0.0

            # Calculate performance metrics
            try:
                perf_metrics = calculate_performance_metrics(
                    start_time, end_time, img_rgb.shape
                )
                print(f"✓ Latency: {perf_metrics['latency']:.4f} s")
                print(f"✓ FPS: {perf_metrics['fps']:.2f}")
                print(f"✓ FLOPs: {perf_metrics['flops_mflops']:.2f} M")
                print(f"✓ Memory: {perf_metrics['memory_mb']:.2f} MB")
                print(f"✓ Energy: {perf_metrics['energy_joules']:.4f} J")
            except Exception as e:
                print(f"✗ Performance metrics error: {e}")
                perf_metrics = {
                    'latency': 0.0, 'fps': 0.0, 'flops_mflops': 0.0,
                    'memory_mb': 0.0, 'energy_joules': 0.0, 'battery_wh': 0.0
                }

            # Prepare info dictionary for template
            info = {
                'A': f"[{atmospheric_light[0]}, {atmospheric_light[1]}, {atmospheric_light[2]}]",
                'trans_mean': f"{transmission_refined.mean():.4f}",
                'trans_min': f"{transmission_refined.min():.4f}",
                'trans_max': f"{transmission_refined.max():.4f}",
                'psnr': f"{psnr:.2f}",
                'ssim': f"{ssim:.4f}",
                'uiqm': f"{uiqm:.4f}",
                'uciqe': f"{uciqe:.4f}",
                'niqe': f"{niqe:.4f}",
                'latency': f"{perf_metrics['latency']:.4f}",
                'fps': f"{perf_metrics['fps']:.2f}",
                'flops': f"{perf_metrics['flops_mflops']:.2f}",
                'model_size': f"{perf_metrics.get('model_size_mb', 0):.2f}",
                'mem_used': f"{perf_metrics['memory_mb']:.2f}",
                'energy_joule': f"{perf_metrics['energy_joules']:.4f}",
                'battery_wh': f"{perf_metrics['battery_wh']:.6f}"
            }

            print("\n" + "="*80)
            print("✅ HOÀN THÀNH XỬ LÝ")
            print("="*80)
            print(f"Tổng thời gian xử lý: {perf_metrics['latency']:.4f} giây")
            print("="*80 + "\n")

            # Render template with results
            return render_template('index.html',
                                   input_image='input.jpg',
                                   output_image='output.jpg',
                                   dark_image='dark_channel.jpg',
                                   trans_raw_image='transmission_raw.jpg',
                                   trans_refined_image='transmission_refined.jpg',
                                   heatmap_image='transmission_heatmap.jpg',
                                   pixel_in=pixel_in,
                                   pixel_out=pixel_out,
                                   info=info)
        else:
            return render_template('index.html',
                                   error='File không hợp lệ. Chỉ chấp nhận: png, jpg, jpeg, bmp, tiff')

    # GET request - show upload form
    return render_template('index.html')


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🌊 UDCP Web Application - Underwater Image Enhancement")
    print("="*80)
    print("Starting Flask server...")
    print("Open your browser and go to: http://localhost:5002")
    print("="*80 + "\n")

    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
