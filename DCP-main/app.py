"""
Flask Web Application for DCP Image Dehazing
=============================================
This application provides a web interface for removing haze from images
using the Dark Channel Prior (DCP) algorithm.
"""

import os
import time
from flask import Flask, render_template, request, url_for
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Import DCP processor
from dcp import DCPProcessor

# Import metrics
from metrics import (
    evaluate_all_image_metrics,
    calculate_performance_metrics
)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_folders():
    """Create necessary folders if they don't exist."""
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    os.makedirs('templates', exist_ok=True)


def get_pixel_matrix(img, rows=3, cols=5):
    """
    Extract pixel matrix from top-left corner of image.

    Args:
        img: Input image (H, W, 3) in BGR format
        rows: Number of rows to extract (default: 3)
        cols: Number of columns to extract (default: 5)

    Returns:
        list: List of lists containing pixel values as [R, G, B]
    """
    pixel_matrix = []
    for i in range(min(rows, img.shape[0])):
        row = []
        for j in range(min(cols, img.shape[1])):
            # Convert BGR to RGB for display
            b, g, r = img[i, j]
            row.append(f"[{r},{g},{b}]")
        pixel_matrix.append(row)
    return pixel_matrix


def create_transmission_heatmap(transmission_map):
    """
    Create colorful heatmap visualization of transmission map.

    Args:
        transmission_map: Transmission map (H, W) with values in [0, 1]

    Returns:
        heatmap: Colorized transmission map (H, W, 3) in BGR format
    """
    # Normalize to 0-255 if needed
    if transmission_map.max() <= 1.0:
        trans_normalized = (transmission_map * 255).astype(np.uint8)
    else:
        trans_normalized = transmission_map.astype(np.uint8)

    # Apply JET colormap (Red = high transmission, Blue = low transmission)
    heatmap = cv2.applyColorMap(trans_normalized, cv2.COLORMAP_JET)

    return heatmap


@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route for uploading and processing images."""
    if request.method == 'GET':
        # Show upload form
        return render_template('index.html')

    # POST request - process uploaded image
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']

    if file.filename == '':
        return "No file selected", 400

    if not allowed_file(file.filename):
        return "Invalid file type. Please upload an image file.", 400

    # Ensure folders exist
    ensure_folders()

    # Save uploaded file
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
    file.save(input_path)

    print("\n" + "="*80)
    print("NEW IMAGE PROCESSING REQUEST")
    print("="*80)
    print(f"Uploaded file: {filename}")

    # Load image
    img = cv2.imread(input_path)
    if img is None:
        return "Failed to load image", 500

    print(f"Image loaded: {img.shape}")

    # Record start time for performance metrics
    start_time = time.time()

    # Process with DCP
    processor = DCPProcessor(
        patch_size=15,
        omega=0.95,
        t0=0.1,
        guided_r=60,
        guided_eps=0.0001
    )

    results = processor.process(img)

    # Record end time
    end_time = time.time()

    # Extract results
    dark_channel = results['dark_channel']  # uint8 [0, 255]
    transmission_raw = results['transmission_raw']  # uint8 [0, 255]
    transmission_refined = results['transmission_refined']  # uint8 [0, 255]
    transmission_float = results['transmission_float']  # float [0, 1]
    atmospheric_light = results['atmospheric_light']  # shape (1, 3), uint8 [0, 255]
    recovered_image = results['recovered_image']  # uint8 [0, 255]
    trans_stats = results['trans_stats']

    # Save intermediate images
    print("\n[Saving Results]")

    # 1. Dark channel
    dark_path = os.path.join(app.config['RESULTS_FOLDER'], 'dark_channel.png')
    cv2.imwrite(dark_path, dark_channel)
    print(f"✓ Dark channel saved: {dark_path}")

    # 2. Transmission raw (before refinement)
    trans_raw_path = os.path.join(app.config['RESULTS_FOLDER'], 'transmission_raw.png')
    cv2.imwrite(trans_raw_path, transmission_raw)
    print(f"✓ Transmission (raw) saved: {trans_raw_path}")

    # 3. Transmission refined (grayscale)
    trans_refined_path = os.path.join(app.config['RESULTS_FOLDER'], 'transmission_refined.png')
    cv2.imwrite(trans_refined_path, transmission_refined)
    print(f"✓ Transmission (refined) saved: {trans_refined_path}")

    # 4. Transmission heatmap (colorized)
    trans_heatmap_path = os.path.join(app.config['RESULTS_FOLDER'], 'transmission_heatmap.png')
    trans_heatmap = create_transmission_heatmap(transmission_float)
    cv2.imwrite(trans_heatmap_path, trans_heatmap)
    print(f"✓ Transmission (heatmap) saved: {trans_heatmap_path}")

    # 5. Recovered image
    recovered_path = os.path.join(app.config['RESULTS_FOLDER'], 'recovered.png')
    cv2.imwrite(recovered_path, recovered_image)
    print(f"✓ Recovered image saved: {recovered_path}")

    # Calculate image quality metrics
    print("\n[Calculating Image Quality Metrics]")

    # Convert BGR to RGB for metrics calculation
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    recovered_rgb = cv2.cvtColor(recovered_image, cv2.COLOR_BGR2RGB)

    try:
        # Pass recovered_rgb as output_tensor (numpy array) for NIQE calculation
        img_metrics = evaluate_all_image_metrics(img_rgb, recovered_rgb, output_tensor=recovered_rgb)
        print(f"✓ PSNR: {img_metrics.get('psnr', -1):.2f} dB")
        print(f"✓ SSIM: {img_metrics.get('ssim', -1):.4f}")
        print(f"✓ UIQM: {img_metrics.get('uiqm', -1):.4f}")
        print(f"✓ UCIQE: {img_metrics.get('uciqe', -1):.4f}")
        print(f"✓ NIQE: {img_metrics.get('niqe', -1):.4f}")
    except Exception as e:
        print(f"Error calculating image metrics: {e}")
        img_metrics = {
            'psnr': -1.0,
            'ssim': -1.0,
            'uiqm': -1.0,
            'uciqe': -1.0,
            'niqe': -1.0
        }

    # Calculate performance metrics
    print("\n[Calculating Performance Metrics]")
    try:
        perf_metrics = calculate_performance_metrics(
            start_time,
            end_time,
            img.shape,
            model_path='dcp.py'  # Using script itself as "model"
        )
        print(f"✓ Latency: {perf_metrics.get('latency', 0):.4f} s")
        print(f"✓ FPS: {perf_metrics.get('fps', 0):.2f}")
        print(f"✓ FLOPs: {perf_metrics.get('flops_mflops', 0):.2f} M")
        print(f"✓ Memory: {perf_metrics.get('memory_mb', 0):.2f} MB")
        print(f"✓ Energy: {perf_metrics.get('energy_joules', 0):.4f} J")
        print(f"✓ Battery: {perf_metrics.get('battery_wh', 0):.6f} Wh")
    except Exception as e:
        print(f"Error calculating performance metrics: {e}")
        perf_metrics = {
            'latency': end_time - start_time,
            'fps': 0.0,
            'flops_mflops': 0.0,
            'memory_mb': 0.0,
            'model_size_mb': 0.0,
            'energy_joules': 0.0,
            'battery_wh': 0.0
        }

    # Extract pixel matrices (3x5)
    print("\n[Extracting Pixel Matrices]")
    pixel_in = get_pixel_matrix(img, rows=3, cols=5)
    pixel_out = get_pixel_matrix(recovered_image, rows=3, cols=5)
    print(f"✓ Input pixel matrix (3x5) extracted")
    print(f"✓ Output pixel matrix (3x5) extracted")

    # Prepare data for template
    # Atmospheric light is shape (1, 3) in BGR format, convert to RGB for display
    A_flat = atmospheric_light.flatten()
    info = {
        # Atmospheric Light (convert BGR to RGB for display)
        'A': f"[R={A_flat[2]:.2f}, G={A_flat[1]:.2f}, B={A_flat[0]:.2f}]",

        # Transmission stats
        'trans_min': f"{trans_stats['min']:.4f}",
        'trans_max': f"{trans_stats['max']:.4f}",
        'trans_mean': f"{trans_stats['mean']:.4f}",

        # Image quality metrics
        'psnr': f"{img_metrics.get('psnr', -1):.2f}",
        'ssim': f"{img_metrics.get('ssim', -1):.4f}",
        'uiqm': f"{img_metrics.get('uiqm', -1):.4f}",
        'uciqe': f"{img_metrics.get('uciqe', -1):.4f}",
        'niqe': f"{img_metrics.get('niqe', -1):.4f}",

        # Performance metrics
        'latency': f"{perf_metrics.get('latency', 0):.4f}",
        'fps': f"{perf_metrics.get('fps', 0):.2f}",
        'flops': f"{perf_metrics.get('flops_mflops', 0):.2f}",
        'model_size': f"{perf_metrics.get('model_size_mb', 0):.2f}",
        'mem_used': f"{perf_metrics.get('memory_mb', 0):.2f}",

        # Energy metrics
        'energy_joule': f"{perf_metrics.get('energy_joules', 0):.4f}",
        'battery_wh': f"{perf_metrics.get('battery_wh', 0):.6f}"
    }

    print("\n" + "="*80)
    print("PROCESSING COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")

    # Render template with results
    return render_template(
        'index.html',
        input_image='uploads/input.jpg',
        dark_image='results/dark_channel.png',
        trans_raw_image='results/transmission_raw.png',
        trans_refined_image='results/transmission_refined.png',
        heatmap_image='results/transmission_heatmap.png',
        output_image='results/recovered.png',
        pixel_in=pixel_in,
        pixel_out=pixel_out,
        info=info
    )


if __name__ == '__main__':
    print("\n" + "="*80)
    print("DCP Image Dehazing - Flask Web Application")
    print("="*80)
    print("Starting server...")
    print("Access the application at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*80 + "\n")

    ensure_folders()
    app.run(debug=True, host='0.0.0.0', port=5001)