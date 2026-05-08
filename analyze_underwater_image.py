#!/usr/bin/env python3
"""
Underwater Image Quality Analysis Script

This script analyzes a single underwater image (raw or enhanced) and creates
a composite visualization showing:
- Original image
- RGB histogram
- HSV saturation histogram
- Laplacian sharpness analysis
- Canny edge detection map

Usage:
    python analyze_underwater_image.py <input_image_path> [output_path] [options]

Example:
    python analyze_underwater_image.py input.jpg analysis_output.png
    python analyze_underwater_image.py input.jpg --plot-type line
    python analyze_underwater_image.py input.jpg --plot-type bar
    python analyze_underwater_image.py input.jpg --plot-type both
    python analyze_underwater_image.py input.jpg --plots rgb hsv
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import os
import sys


def load_image(image_path):
    """Load image from path and convert to RGB."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert BGR to RGB for proper display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb


def compute_rgb_histogram(img_rgb):
    """Compute RGB histogram."""
    colors = ('r', 'g', 'b')
    histograms = []

    for i, color in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        histograms.append(hist)

    return histograms, colors


def compute_hsv_saturation(img_bgr):
    """Compute HSV saturation histogram."""
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation = img_hsv[:, :, 1]
    hist = cv2.calcHist([saturation], [0], None, [256], [0, 256])
    return hist, saturation


def compute_laplacian_sharpness(img_bgr):
    """Compute Laplacian for sharpness analysis."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Compute sharpness metric (variance of Laplacian)
    sharpness_score = laplacian.var()

    # Normalize for visualization
    laplacian_abs = np.abs(laplacian)
    laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return laplacian_norm, sharpness_score


def compute_canny_edges(img_bgr, low_threshold=50, high_threshold=150):
    """Compute Canny edge detection."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # Compute edge density (percentage of edge pixels)
    edge_density = (np.sum(edges > 0) / edges.size) * 100

    return edges, edge_density


def plot_rgb_histogram(ax, rgb_hists, colors, plot_type='line'):
    """Plot RGB histogram with specified type."""
    if plot_type == 'line':
        for i, (hist, color) in enumerate(zip(rgb_hists, colors)):
            ax.plot(hist, color=color, alpha=0.7, linewidth=2, label=color.upper())
    elif plot_type == 'bar':
        # Downsample for bar chart to avoid overcrowding
        x = np.arange(0, 256, 4)  # Every 4th bin
        width = 3
        offset = [-width, 0, width]
        for i, (hist, color) in enumerate(zip(rgb_hists, colors)):
            downsampled = hist[::4]
            ax.bar(x + offset[i], downsampled.flatten(), width=width,
                   color=color, alpha=0.6, label=color.upper())
    elif plot_type == 'both':
        # Bar chart
        x = np.arange(0, 256, 4)
        width = 3
        offset = [-width, 0, width]
        for i, (hist, color) in enumerate(zip(rgb_hists, colors)):
            downsampled = hist[::4]
            ax.bar(x + offset[i], downsampled.flatten(), width=width,
                   color=color, alpha=0.3)
        # Line chart on top
        for i, (hist, color) in enumerate(zip(rgb_hists, colors)):
            ax.plot(hist, color=color, alpha=0.9, linewidth=2, label=color.upper())

    ax.set_xlim([0, 256])
    ax.set_xlabel('Pixel Intensity', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    # ax.set_title(f'RGB Histogram ({plot_type.title()})', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def render_plot_to_array(fig):
    """Render matplotlib figure to numpy array."""
    # Draw the canvas
    fig.canvas.draw()

    # Convert canvas to RGB array
    # Use the newer buffer_rgba() method instead of deprecated tostring_rgb()
    w, h = fig.canvas.get_width_height()

    try:
        # Try newer method first (matplotlib >= 3.0)
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape((h, w, 4))  # RGBA format
        # Convert RGBA to RGB
        buf = buf[:, :, :3]  # Drop alpha channel
    except AttributeError:
        # Fallback to older method
        try:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape((h, w, 3))
        except AttributeError:
            # Last resort: use renderer
            buf = np.array(fig.canvas.renderer.buffer_rgba())
            buf = buf[:, :, :3]  # Convert RGBA to RGB

    return buf


def create_composite_visualization(img_rgb, img_bgr, output_path, plot_type='line', selected_plots=None):
    """Create composite visualization with all analyses.

    Args:
        img_rgb: RGB image array
        img_bgr: BGR image array
        output_path: Path to save output
        plot_type: Type of plot for histograms ('line', 'bar', 'both')
        selected_plots: List of plots to include (None = all plots)
    """
    # Default to all plots if none selected
    if selected_plots is None:
        selected_plots = ['image', 'rgb', 'hsv', 'laplacian', 'edge']

    # Compute all metrics
    rgb_hists, colors = compute_rgb_histogram(img_rgb)
    sat_hist, saturation = compute_hsv_saturation(img_bgr)
    laplacian, sharpness_score = compute_laplacian_sharpness(img_bgr)
    edges, edge_density = compute_canny_edges(img_bgr)

    # Check for special case: use vertical stacking for image + plots
    has_image = 'image' in selected_plots
    other_plots = [p for p in selected_plots if p != 'image']

    if has_image and len(other_plots) >= 1:
        # Create vertical stacking: image at top, plots below
        print(f"Creating vertical composite with {len(other_plots)} plot(s)...")

        # Use original image width as target (keep original size)
        img_width = img_rgb.shape[1]
        img_height = img_rgb.shape[0]

        # Calculate figure size to match image width (make it square)
        # Assuming 100 DPI, convert pixels to inches
        dpi = 100
        fig_width_inches = img_width / dpi
        fig_height_inches = img_height / dpi  # Make plot square (same as image dimensions)

        # Prepare all plot images
        plot_arrays = []

        for plot_name in other_plots:
            # Create individual plot with same dimensions as original image (square)
            fig_plot = plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=dpi)
            ax = fig_plot.add_subplot(111)

            if plot_name == 'rgb':
                plot_rgb_histogram(ax, rgb_hists, colors, plot_type)
            elif plot_name == 'hsv':
                if plot_type == 'line' or plot_type == 'both':
                    ax.plot(sat_hist, color='purple', linewidth=2)
                    ax.fill_between(range(256), sat_hist.flatten(), alpha=0.3, color='purple')
                else:  # bar
                    x = np.arange(0, 256, 4)
                    downsampled = sat_hist[::4]
                    ax.bar(x, downsampled.flatten(), width=3, color='purple', alpha=0.6)
                ax.set_xlim([0, 256])
                ax.set_xlabel('Saturation Level', fontsize=9)
                ax.set_ylabel('Frequency', fontsize=9)
                mean_sat = np.mean(saturation)
                ax.set_title(f'HSV Saturation ({plot_type.title()}) (Mean: {mean_sat:.2f})',
                             fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
            elif plot_name == 'laplacian':
                im_lap = ax.imshow(laplacian, cmap='hot', aspect='auto')
                ax.set_title(f'Laplacian Sharpness (Score: {sharpness_score:.2f})',
                             fontsize=11, fontweight='bold')
                ax.axis('off')
                plt.colorbar(im_lap, ax=ax, fraction=0.046, pad=0.04)
            elif plot_name == 'edge':
                ax.imshow(edges, cmap='gray', aspect='auto')
                ax.set_title(f'Canny Edge Map (Density: {edge_density:.2f}%)',
                             fontsize=11, fontweight='bold')
                ax.axis('off')

            plt.tight_layout()

            # Render plot to array
            plot_array = render_plot_to_array(fig_plot)
            plt.close(fig_plot)

            # Resize plot to match image width exactly
            if plot_array.shape[1] != img_width:
                aspect_ratio = plot_array.shape[0] / plot_array.shape[1]
                new_height = int(img_width * aspect_ratio)
                plot_resized = cv2.resize(plot_array, (img_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            else:
                plot_resized = plot_array.copy()

            plot_arrays.append(plot_resized)

        # Start with the original image (no resizing)
        combined_parts = [img_rgb.copy()]

        # Add separator
        separator = np.ones((10, img_width, 3), dtype=np.uint8) * 255
        combined_parts.append(separator)

        # Add each plot
        for plot_array in plot_arrays:
            combined_parts.append(plot_array)
            # Add separator between plots
            combined_parts.append(separator)

        # Remove last separator
        combined_parts = combined_parts[:-1]

        # Stack all parts vertically
        combined = np.vstack(combined_parts)

        # Save directly using OpenCV (preserves quality)
        cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"✓ Analysis saved to: {output_path}")
        print(f"  Original image size: {img_width}x{img_height}")
        print(f"  Combined output size: {combined.shape[1]}x{combined.shape[0]}")

        # Print metrics summary
        print("\n" + "="*50)
        print("QUALITY METRICS SUMMARY")
        print("="*50)
        print(f"Resolution: {img_rgb.shape[1]} x {img_rgb.shape[0]}")
        print(f"Mean Saturation: {np.mean(saturation):.2f}")
        print(f"Sharpness Score (Laplacian Variance): {sharpness_score:.2f}")
        print(f"Edge Density: {edge_density:.2f}%")

        # RGB channel statistics
        print("\nRGB Channel Statistics:")
        for i, color in enumerate(['Red', 'Green', 'Blue']):
            channel_mean = np.mean(img_rgb[:, :, i])
            channel_std = np.std(img_rgb[:, :, i])
            print(f"  {color:5s}: Mean={channel_mean:6.2f}, Std={channel_std:6.2f}")

        print("="*50 + "\n")

        return

    # Calculate layout based on selected plots
    num_plots = len(selected_plots)

    # If only image is selected, show just the image
    if selected_plots == ['image']:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_rgb)
        ax.set_title(f'Original Image - Resolution: {img_rgb.shape[1]}x{img_rgb.shape[0]}',
                     fontsize=14, fontweight='bold')
        ax.axis('off')
    else:
        # Determine grid layout
        has_image = 'image' in selected_plots
        other_plots = [p for p in selected_plots if p != 'image']
        num_other = len(other_plots)

        if has_image and num_other > 0:
            # Image on top, other plots below
            rows = 1 + (num_other + 1) // 2
            fig = plt.figure(figsize=(16, 4 * rows + 2))
            gs = gridspec.GridSpec(rows, 2, height_ratios=[1.5] + [1] * (rows - 1),
                                   hspace=0.3, wspace=0.3)

            # Top: Original image
            ax_img = plt.subplot(gs[0, :])
            ax_img.imshow(img_rgb)
            ax_img.set_title(f'Original Image - Resolution: {img_rgb.shape[1]}x{img_rgb.shape[0]}',
                             fontsize=14, fontweight='bold')
            ax_img.axis('off')

            # Other plots
            plot_idx = 0
            for i, plot_name in enumerate(other_plots):
                row = 1 + plot_idx // 2
                col = plot_idx % 2
                ax = plt.subplot(gs[row, col])

                if plot_name == 'rgb':
                    plot_rgb_histogram(ax, rgb_hists, colors, plot_type)
                elif plot_name == 'hsv':
                    if plot_type == 'line' or plot_type == 'both':
                        ax.plot(sat_hist, color='purple', linewidth=2)
                        ax.fill_between(range(256), sat_hist.flatten(), alpha=0.3, color='purple')
                    else:  # bar
                        x = np.arange(0, 256, 4)
                        downsampled = sat_hist[::4]
                        ax.bar(x, downsampled.flatten(), width=3, color='purple', alpha=0.6)
                    ax.set_xlim([0, 256])
                    ax.set_xlabel('Saturation Level', fontsize=10)
                    ax.set_ylabel('Frequency', fontsize=10)
                    mean_sat = np.mean(saturation)
                    ax.set_title(f'HSV Saturation ({plot_type.title()}) (Mean: {mean_sat:.2f})',
                                 fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                elif plot_name == 'laplacian':
                    im_lap = ax.imshow(laplacian, cmap='hot', aspect='auto')
                    ax.set_title(f'Laplacian Sharpness (Score: {sharpness_score:.2f})',
                                 fontsize=12, fontweight='bold')
                    ax.axis('off')
                    plt.colorbar(im_lap, ax=ax, fraction=0.046, pad=0.04)
                elif plot_name == 'edge':
                    ax.imshow(edges, cmap='gray', aspect='auto')
                    ax.set_title(f'Canny Edge Map (Density: {edge_density:.2f}%)',
                                 fontsize=12, fontweight='bold')
                    ax.axis('off')

                plot_idx += 1
        else:
            # No image, just plots in grid
            cols = 2 if num_other > 1 else 1
            rows = (num_other + 1) // 2
            fig = plt.figure(figsize=(8 * cols, 6 * rows))

            for i, plot_name in enumerate(other_plots):
                ax = plt.subplot(rows, cols, i + 1)

                if plot_name == 'rgb':
                    plot_rgb_histogram(ax, rgb_hists, colors, plot_type)
                elif plot_name == 'hsv':
                    if plot_type == 'line' or plot_type == 'both':
                        ax.plot(sat_hist, color='purple', linewidth=2)
                        ax.fill_between(range(256), sat_hist.flatten(), alpha=0.3, color='purple')
                    else:
                        x = np.arange(0, 256, 4)
                        downsampled = sat_hist[::4]
                        ax.bar(x, downsampled.flatten(), width=3, color='purple', alpha=0.6)
                    ax.set_xlim([0, 256])
                    ax.set_xlabel('Saturation Level', fontsize=10)
                    ax.set_ylabel('Frequency', fontsize=10)
                    mean_sat = np.mean(saturation)
                    ax.set_title(f'HSV Saturation ({plot_type.title()}) (Mean: {mean_sat:.2f})',
                                 fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                elif plot_name == 'laplacian':
                    im_lap = ax.imshow(laplacian, cmap='hot', aspect='auto')
                    ax.set_title(f'Laplacian Sharpness (Score: {sharpness_score:.2f})',
                                 fontsize=12, fontweight='bold')
                    ax.axis('off')
                    plt.colorbar(im_lap, ax=ax, fraction=0.046, pad=0.04)
                elif plot_name == 'edge':
                    ax.imshow(edges, cmap='gray', aspect='auto')
                    ax.set_title(f'Canny Edge Map (Density: {edge_density:.2f}%)',
                                 fontsize=12, fontweight='bold')
                    ax.axis('off')

    # Add overall title
    plot_desc = f" ({plot_type.title()} plots)" if any(p in selected_plots for p in ['rgb', 'hsv']) else ""
    fig.suptitle(f'Underwater Image Quality Analysis{plot_desc}',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Analysis saved to: {output_path}")

    # Print metrics summary
    print("\n" + "="*50)
    print("QUALITY METRICS SUMMARY")
    print("="*50)
    print(f"Resolution: {img_rgb.shape[1]} x {img_rgb.shape[0]}")
    print(f"Mean Saturation: {mean_sat:.2f}")
    print(f"Sharpness Score (Laplacian Variance): {sharpness_score:.2f}")
    print(f"Edge Density: {edge_density:.2f}%")

    # RGB channel statistics
    print("\nRGB Channel Statistics:")
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        channel_mean = np.mean(img_rgb[:, :, i])
        channel_std = np.std(img_rgb[:, :, i])
        print(f"  {color:5s}: Mean={channel_mean:6.2f}, Std={channel_std:6.2f}")

    print("="*50 + "\n")

    plt.close()


def process_single_image(input_path, output_path, plot_type, selected_plots):
    """Process a single image."""
    try:
        print(f"\n{'='*60}")
        print(f"Analyzing: {input_path}")
        print(f"Plot type: {plot_type}")
        if selected_plots:
            print(f"Selected plots: {', '.join(selected_plots)}")
        else:
            print(f"Selected plots: all (default)")
        print(f"Output: {output_path}")
        print('='*60)

        # Load image
        img_bgr, img_rgb = load_image(input_path)

        # Create composite visualization
        create_composite_visualization(img_rgb, img_bgr, output_path,
                                       plot_type=plot_type,
                                       selected_plots=selected_plots)

        print(f"✓ Completed successfully!\n")
        return True

    except Exception as e:
        print(f"✗ Error processing {input_path}: {e}\n", file=sys.stderr)
        return False


def get_image_files(directory):
    """Get all image files from a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file.lower())[1] in image_extensions:
                image_files.append(os.path.join(root, file))

    return sorted(image_files)


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze underwater image quality with visual metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phân tích 1 ảnh với tất cả plots (mặc định line)
  python analyze_underwater_image.py input.jpg

  # Phân tích 1 ảnh với biểu đồ bar
  python analyze_underwater_image.py input.jpg --plot-type bar

  # Phân tích 1 ảnh với cả line và bar
  python analyze_underwater_image.py input.jpg --plot-type both

  # Chỉ hiển thị RGB histogram
  python analyze_underwater_image.py input.jpg --plots rgb

  # Hiển thị image và RGB histogram
  python analyze_underwater_image.py input.jpg --plots image rgb

  # Phân tích tất cả ảnh trong thư mục
  python analyze_underwater_image.py /path/to/images/ --output-dir /path/to/output/

  # Phân tích thư mục với bar chart và chỉ RGB + HSV
  python analyze_underwater_image.py /path/to/images/ --output-dir /path/to/output/ --plot-type bar --plots rgb hsv

  # Phân tích thư mục, tự động tạo thư mục output
  python analyze_underwater_image.py /path/to/images/
        """
    )

    parser.add_argument('input', type=str,
                       help='Path to input image file or directory containing images')
    parser.add_argument('output_path', type=str, nargs='?', default=None,
                       help='Path to save output (file for single image, ignored for directory)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for batch processing (used when input is a directory)')
    parser.add_argument('--plot-type', type=str, choices=['line', 'bar', 'both'],
                       default='line',
                       help='Type of plot for histograms: line (default), bar, or both')
    parser.add_argument('--plots', type=str, nargs='+',
                       choices=['image', 'rgb', 'hsv', 'laplacian', 'edge'],
                       default=None,
                       help='Select which plots to display (default: all). Options: image, rgb, hsv, laplacian, edge')

    args = parser.parse_args()

    # Check if input is a file or directory
    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    is_directory = os.path.isdir(args.input)

    if is_directory:
        # Batch processing mode
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING MODE")
        print(f"{'='*60}")
        print(f"Input directory: {args.input}")

        # Get all image files
        image_files = get_image_files(args.input)

        if not image_files:
            print(f"Error: No image files found in directory: {args.input}", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(image_files)} image(s) to process")

        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            # Create output directory next to input directory
            output_dir = os.path.join(args.input, 'analysis_output')

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # Process each image
        success_count = 0
        failed_count = 0

        for i, img_path in enumerate(image_files, 1):
            # Generate output path
            rel_path = os.path.relpath(img_path, args.input)
            base_name = os.path.splitext(rel_path)[0]
            plot_suffix = f"_{args.plot_type}" if args.plots and any(p in args.plots for p in ['rgb', 'hsv']) else ""
            output_path = os.path.join(output_dir, f"{base_name}_analysis{plot_suffix}.png")

            # Create subdirectories if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            print(f"\n[{i}/{len(image_files)}]", end=" ")

            if process_single_image(img_path, output_path, args.plot_type, args.plots):
                success_count += 1
            else:
                failed_count += 1

        # Summary
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total images: {len(image_files)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {failed_count}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")

    else:
        # Single file mode
        # Determine output path
        if args.output_path is None:
            base_name = os.path.splitext(args.input)[0]
            plot_suffix = f"_{args.plot_type}" if args.plots and any(p in args.plots for p in ['rgb', 'hsv']) else ""
            output_path = f"{base_name}_analysis{plot_suffix}.png"
        else:
            output_path = args.output_path

        success = process_single_image(args.input, output_path, args.plot_type, args.plots)

        if not success:
            sys.exit(1)


if __name__ == '__main__':
    main()
