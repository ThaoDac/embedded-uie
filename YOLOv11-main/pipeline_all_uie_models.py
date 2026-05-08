"""
Multi-UIE Model + YOLO Detection Evaluation Pipeline
=====================================================

Pipeline:
1. Run UIE model to enhance test images
2. Run YOLO detection on ENHANCED images with visualizations
3. Run YOLO detection on ORIGINAL images with visualizations
4. Create 3-way side-by-side comparison images:
   [Raw Image + Ground Truth (Green)] | [Enhanced + Detection] | [Original + Detection]
5. Compare metrics and generate comprehensive report

Features:
- 11 UIE models (5 non-DL + 6 DL)
- YOLO detection with bounding box visualizations
- 3-way visual comparisons with ground truth
- Comprehensive metrics comparison report
- Automatic directory structure handling

Output Structure:
UIE_YOLO_AllModels_{timestamp}/
├── Original/
│   ├── yolo/results/                # YOLO metrics
│   └── visualizations/              # Original images with detection boxes
├── {Model_Name}/
│   ├── enhanced/
│   │   ├── images/                  # Enhanced images
│   │   └── labels/                  # Ground truth labels
│   ├── yolo/
│   │   ├── results/                 # YOLO metrics
│   │   └── visualizations/          # Enhanced images with detection boxes
│   └── comparisons/                 # 3-way comparison images
│       └── comparison_*.jpg         # [Raw+GT | Enhanced+Det | Original+Det]
├── UIE_YOLO_Report.md               # Comprehensive markdown report
└── results.yaml                     # All metrics in YAML format

Usage:
    python pipeline_all_uie_models.py
"""

import os
import sys
import yaml
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path("/home/ndpthao/eject/IMPLEMENTATION")
DATASET_DIR = BASE_DIR / "Dataset/URPC2020"
TEST_DIR = DATASET_DIR / "test"
YOLO_MODEL = "/home/ndpthao/eject/runs/detect/train8/weights/best.pt"
DATA_YAML = DATASET_DIR / "data.yaml"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_BASE = BASE_DIR / f"UIE_YOLO_AllModels_{TIMESTAMP}"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# UIE Models Configuration
UIE_MODELS = {
    'CLAHE': {'script': BASE_DIR / 'CLAHE-main/app_console.py', 'type': 'non-DL'},
    'DCP': {'script': BASE_DIR / 'DCP-main/app_console.py', 'type': 'non-DL'},
    'RGHS': {'script': BASE_DIR / 'RGHS/app_console.py', 'type': 'non-DL'},
    'UDCP': {'script': BASE_DIR / 'UDCP-main/app_console.py', 'type': 'non-DL'},
    'ULAP': {'script': BASE_DIR / 'ULAP-main/app_console.py', 'type': 'non-DL'},
    'FUnIE-GAN': {'script': BASE_DIR / 'FUnIE-GAN-main/app_console_pytorch.py', 'type': 'DL'},
    'HCLR-Net': {'script': BASE_DIR / 'HCLR-Net-main/app_console.py', 'type': 'DL'},
    'LU2Net': {'script': BASE_DIR / 'LU2Net-master/app_console.py', 'type': 'DL'},
    'PGHS': {'script': BASE_DIR / 'PGHS-main/app_console.py', 'type': 'DL'},
    'UIR-PolyKernel': {'script': BASE_DIR / 'UIR-PolyKernel-main/app_console.py', 'type': 'DL'},
    'WaterFormer': {'script': BASE_DIR / 'WaterFormer-master/app_console.py', 'type': 'DL'}
}

def run_uie_model(model_name, script_path, input_dir, output_dir):
    """Run UIE enhancement"""
    if not script_path.exists():
        print(f"⚠️  {model_name}: Script not found")
        return False

    print(f"\n{'='*80}\nRunning {model_name}\n{'='*80}")

    # Create output structure: output_dir/images/ and output_dir/labels/
    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    # UIE models will output enhanced images to images/ subdirectory
    cmd = [sys.executable, str(script_path), '--input', str(input_dir / 'images'), '--output', str(images_output_dir)]

    # IMPORTANT: Run from model directory (some models need relative paths for config/weights)
    model_dir = script_path.parent

    # Set timeout based on model type (non-DL models are slower)
    # DCP, UDCP, ULAP can take 4+ hours for 800 images
    is_slow_model = any(x in model_name.upper() for x in ['DCP', 'UDCP', 'ULAP', 'RGHS'])
    timeout = 14400 if is_slow_model else 7200  # 4 hours vs 2 hours

    if is_slow_model:
        print(f"⚠️  WARNING: {model_name} is a slow algorithm. This may take 2-4 hours for 800 images.")
        print(f"   Timeout set to {timeout/3600:.1f} hours")

    try:
        start = time.time()
        print(f"Starting at {time.strftime('%H:%M:%S')}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(model_dir))
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"✗ Failed: {result.stderr[:200]}")
            return False

        print(f"✓ Completed in {elapsed:.2f}s")

        # Copy labels from test set to labels/ subdirectory
        labels_src = input_dir / 'labels'
        if labels_src.exists():
            shutil.copytree(labels_src, labels_output_dir, dirs_exist_ok=True)
            num_labels = len(list(labels_src.glob('*.txt')))
            print(f"✓ Copied {num_labels} label files")

            # Fix label filenames to match enhanced image filenames
            # Some models (like PGHS) rename images with prefix (e.g., "enhanced_000001.jpg")
            # We need to rename labels accordingly
            image_files = {f.stem: f for f in images_output_dir.glob('*') if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}}
            label_files = list(labels_output_dir.glob('*.txt'))

            for label_file in label_files:
                label_stem = label_file.stem  # e.g., "000001"

                # Find matching image with any prefix
                matching_image = None
                for img_stem, img_path in image_files.items():
                    if img_stem == label_stem or img_stem.endswith(label_stem):
                        matching_image = img_stem
                        break

                # Rename label to match image
                if matching_image and matching_image != label_stem:
                    new_label_name = matching_image + '.txt'
                    new_label_path = labels_output_dir / new_label_name
                    label_file.rename(new_label_path)

            print(f"✓ Synchronized label filenames with enhanced images")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def run_yolo_on_enhanced(model_path, enhanced_dir, data_yaml, output_dir, save_viz=True):
    """Run YOLO detection on enhanced images with visualization"""
    print(f"\n{'='*80}\nRunning YOLO Detection\n{'='*80}")
    print(f"Enhanced images: {enhanced_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Update data.yaml
    # IMPORTANT: enhanced_dir should already contain /images and /labels subdirectories
    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)

    # Point to the images subdirectory (YOLO will automatically find labels/ sibling directory)
    data_cfg['test'] = str(Path(enhanced_dir) / 'images')

    temp_yaml = output_path / 'data_temp.yaml'
    with open(temp_yaml, 'w') as f:
        yaml.dump(data_cfg, f)

    model = YOLO(model_path)

    try:
        results = model.val(
            data=str(temp_yaml),
            split='test',
            imgsz=256,
            batch=1,
            device='0',
            save_json=True,
            project=str(output_path),
            name='results',
            exist_ok=True,
            verbose=False
        )

        box = getattr(results, 'box', None)
        metrics = {}
        if box:
            metrics['mAP50-95'] = float(box.map) if hasattr(box, 'map') else 0.0
            metrics['mAP50'] = float(box.map50) if hasattr(box, 'map50') else 0.0
            metrics['precision'] = float(box.mp) if hasattr(box, 'mp') else 0.0
            metrics['recall'] = float(box.mr) if hasattr(box, 'mr') else 0.0
            if metrics['precision'] > 0 and metrics['recall'] > 0:
                metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])

        with open(output_path / 'metrics.yaml', 'w') as f:
            yaml.dump(metrics, f)

        print(f"mAP@0.5: {metrics.get('mAP50', 0):.4f} | Precision: {metrics.get('precision', 0):.4f} | Recall: {metrics.get('recall', 0):.4f}")

        # Save detection visualizations
        if save_viz:
            viz_output = output_path / 'visualizations'
            viz_output.mkdir(parents=True, exist_ok=True)

            test_images_dir = Path(enhanced_dir) / 'images'
            image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = [f for f in test_images_dir.glob('*') if f.suffix.lower() in image_exts]

            for img_file in tqdm(image_files, desc="Saving detection visualizations", leave=False):
                try:
                    results_pred = model.predict(
                        source=str(img_file),
                        imgsz=256,
                        conf=0.25,
                        device='0',
                        save=False,
                        verbose=False
                    )

                    if len(results_pred) > 0:
                        result = results_pred[0]
                        annotated_img = result.plot()
                        output_file = viz_output / img_file.name
                        cv2.imwrite(str(output_file), annotated_img)

                except Exception as e:
                    print(f"\nError visualizing {img_file.name}: {e}")
                    continue

            print(f"✓ Visualizations saved to: {viz_output}")
            metrics['visualization_dir'] = str(viz_output)

        return metrics
    except Exception as e:
        print(f"✗ YOLO failed: {e}")
        return {}

def draw_yolo_boxes_on_image(image, label_file, class_names=None, color=(0, 255, 0), thickness=2):
    """
    Draw YOLO format bounding boxes on image

    Args:
        image: Input image (numpy array)
        label_file: Path to YOLO format label file (.txt)
        class_names: List of class names (optional)
        color: Box color in BGR format
        thickness: Line thickness

    Returns:
        Image with bounding boxes drawn
    """
    img_h, img_w = image.shape[:2]
    img_with_boxes = image.copy()

    if not os.path.exists(label_file):
        return img_with_boxes

    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert YOLO format (normalized) to pixel coordinates
            x_center_px = int(x_center * img_w)
            y_center_px = int(y_center * img_h)
            width_px = int(width * img_w)
            height_px = int(height * img_h)

            # Calculate top-left and bottom-right corners
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)

            # Draw rectangle
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)

            # Draw label if class names provided
            if class_names and class_id < len(class_names):
                label_text = class_names[class_id]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1

                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, font, font_scale, font_thickness
                )

                # Draw background rectangle for text
                cv2.rectangle(
                    img_with_boxes,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1  # Filled
                )

                # Draw text
                cv2.putText(
                    img_with_boxes,
                    label_text,
                    (x1, y1 - baseline - 2),
                    font,
                    font_scale,
                    (255, 255, 255),  # White text
                    font_thickness
                )

    except Exception as e:
        print(f"Warning: Error drawing boxes from {label_file}: {e}")

    return img_with_boxes


def create_comparison_images(raw_images_dir, labels_dir, enhanced_viz_dir, original_viz_dir,
                            output_dir, class_names=None):
    """
    Create 3-way side-by-side comparison images:
    [Raw Image + Ground Truth] | [Enhanced + Detection] | [Original + Detection]

    Args:
        raw_images_dir: Directory with raw original images (no bounding boxes)
        labels_dir: Directory with ground truth labels (YOLO format)
        enhanced_viz_dir: Directory with enhanced + detection visualizations
        original_viz_dir: Directory with original + detection visualizations
        output_dir: Output directory for comparison images
        class_names: List of class names for labeling
    """
    print(f"\n{'='*80}\nCREATING 3-WAY VISUAL COMPARISONS\n{'='*80}")
    print(f"Raw images:  {raw_images_dir}")
    print(f"Labels:      {labels_dir}")
    print(f"Enhanced:    {enhanced_viz_dir}")
    print(f"Original:    {original_viz_dir}")
    print(f"Output:      {output_dir}\n")

    raw_path = Path(raw_images_dir)
    labels_path = Path(labels_dir)
    enhanced_path = Path(enhanced_viz_dir)
    original_path = Path(original_viz_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get image files from enhanced directory
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    enhanced_files = {f.name: f for f in enhanced_path.glob('*') if f.suffix.lower() in image_exts}

    comparison_count = 0
    for img_name, enhanced_file in tqdm(enhanced_files.items(), desc="Creating 3-way comparisons"):
        original_file = original_path / img_name
        raw_file = raw_path / img_name

        # Get corresponding label file
        label_name = Path(img_name).stem + '.txt'
        label_file = labels_path / label_name

        # Check all files exist
        if not original_file.exists():
            continue

        if not raw_file.exists():
            continue

        try:
            # Read all three images
            raw_img = cv2.imread(str(raw_file))
            enhanced_img = cv2.imread(str(enhanced_file))
            original_img = cv2.imread(str(original_file))

            if raw_img is None or enhanced_img is None or original_img is None:
                continue

            # Get target size from enhanced image
            h, w = enhanced_img.shape[:2]

            # Resize raw image to match
            raw_img_resized = cv2.resize(raw_img, (w, h))

            # Draw ground truth bounding boxes on raw image
            raw_img_with_gt = draw_yolo_boxes_on_image(
                raw_img_resized,
                label_file,
                class_names=class_names,
                color=(0, 255, 0),  # Green for ground truth
                thickness=2
            )

            # Resize original detection to match
            original_img = cv2.resize(original_img, (w, h))

            # Add text labels
            label_height = 50
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            font_color = (255, 255, 255)
            bg_color = (0, 0, 0)

            # Labels for each image
            labels = [
                'Raw Image + Ground Truth',
                'Enhanced + Detection',
                'Original + Detection'
            ]
            images = [raw_img_with_gt, enhanced_img, original_img]

            # Create labeled images
            labeled_images = []
            for img, label in zip(images, labels):
                # Create image with label header
                labeled = np.zeros((h + label_height, w, 3), dtype=np.uint8)
                labeled[:label_height] = bg_color
                labeled[label_height:] = img

                # Add centered text
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                text_x = (w - text_size[0]) // 2
                text_y = (label_height + text_size[1]) // 2
                cv2.putText(labeled, label, (text_x, text_y),
                           font, font_scale, font_color, font_thickness)

                labeled_images.append(labeled)

            # Concatenate all three images horizontally
            comparison = np.hstack(labeled_images)

            # Save comparison image
            output_file = output_path / f'comparison_{img_name}'
            cv2.imwrite(str(output_file), comparison)
            comparison_count += 1

        except Exception as e:
            print(f"\nError creating comparison for {img_name}: {e}")
            continue

    print(f"\n✓ Created {comparison_count} 3-way comparison images")
    print(f"  Format: [Raw+GT] | [Enhanced+Detection] | [Original+Detection]")
    print(f"  Ground truth boxes: Green")
    print(f"  Saved to: {output_path}\n")


def generate_report(results, output_file):
    """Generate markdown report"""
    report = [
        "# UIE + YOLO Detection Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Model | Type | mAP50 | mAP50-95 | Precision | Recall | F1 | Status |",
        "|-------|------|-------|----------|-----------|--------|----|----|"
    ]

    for name in sorted(results.keys()):
        r = results[name]
        m = r.get('metrics', {})
        status = "✓" if r.get('status') == 'Success' else "✗"
        report.append(
            f"| {name} | {r.get('type', '-')} | {m.get('mAP50', 0):.4f} | "
            f"{m.get('mAP50-95', 0):.4f} | {m.get('precision', 0):.4f} | "
            f"{m.get('recall', 0):.4f} | {m.get('f1', 0):.4f} | {status} |"
        )

    report.extend(["", "## Detailed Results", ""])

    for name in sorted(results.keys()):
        r = results[name]
        if r.get('status') != 'Success':
            continue
        m = r['metrics']
        report.extend([
            f"### {name}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| mAP@0.5 | {m.get('mAP50', 0):.4f} |",
            f"| mAP@0.5:0.95 | {m.get('mAP50-95', 0):.4f} |",
            f"| Precision | {m.get('precision', 0):.4f} |",
            f"| Recall | {m.get('recall', 0):.4f} |",
            f"| F1-Score | {m.get('f1', 0):.4f} |",
            ""
        ])

    if 'Original' in results:
        baseline = results['Original']['metrics']
        report.extend(["## Improvement Over Baseline", "", "| Model | ΔmAP50 | Status |", "|-------|--------|--------|"])
        for name in sorted(results.keys()):
            if name == 'Original' or results[name].get('status') != 'Success':
                continue
            delta = results[name]['metrics'].get('mAP50', 0) - baseline.get('mAP50', 0)
            status = "✓" if delta > 0 else "✗"
            report.append(f"| {name} | {delta:+.4f} | {status} |")

    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    print(f"\n✓ Report: {output_file}")

def main():
    print(f"\n{'='*80}\nMULTI-UIE + YOLO PIPELINE\n{'='*80}\nOutput: {OUTPUT_BASE}\n{'='*80}\n")

    results = {}

    # Baseline
    print(f"\nBASELINE: Original Images")
    original_output = OUTPUT_BASE / 'Original'
    baseline_metrics = run_yolo_on_enhanced(YOLO_MODEL, TEST_DIR, DATA_YAML, original_output, save_viz=True)
    results['Original'] = {'type': 'Baseline', 'status': 'Success' if baseline_metrics else 'Failed', 'metrics': baseline_metrics}

    # Each UIE model
    for idx, (name, config) in enumerate(UIE_MODELS.items(), 1):
        print(f"\n[{idx}/{len(UIE_MODELS)}] {name}")

        enhanced_dir = OUTPUT_BASE / name / 'enhanced'
        success = run_uie_model(name, config['script'], TEST_DIR, enhanced_dir)

        if not success:
            results[name] = {'type': config['type'], 'status': 'Failed-UIE', 'metrics': {}}
            continue

        yolo_output = OUTPUT_BASE / name / 'yolo'
        metrics = run_yolo_on_enhanced(YOLO_MODEL, enhanced_dir, DATA_YAML, yolo_output, save_viz=True)
        results[name] = {'type': config['type'], 'status': 'Success' if metrics else 'Failed-YOLO', 'metrics': metrics}

    # Create 3-way comparisons for each successful UIE model
    print(f"\n{'='*80}\nCREATING 3-WAY COMPARISONS\n{'='*80}\n")

    # Load class names from data.yaml
    class_names = None
    if DATA_YAML.exists():
        try:
            with open(DATA_YAML, 'r') as f:
                data_cfg = yaml.safe_load(f)
                class_names = data_cfg.get('names', None)
                if class_names:
                    print(f"Loaded {len(class_names)} class names: {class_names}")
        except Exception as e:
            print(f"Warning: Could not load class names: {e}")

    # Raw images and labels
    raw_images_dir = TEST_DIR / 'images'
    labels_dir = TEST_DIR / 'labels'

    # Original detection visualizations
    original_viz_dir = results['Original']['metrics'].get('visualization_dir')

    if original_viz_dir and Path(original_viz_dir).exists():
        for name in sorted(results.keys()):
            if name == 'Original' or results[name].get('status') != 'Success':
                continue

            enhanced_viz_dir = results[name]['metrics'].get('visualization_dir')
            if not enhanced_viz_dir or not Path(enhanced_viz_dir).exists():
                continue

            comparison_dir = OUTPUT_BASE / name / 'comparisons'

            print(f"\nCreating comparisons for {name}...")
            create_comparison_images(
                raw_images_dir=str(raw_images_dir),
                labels_dir=str(labels_dir),
                enhanced_viz_dir=enhanced_viz_dir,
                original_viz_dir=original_viz_dir,
                output_dir=str(comparison_dir),
                class_names=class_names
            )
    else:
        print("⚠️  Original visualizations not found, skipping comparisons")

    # Report
    report_file = OUTPUT_BASE / 'UIE_YOLO_Report.md'
    generate_report(results, report_file)

    with open(OUTPUT_BASE / 'results.yaml', 'w') as f:
        yaml.dump(results, f)

    print(f"\n{'='*80}\nCOMPLETED\n{'='*80}\n")
    print(f"Results saved to: {OUTPUT_BASE}")
    print(f"Report: {report_file}")

if __name__ == '__main__':
    main()
