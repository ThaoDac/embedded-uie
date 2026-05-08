"""
DCP-Only UIE + YOLO Detection Pipeline
======================================

Dedicated pipeline for Dark Channel Prior (DCP) model.
DCP is slow (3-4 hours for 800 images) so it runs separately.

Pipeline:
1. Run DCP enhancement on test images
2. Run YOLO detection on ENHANCED images with visualizations
3. Run YOLO detection on ORIGINAL images with visualizations
4. Create 3-way side-by-side comparison images
5. Generate comprehensive report

Usage:
    python pipeline_dcp_only.py
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
OUTPUT_BASE = BASE_DIR / f"UIE_YOLO_DCP_{TIMESTAMP}"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

DCP_SCRIPT = BASE_DIR / 'DCP-main/app_console.py'

# ============================================================================
# Helper Functions
# ============================================================================

def run_dcp_enhancement(input_dir, output_dir):
    """Run DCP enhancement with extended timeout"""
    print(f"\n{'='*80}\nRunning DCP Enhancement\n{'='*80}")
    print(f"⚠️  WARNING: DCP is very slow. Expected time: 3-4 hours for 800 images")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    if not DCP_SCRIPT.exists():
        print(f"✗ DCP script not found: {DCP_SCRIPT}")
        return False

    # Create output structure
    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare command
    cmd = [sys.executable, str(DCP_SCRIPT), '--input', str(input_dir / 'images'), '--output', str(images_output_dir)]

    model_dir = DCP_SCRIPT.parent
    timeout = 18000  # 5 hours (extra buffer for safety)

    print(f"\nStarting DCP at {time.strftime('%H:%M:%S')}...")
    print(f"Timeout: {timeout/3600:.1f} hours")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(model_dir))
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"\n✗ DCP Failed!")
            print(f"Error: {result.stderr[:500]}")
            return False

        hours = elapsed / 3600
        print(f"\n✓ DCP Completed in {hours:.2f} hours ({elapsed:.0f}s)")

        # Copy labels
        labels_src = input_dir / 'labels'
        if labels_src.exists():
            shutil.copytree(labels_src, labels_output_dir, dirs_exist_ok=True)
            num_labels = len(list(labels_src.glob('*.txt')))
            print(f"✓ Copied {num_labels} label files")

        # Check output
        num_images = len(list(images_output_dir.glob('*.jpg')))
        print(f"✓ Enhanced {num_images} images")

        return True

    except subprocess.TimeoutExpired:
        print(f"\n✗ DCP TIMEOUT after {timeout/3600:.1f} hours!")
        print(f"   DCP may be too slow for this dataset size.")
        return False
    except Exception as e:
        print(f"\n✗ DCP Error: {e}")
        return False


def run_yolo_detection(model_path, test_dir, data_yaml, output_dir, save_viz=True):
    """Run YOLO detection on images with visualization"""
    print(f"\n{'='*80}\nRunning YOLO Detection\n{'='*80}")
    print(f"Test directory: {test_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Update data.yaml
    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)

    data_cfg['test'] = str(Path(test_dir) / 'images')

    temp_yaml = output_path / 'data_temp.yaml'
    with open(temp_yaml, 'w') as f:
        yaml.dump(data_cfg, f)

    model = YOLO(model_path)

    try:
        # Run validation
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

        # Extract metrics
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

        # Save visualizations
        if save_viz:
            viz_output = output_path / 'visualizations'
            viz_output.mkdir(parents=True, exist_ok=True)

            test_images_dir = Path(test_dir) / 'images'
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
                    continue

            print(f"✓ Visualizations saved to: {viz_output}")
            metrics['visualization_dir'] = str(viz_output)

        return metrics
    except Exception as e:
        print(f"✗ YOLO failed: {e}")
        return {}


def draw_yolo_boxes_on_image(image, label_file, class_names=None, color=(0, 255, 0), thickness=2):
    """Draw YOLO format bounding boxes on image"""
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

            # Convert YOLO format to pixel coordinates
            x_center_px = int(x_center * img_w)
            y_center_px = int(y_center * img_h)
            width_px = int(width * img_w)
            height_px = int(height * img_h)

            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)

            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)

            if class_names and class_id < len(class_names):
                label_text = class_names[class_id]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1

                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, font, font_scale, font_thickness
                )

                cv2.rectangle(
                    img_with_boxes,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )

                cv2.putText(
                    img_with_boxes,
                    label_text,
                    (x1, y1 - baseline - 2),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness
                )

    except Exception as e:
        pass

    return img_with_boxes


def create_comparison_images(raw_images_dir, labels_dir, enhanced_viz_dir, original_viz_dir, output_dir, class_names=None):
    """Create 3-way side-by-side comparison images"""
    print(f"\n{'='*80}\nCREATING 3-WAY VISUAL COMPARISONS\n{'='*80}")

    raw_path = Path(raw_images_dir)
    labels_path = Path(labels_dir)
    enhanced_path = Path(enhanced_viz_dir)
    original_path = Path(original_viz_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    enhanced_files = {f.name: f for f in enhanced_path.glob('*') if f.suffix.lower() in image_exts}

    comparison_count = 0
    for img_name, enhanced_file in tqdm(enhanced_files.items(), desc="Creating 3-way comparisons"):
        original_file = original_path / img_name
        raw_file = raw_path / img_name
        label_name = Path(img_name).stem + '.txt'
        label_file = labels_path / label_name

        if not original_file.exists() or not raw_file.exists():
            continue

        try:
            raw_img = cv2.imread(str(raw_file))
            enhanced_img = cv2.imread(str(enhanced_file))
            original_img = cv2.imread(str(original_file))

            if raw_img is None or enhanced_img is None or original_img is None:
                continue

            h, w = enhanced_img.shape[:2]
            raw_img_resized = cv2.resize(raw_img, (w, h))

            raw_img_with_gt = draw_yolo_boxes_on_image(
                raw_img_resized,
                label_file,
                class_names=class_names,
                color=(0, 255, 0),
                thickness=2
            )

            original_img = cv2.resize(original_img, (w, h))

            label_height = 50
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            font_color = (255, 255, 255)
            bg_color = (0, 0, 0)

            labels = ['Raw Image + Ground Truth', 'DCP Enhanced + Detection', 'Original + Detection']
            images = [raw_img_with_gt, enhanced_img, original_img]

            labeled_images = []
            for img, label in zip(images, labels):
                labeled = np.zeros((h + label_height, w, 3), dtype=np.uint8)
                labeled[:label_height] = bg_color
                labeled[label_height:] = img

                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                text_x = (w - text_size[0]) // 2
                text_y = (label_height + text_size[1]) // 2
                cv2.putText(labeled, label, (text_x, text_y), font, font_scale, font_color, font_thickness)

                labeled_images.append(labeled)

            comparison = np.hstack(labeled_images)
            output_file = output_path / f'comparison_{img_name}'
            cv2.imwrite(str(output_file), comparison)
            comparison_count += 1

        except Exception as e:
            continue

    print(f"\n✓ Created {comparison_count} 3-way comparison images")
    print(f"  Saved to: {output_path}\n")


def generate_report(original_metrics, dcp_metrics, output_file):
    """Generate markdown report"""
    report = [
        "# DCP + YOLO Detection Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Model | mAP50 | mAP50-95 | Precision | Recall | F1 |",
        "|-------|-------|----------|-----------|--------|----| "
    ]

    # Original
    om = original_metrics
    report.append(
        f"| Original | {om.get('mAP50', 0):.4f} | {om.get('mAP50-95', 0):.4f} | "
        f"{om.get('precision', 0):.4f} | {om.get('recall', 0):.4f} | {om.get('f1', 0):.4f} |"
    )

    # DCP
    dm = dcp_metrics
    report.append(
        f"| DCP Enhanced | {dm.get('mAP50', 0):.4f} | {dm.get('mAP50-95', 0):.4f} | "
        f"{dm.get('precision', 0):.4f} | {dm.get('recall', 0):.4f} | {dm.get('f1', 0):.4f} |"
    )

    report.extend(["", "## Improvement Analysis", ""])

    # Calculate improvements
    improvements = {}
    for key in ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']:
        orig_val = om.get(key, 0)
        dcp_val = dm.get(key, 0)
        if orig_val > 0:
            improvement = ((dcp_val - orig_val) / orig_val) * 100
            improvements[key] = improvement

    report.extend([
        "| Metric | Original | DCP Enhanced | Improvement |",
        "|--------|----------|--------------|-------------|"
    ])

    for key in ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']:
        imp = improvements.get(key, 0)
        imp_str = f"{imp:+.2f}%" if abs(imp) > 0 else "0.00%"
        status = "✓" if imp > 0 else "✗"
        report.append(
            f"| {key} | {om.get(key, 0):.4f} | {dm.get(key, 0):.4f} | {imp_str} {status} |"
        )

    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    print(f"\n✓ Report: {output_file}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    print(f"\n{'='*80}\nDCP-ONLY UIE + YOLO PIPELINE\n{'='*80}\nOutput: {OUTPUT_BASE}\n{'='*80}\n")

    # Step 1: Run YOLO on Original Images (Baseline)
    print(f"\n{'='*80}\nSTEP 1: BASELINE - Original Images\n{'='*80}")
    original_output = OUTPUT_BASE / 'Original'
    original_metrics = run_yolo_detection(YOLO_MODEL, TEST_DIR, DATA_YAML, original_output, save_viz=True)

    if not original_metrics:
        print("✗ Failed to run baseline YOLO detection. Exiting.")
        return

    # Step 2: Run DCP Enhancement
    print(f"\n{'='*80}\nSTEP 2: DCP ENHANCEMENT\n{'='*80}")
    enhanced_dir = OUTPUT_BASE / 'DCP' / 'enhanced'
    success = run_dcp_enhancement(TEST_DIR, enhanced_dir)

    if not success:
        print("✗ DCP enhancement failed. Exiting.")
        return

    # Step 3: Run YOLO on DCP Enhanced Images
    print(f"\n{'='*80}\nSTEP 3: YOLO on DCP Enhanced Images\n{'='*80}")
    yolo_output = OUTPUT_BASE / 'DCP' / 'yolo'
    dcp_metrics = run_yolo_detection(YOLO_MODEL, enhanced_dir, DATA_YAML, yolo_output, save_viz=True)

    if not dcp_metrics:
        print("✗ Failed to run YOLO on DCP enhanced images.")
        return

    # Step 4: Create 3-way Comparisons
    print(f"\n{'='*80}\nSTEP 4: CREATING VISUAL COMPARISONS\n{'='*80}")

    # Load class names
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

    raw_images_dir = TEST_DIR / 'images'
    labels_dir = TEST_DIR / 'labels'
    original_viz_dir = original_metrics.get('visualization_dir')
    dcp_viz_dir = dcp_metrics.get('visualization_dir')

    if original_viz_dir and dcp_viz_dir:
        comparison_dir = OUTPUT_BASE / 'DCP' / 'comparisons'
        create_comparison_images(
            raw_images_dir=str(raw_images_dir),
            labels_dir=str(labels_dir),
            enhanced_viz_dir=dcp_viz_dir,
            original_viz_dir=original_viz_dir,
            output_dir=str(comparison_dir),
            class_names=class_names
        )
    else:
        print("⚠️  Visualizations not found, skipping comparisons")

    # Step 5: Generate Report
    print(f"\n{'='*80}\nSTEP 5: GENERATING REPORT\n{'='*80}")
    report_file = OUTPUT_BASE / 'DCP_YOLO_Report.md'
    generate_report(original_metrics, dcp_metrics, report_file)

    # Save results
    results = {
        'Original': original_metrics,
        'DCP': dcp_metrics
    }
    with open(OUTPUT_BASE / 'results.yaml', 'w') as f:
        yaml.dump(results, f)

    print(f"\n{'='*80}\n✓✓✓ PIPELINE COMPLETED! ✓✓✓\n{'='*80}")
    print(f"\nResults saved to: {OUTPUT_BASE}")
    print(f"Report: {report_file}")
    print(f"\nComparison images: {OUTPUT_BASE / 'DCP' / 'comparisons'}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()