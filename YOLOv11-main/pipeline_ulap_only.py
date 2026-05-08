"""
ULAP-Only UIE + YOLO Detection Pipeline
=======================================

Dedicated pipeline for Underwater Light Attenuation Prior (ULAP) model.
ULAP is slow (2-3 hours for 800 images) so it runs separately.

Pipeline:
1. Run ULAP enhancement on test images
2. Run YOLO detection on ENHANCED images with visualizations
3. Run YOLO detection on ORIGINAL images with visualizations
4. Create 3-way side-by-side comparison images
5. Generate comprehensive report

Usage:
    python pipeline_ulap_only.py
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
OUTPUT_BASE = BASE_DIR / f"UIE_YOLO_ULAP_{TIMESTAMP}"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

ULAP_SCRIPT = BASE_DIR / 'ULAP-main/app_console.py'

# Import helper functions
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_dcp_only import (
    run_yolo_detection,
    draw_yolo_boxes_on_image
)

# ============================================================================
# ULAP Enhancement
# ============================================================================

def run_ulap_enhancement(input_dir, output_dir):
    """Run ULAP enhancement with extended timeout"""
    print(f"\n{'='*80}\nRunning ULAP Enhancement\n{'='*80}")
    print(f"⚠️  WARNING: ULAP is slow. Expected time: 2-3 hours for 800 images")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    if not ULAP_SCRIPT.exists():
        print(f"✗ ULAP script not found: {ULAP_SCRIPT}")
        return False

    # Create output structure
    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare command
    cmd = [sys.executable, str(ULAP_SCRIPT), '--input', str(input_dir / 'images'), '--output', str(images_output_dir)]

    model_dir = ULAP_SCRIPT.parent
    timeout = 14400  # 4 hours (ULAP is slightly faster than DCP/UDCP)

    print(f"\nStarting ULAP at {time.strftime('%H:%M:%S')}...")
    print(f"Timeout: {timeout/3600:.1f} hours")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(model_dir))
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"\n✗ ULAP Failed!")
            print(f"Error: {result.stderr[:500]}")
            return False

        hours = elapsed / 3600
        print(f"\n✓ ULAP Completed in {hours:.2f} hours ({elapsed:.0f}s)")

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
        print(f"\n✗ ULAP TIMEOUT after {timeout/3600:.1f} hours!")
        return False
    except Exception as e:
        print(f"\n✗ ULAP Error: {e}")
        return False


def create_ulap_comparisons(raw_images_dir, labels_dir, enhanced_viz_dir, original_viz_dir, output_dir, class_names=None):
    """Create 3-way comparisons with ULAP label"""
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
                raw_img_resized, label_file, class_names=class_names, color=(0, 255, 0), thickness=2
            )

            original_img = cv2.resize(original_img, (w, h))

            label_height = 50
            font = cv2.FONT_HERSHEY_SIMPLEX
            labels = ['Raw Image + Ground Truth', 'ULAP Enhanced + Detection', 'Original + Detection']
            images = [raw_img_with_gt, enhanced_img, original_img]

            labeled_images = []
            for img, label in zip(images, labels):
                labeled = np.zeros((h + label_height, w, 3), dtype=np.uint8)
                labeled[:label_height] = (0, 0, 0)
                labeled[label_height:] = img

                text_size = cv2.getTextSize(label, font, 0.8, 2)[0]
                text_x = (w - text_size[0]) // 2
                text_y = (label_height + text_size[1]) // 2
                cv2.putText(labeled, label, (text_x, text_y), font, 0.8, (255, 255, 255), 2)

                labeled_images.append(labeled)

            comparison = np.hstack(labeled_images)
            output_file = output_path / f'comparison_{img_name}'
            cv2.imwrite(str(output_file), comparison)
            comparison_count += 1

        except Exception as e:
            continue

    print(f"\n✓ Created {comparison_count} 3-way comparison images")
    print(f"  Saved to: {output_path}\n")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    print(f"\n{'='*80}\nULAP-ONLY UIE + YOLO PIPELINE\n{'='*80}\nOutput: {OUTPUT_BASE}\n{'='*80}\n")

    # Step 1: Baseline
    print(f"\n{'='*80}\nSTEP 1: BASELINE - Original Images\n{'='*80}")
    original_output = OUTPUT_BASE / 'Original'
    original_metrics = run_yolo_detection(YOLO_MODEL, TEST_DIR, DATA_YAML, original_output, save_viz=True)

    if not original_metrics:
        print("✗ Failed to run baseline YOLO detection. Exiting.")
        return

    # Step 2: ULAP Enhancement
    print(f"\n{'='*80}\nSTEP 2: ULAP ENHANCEMENT\n{'='*80}")
    enhanced_dir = OUTPUT_BASE / 'ULAP' / 'enhanced'
    success = run_ulap_enhancement(TEST_DIR, enhanced_dir)

    if not success:
        print("✗ ULAP enhancement failed. Exiting.")
        return

    # Step 3: YOLO on ULAP Enhanced Images
    print(f"\n{'='*80}\nSTEP 3: YOLO on ULAP Enhanced Images\n{'='*80}")
    yolo_output = OUTPUT_BASE / 'ULAP' / 'yolo'
    ulap_metrics = run_yolo_detection(YOLO_MODEL, enhanced_dir, DATA_YAML, yolo_output, save_viz=True)

    if not ulap_metrics:
        print("✗ Failed to run YOLO on ULAP enhanced images.")
        return

    # Step 4: Create Comparisons
    print(f"\n{'='*80}\nSTEP 4: CREATING VISUAL COMPARISONS\n{'='*80}")

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
    ulap_viz_dir = ulap_metrics.get('visualization_dir')

    if original_viz_dir and ulap_viz_dir:
        comparison_dir = OUTPUT_BASE / 'ULAP' / 'comparisons'
        create_ulap_comparisons(
            raw_images_dir=str(raw_images_dir),
            labels_dir=str(labels_dir),
            enhanced_viz_dir=ulap_viz_dir,
            original_viz_dir=original_viz_dir,
            output_dir=str(comparison_dir),
            class_names=class_names
        )
    else:
        print("⚠️  Visualizations not found, skipping comparisons")

    # Step 5: Generate Report
    print(f"\n{'='*80}\nSTEP 5: GENERATING REPORT\n{'='*80}")

    report = [
        "# ULAP + YOLO Detection Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Model | mAP50 | mAP50-95 | Precision | Recall | F1 |",
        "|-------|-------|----------|-----------|--------|----| "
    ]

    om = original_metrics
    um = ulap_metrics
    report.append(f"| Original | {om.get('mAP50', 0):.4f} | {om.get('mAP50-95', 0):.4f} | {om.get('precision', 0):.4f} | {om.get('recall', 0):.4f} | {om.get('f1', 0):.4f} |")
    report.append(f"| ULAP Enhanced | {um.get('mAP50', 0):.4f} | {um.get('mAP50-95', 0):.4f} | {um.get('precision', 0):.4f} | {um.get('recall', 0):.4f} | {um.get('f1', 0):.4f} |")

    report.extend(["", "## Improvement Analysis", ""])
    improvements = {}
    for key in ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']:
        orig_val = om.get(key, 0)
        ulap_val = um.get(key, 0)
        if orig_val > 0:
            improvement = ((ulap_val - orig_val) / orig_val) * 100
            improvements[key] = improvement

    report.extend([
        "| Metric | Original | ULAP Enhanced | Improvement |",
        "|--------|----------|---------------|-------------|"
    ])

    for key in ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']:
        imp = improvements.get(key, 0)
        imp_str = f"{imp:+.2f}%" if abs(imp) > 0 else "0.00%"
        status = "✓" if imp > 0 else "✗"
        report.append(f"| {key} | {om.get(key, 0):.4f} | {um.get(key, 0):.4f} | {imp_str} {status} |")

    report_file = OUTPUT_BASE / 'ULAP_YOLO_Report.md'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    print(f"\n✓ Report: {report_file}")

    # Save results
    results = {'Original': original_metrics, 'ULAP': ulap_metrics}
    with open(OUTPUT_BASE / 'results.yaml', 'w') as f:
        yaml.dump(results, f)

    print(f"\n{'='*80}\n✓✓✓ PIPELINE COMPLETED! ✓✓✓\n{'='*80}")
    print(f"\nResults saved to: {OUTPUT_BASE}")
    print(f"Report: {report_file}")
    print(f"Comparison images: {OUTPUT_BASE / 'ULAP' / 'comparisons'}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()