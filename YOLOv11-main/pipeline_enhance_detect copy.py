"""
Pipeline: Image Enhancement → YOLO Detection → 3-Way Visual Comparison
========================================================================

Optimized Workflow:
1. Enhance images with LU2Net (PyTorch or ONNX)
2. Run YOLO detection on ENHANCED images → Save detection results with bounding boxes
3. Run YOLO detection on ORIGINAL images → Save detection results with bounding boxes
4. Create 3-way side-by-side comparison images:
   [Raw Image + Ground Truth (Green)] | [Enhanced + Detection] | [Original + Detection]
5. Calculate and compare metrics between enhanced and original approaches

Features:
- Efficient processing with minimal memory footprint
- Saves detection visualizations for both enhanced and original images
- Creates 3-way visual comparison for comprehensive analysis
- Ground truth boxes (green) drawn on raw images from YOLO label files
- Easy visual comparison: GT vs Enhanced Detection vs Original Detection
- Comprehensive metrics comparison with improvement percentages
- Easy visual identification of false positives, false negatives, and improvements

Output Structure:
detection_results_{model_name}/
├── enhanced/
│   ├── metrics.yaml
│   └── visualizations/          # Enhanced images with bounding boxes
├── original/
│   ├── metrics.yaml
│   └── visualizations/          # Original images with bounding boxes
├── comparisons/                  # 3-way side-by-side comparisons
│   ├── comparison_image1.jpg    # [Raw | Enhanced+Det | Original+Det]
│   └── ...
└── metrics_comparison.yaml       # Detailed metrics comparison

Usage:
    # Full pipeline with LU2Net PyTorch
    python pipeline_enhance_detect.py --enhance-model pytorch --model-name lu2net

    # Full pipeline with LU2Net ONNX
    python pipeline_enhance_detect.py --enhance-model onnx --model-name lu2net_onnx

    # Skip enhancement if already done
    python pipeline_enhance_detect.py --skip-enhancement --model-name lu2net

    # Custom paths
    python pipeline_enhance_detect.py \
        --enhance-model pytorch \
        --test-dir /path/to/test \
        --yolo-model /path/to/best.pt \
        --imgsz 640 \
        --device 0

Author: Claude Code (Optimized)
Date: 2025-12-13
"""

import os
import sys
import argparse
import yaml
import time
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch

# Import YOLO
from ultralytics import YOLO

# Import LU2Net enhancement models
sys.path.insert(0, '/home/ndpthao/eject/IMPLEMENTATION/LU2Net-master')
import LU2Net
import metrics as lu2net_metrics


# ============================================================================
# Image Enhancement Module
# ============================================================================

class ImageEnhancer:
    """Base class for image enhancement"""
    def __init__(self):
        pass

    def enhance(self, image_path):
        """
        Enhance single image

        Args:
            image_path: Path to input image

        Returns:
            enhanced_image: Enhanced image (numpy array, RGB, uint8)
        """
        raise NotImplementedError


class LU2NetPyTorchEnhancer(ImageEnhancer):
    """LU2Net PyTorch model enhancer"""
    def __init__(self, model_path='/home/ndpthao/eject/IMPLEMENTATION/LU2Net-master/LightUNet_170.pth'):
        super().__init__()
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.load_model()

    def load_model(self):
        print(f"\n{'='*70}")
        print(f"Loading LU2Net PyTorch Model")
        print(f"{'='*70}")
        print(f"Model path: {self.model_path}")
        print(f"Device: {self.device}")

        self.model = LU2Net.LU2Net()
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        # Model complexity
        flops_info = lu2net_metrics.calculate_flops(self.model, input_shape=(1, 3, 256, 256), device=self.device)
        memory_mb = lu2net_metrics.get_memory_usage(self.model, device=self.device)

        print(f"✓ Model loaded successfully!")
        print(f"  FLOPs: {flops_info['flops']:.3f} GFLOPs")
        print(f"  MACs: {flops_info['macs']:.3f} GMACs")
        print(f"  Parameters: {flops_info['params']:.3f} M")
        print(f"  Memory: {memory_mb:.2f} MB")
        print(f"{'='*70}\n")

    def enhance(self, image_path):
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        original_size = (image.shape[1], image.shape[0])  # (width, height)

        # Preprocess
        resized = cv2.resize(image, (256, 256))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(tensor)
            output = torch.clamp(output, 0.0, 1.0)

        # Postprocess
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_uint8 = (output_np * 255.0).astype(np.uint8)

        # Resize back to original size
        output_resized = cv2.resize(output_uint8, original_size)

        return output_resized  # RGB format


class LU2NetONNXEnhancer(ImageEnhancer):
    """LU2Net ONNX model enhancer"""
    def __init__(self, model_path='/home/ndpthao/eject/IMPLEMENTATION/LU2Net-master/LightUNet_170.onnx'):
        super().__init__()
        import onnxruntime as ort

        self.model_path = model_path
        self.device = 'cpu'  # ONNX usually runs on CPU

        print(f"\n{'='*70}")
        print(f"Loading LU2Net ONNX Model")
        print(f"{'='*70}")
        print(f"Model path: {self.model_path}")
        print(f"Device: {self.device}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4

        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        model_size = os.path.getsize(model_path) / (1024 ** 2)
        print(f"✓ Model loaded successfully!")
        print(f"  Input name: {self.input_name}")
        print(f"  Output name: {self.output_name}")
        print(f"  Model size: {model_size:.2f} MB")

        # Warm-up
        print(f"Running warm-up...")
        dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
        _ = self.session.run([self.output_name], {self.input_name: dummy_input})
        print(f"✓ Warm-up completed!")
        print(f"{'='*70}\n")

    def enhance(self, image_path):
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        original_size = (image.shape[1], image.shape[0])

        # Preprocess
        resized = cv2.resize(image, (256, 256))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # Inference
        output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        output = np.clip(output, 0.0, 1.0)

        # Postprocess
        output_np = output.squeeze(0).transpose(1, 2, 0)
        output_uint8 = (output_np * 255.0).astype(np.uint8)

        # Resize back
        output_resized = cv2.resize(output_uint8, original_size)

        return output_resized  # RGB format


# ============================================================================
# Enhancement Pipeline
# ============================================================================

def enhance_dataset(enhancer, input_dir, output_dir, copy_labels=True):
    """
    Enhance all images in input directory and save to output directory

    Args:
        enhancer: ImageEnhancer instance
        input_dir: Input images directory (URPC2020/test/images)
        output_dir: Output directory for enhanced images
        copy_labels: Whether to copy label files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_images_dir = output_path / 'images'
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in input_path.glob('*') if f.suffix.lower() in image_exts]

    print(f"\n{'='*70}")
    print(f"STEP 1: ENHANCING DATASET")
    print(f"{'='*70}")
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_images_dir}")
    print(f"Total images: {len(image_files)}")
    print(f"{'='*70}\n")

    # Enhance images
    for img_file in tqdm(image_files, desc="Enhancing images"):
        try:
            # Enhance image
            enhanced_rgb = enhancer.enhance(img_file)

            # Save enhanced image (convert back to BGR for cv2)
            output_file = output_images_dir / img_file.name
            enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_file), enhanced_bgr)

        except Exception as e:
            print(f"\nError processing {img_file.name}: {e}")
            continue

    # Copy labels if requested
    if copy_labels:
        labels_input_dir = input_path.parent / 'labels'
        if labels_input_dir.exists():
            labels_output_dir = output_path / 'labels'
            labels_output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nCopying label files from {labels_input_dir} to {labels_output_dir}...")
            label_files = list(labels_input_dir.glob('*.txt'))
            for label_file in tqdm(label_files, desc="Copying labels"):
                shutil.copy2(label_file, labels_output_dir / label_file.name)
            print(f"✓ Copied {len(label_files)} label files")

    print(f"\n{'='*70}")
    print(f"✓ Enhancement complete!")
    print(f"  Enhanced images saved to: {output_images_dir}")
    print(f"{'='*70}\n")


# ============================================================================
# YOLO Detection with Visualization
# ============================================================================

def run_yolo_detection_with_visualization(yolo_model_path, data_yaml, test_dir, imgsz=256,
                                         batch=1, device='0', project='urpc_test',
                                         name='detection', save_viz=True):
    """
    Run YOLO detection and save visualizations with bounding boxes

    Args:
        yolo_model_path: Path to YOLO model (.pt file)
        data_yaml: Path to data.yaml configuration
        test_dir: Directory containing test images and labels
        imgsz: Image size for inference
        batch: Batch size
        device: Device ID ('0', 'cpu', etc.)
        project: Output project directory
        name: Run name
        save_viz: Save visualization images with bounding boxes

    Returns:
        dict: Detection metrics and output paths
    """
    print(f"\n{'='*70}")
    print(f"RUNNING YOLO DETECTION: {name}")
    print(f"{'='*70}")
    print(f"Model: {yolo_model_path}")
    print(f"Test directory: {test_dir}")
    print(f"Image size: {imgsz}")
    print(f"Device: {device}")
    print(f"Save visualization: {save_viz}")
    print(f"{'='*70}\n")

    # Update data.yaml to point to test directory
    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)

    data_cfg['test'] = str(test_dir)

    # Create temporary data.yaml
    output_path = Path(project) / name
    output_path.mkdir(parents=True, exist_ok=True)

    temp_yaml = output_path / 'data_temp.yaml'
    with open(temp_yaml, 'w') as f:
        yaml.dump(data_cfg, f)

    # Load YOLO model
    model = YOLO(yolo_model_path)

    # Run validation (calculates all metrics)
    results = model.val(
        data=str(temp_yaml),
        split='test',
        imgsz=imgsz,
        batch=batch,
        device=device,
        save_json=True,
        save_txt=True,
        save_conf=True,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
        plots=save_viz  # Save plots including confusion matrix, PR curves
    )

    # Extract metrics
    metrics_dict = {}

    box = getattr(results, 'box', None)
    if box:
        metrics_dict['mAP50-95'] = float(box.map)  # mAP@0.5:0.95
        metrics_dict['mAP50'] = float(box.map50)   # mAP@0.5
        metrics_dict['mAP75'] = float(box.map75)   # mAP@0.75

        mp = getattr(box, 'mp', None)  # mean precision
        mr = getattr(box, 'mr', None)  # mean recall

        if mp is not None:
            metrics_dict['precision'] = float(mp)
        if mr is not None:
            metrics_dict['recall'] = float(mr)

        # F1 score
        if mp is not None and mr is not None and (mp + mr) > 0:
            metrics_dict['f1'] = float(2 * mp * mr / (mp + mr))

        # Per-class mAP
        if hasattr(box, 'maps'):
            metrics_dict['per_class_mAP'] = box.maps.tolist() if hasattr(box.maps, 'tolist') else list(box.maps)

    # Speed metrics
    if hasattr(results, 'speed'):
        metrics_dict['speed'] = dict(results.speed)

    # Print metrics
    print(f"\n{'='*70}")
    print(f"DETECTION METRICS - {name}")
    print(f"{'='*70}")
    print(f"mAP@0.5:0.95 : {metrics_dict.get('mAP50-95', 'N/A'):.4f}")
    print(f"mAP@0.5      : {metrics_dict.get('mAP50', 'N/A'):.4f}")
    print(f"mAP@0.75     : {metrics_dict.get('mAP75', 'N/A'):.4f}")
    print(f"Precision    : {metrics_dict.get('precision', 'N/A'):.4f}")
    print(f"Recall       : {metrics_dict.get('recall', 'N/A'):.4f}")
    print(f"F1 Score     : {metrics_dict.get('f1', 'N/A'):.4f}")

    if 'speed' in metrics_dict:
        print(f"\nSpeed (ms/image):")
        for k, v in metrics_dict['speed'].items():
            print(f"  {k:15s}: {v:.2f}")

    print(f"{'='*70}\n")

    # Save metrics to file
    metrics_file = output_path / 'metrics.yaml'
    with open(metrics_file, 'w') as f:
        yaml.dump(metrics_dict, f, default_flow_style=False)
    print(f"✓ Metrics saved to: {metrics_file}")

    # Now run predictions to save visualization images with bounding boxes
    if save_viz:
        print(f"\n{'='*70}")
        print(f"SAVING DETECTION VISUALIZATIONS")
        print(f"{'='*70}\n")

        viz_output = output_path / 'visualizations'
        viz_output.mkdir(parents=True, exist_ok=True)

        # Get test images
        test_images_dir = Path(test_dir) / 'images'
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in test_images_dir.glob('*') if f.suffix.lower() in image_exts]

        # Run prediction on each image to get visualization
        for img_file in tqdm(image_files, desc="Saving detection visualizations"):
            try:
                # Run prediction
                results_pred = model.predict(
                    source=str(img_file),
                    imgsz=imgsz,
                    conf=0.25,
                    device=device,
                    save=False,
                    verbose=False
                )

                # Get annotated image
                if len(results_pred) > 0:
                    result = results_pred[0]
                    annotated_img = result.plot()  # Returns BGR image with bounding boxes

                    # Save annotated image
                    output_file = viz_output / img_file.name
                    cv2.imwrite(str(output_file), annotated_img)

            except Exception as e:
                print(f"\nError visualizing {img_file.name}: {e}")
                continue

        print(f"\n✓ Detection visualizations saved to: {viz_output}")
        metrics_dict['visualization_dir'] = str(viz_output)

    metrics_dict['output_dir'] = str(output_path)

    return metrics_dict


# ============================================================================
# Visual Comparison
# ============================================================================

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
    print(f"\n{'='*70}")
    print(f"CREATING 3-WAY VISUAL COMPARISONS")
    print(f"{'='*70}")
    print(f"Raw images dir:  {raw_images_dir}")
    print(f"Labels dir:      {labels_dir}")
    print(f"Enhanced dir:    {enhanced_viz_dir}")
    print(f"Original dir:    {original_viz_dir}")
    print(f"Output dir:      {output_dir}")
    print(f"{'='*70}\n")

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
            print(f"\nWarning: Original detection file not found for {img_name}, skipping...")
            continue

        if not raw_file.exists():
            print(f"\nWarning: Raw image file not found for {img_name}, skipping...")
            continue

        try:
            # Read all three images
            raw_img = cv2.imread(str(raw_file))
            enhanced_img = cv2.imread(str(enhanced_file))
            original_img = cv2.imread(str(original_file))

            if raw_img is None or enhanced_img is None or original_img is None:
                print(f"\nError reading images for {img_name}, skipping...")
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

    print(f"\n{'='*70}")
    print(f"✓ Created {comparison_count} 3-way comparison images")
    print(f"  Format: [Raw+GT] | [Enhanced+Detection] | [Original+Detection]")
    print(f"  Ground truth boxes: Green")
    print(f"  Saved to: {output_path}")
    print(f"{'='*70}\n")


# ============================================================================
# Metrics Comparison
# ============================================================================

def compare_metrics(enhanced_metrics, original_metrics, output_file):
    """
    Compare metrics between enhanced and original detections

    Args:
        enhanced_metrics: Metrics dict from enhanced images
        original_metrics: Metrics dict from original images
        output_file: Output file path for comparison table
    """
    print(f"\n{'='*70}")
    print(f"METRICS COMPARISON")
    print(f"{'='*70}\n")

    # Metrics to compare
    metric_keys = ['mAP50-95', 'mAP50', 'mAP75', 'precision', 'recall', 'f1']

    # Print comparison table
    print(f"{'Metric':<15} {'Enhanced':<12} {'Original':<12} {'Improvement':<12}")
    print(f"{'-'*55}")

    comparison_results = {}
    for key in metric_keys:
        enh_val = enhanced_metrics.get(key, 0.0)
        orig_val = original_metrics.get(key, 0.0)

        if orig_val > 0:
            improvement = ((enh_val - orig_val) / orig_val) * 100
        else:
            improvement = 0.0

        comparison_results[key] = {
            'enhanced': enh_val,
            'original': orig_val,
            'improvement_pct': improvement
        }

        # Format improvement with + or - sign
        imp_str = f"{improvement:+.2f}%"
        print(f"{key:<15} {enh_val:<12.4f} {orig_val:<12.4f} {imp_str:<12}")

    print(f"{'='*70}\n")

    # Save comparison to YAML
    with open(output_file, 'w') as f:
        yaml.dump(comparison_results, f, default_flow_style=False)
    print(f"✓ Comparison saved to: {output_file}\n")

    return comparison_results


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optimized Pipeline: Enhancement → Detection → Comparison")

    # Enhancement options
    parser.add_argument('--enhance-model', type=str, choices=['pytorch', 'onnx'], default='pytorch',
                       help='Enhancement model type: pytorch (LU2Net PyTorch) or onnx (LU2Net ONNX)')
    parser.add_argument('--model-name', type=str, default='lu2net',
                       help='Model name for output directory naming')
    parser.add_argument('--pytorch-model', type=str,
                       default='/home/ndpthao/eject/IMPLEMENTATION/LU2Net-master/LightUNet_170.pth',
                       help='Path to PyTorch model')
    parser.add_argument('--onnx-model', type=str,
                       default='/home/ndpthao/eject/IMPLEMENTATION/LU2Net-master/LightUNet_170.onnx',
                       help='Path to ONNX model')

    # Dataset paths
    parser.add_argument('--test-dir', type=str,
                       default='/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/test',
                       help='Test dataset directory (contains images/ and labels/)')
    parser.add_argument('--output-base', type=str,
                       default='/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020',
                       help='Base output directory for enhanced images')

    # YOLO options
    parser.add_argument('--yolo-model', type=str,
                       default='/home/ndpthao/eject/runs/detect/train6/weights/best.pt',
                       help='Path to YOLO model (.pt file)')
    parser.add_argument('--data-yaml', type=str, default='/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml',
                       help='Path to data.yaml (optional, auto-detected from test-dir parent if not specified)')
    parser.add_argument('--imgsz', type=int, default=256, help='YOLO inference image size')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--device', type=str, default='0', help='Device (0, cpu, etc.)')

    # Pipeline options
    parser.add_argument('--skip-enhancement', action='store_true',
                       help='Skip enhancement step (assumes enhanced images already exist)')
    parser.add_argument('--compare', action='store_true', default=True,
                       help='Create visual comparison images (default: True)')

    args = parser.parse_args()

    # Output directories
    enhanced_dir = Path(args.output_base) / f'test_enh_{args.model_name}'
    project_base = f'detection_results_{args.model_name}'

    print(f"\n{'='*70}")
    print(f"OPTIMIZED PIPELINE CONFIGURATION")
    print(f"{'='*70}")
    print(f"Enhancement model: {args.enhance_model}")
    print(f"Model name: {args.model_name}")
    print(f"Test directory (original): {args.test_dir}")
    print(f"Enhanced output: {enhanced_dir}")
    print(f"YOLO model: {args.yolo_model}")
    print(f"Create comparisons: {args.compare}")
    print(f"{'='*70}\n")

    # ========================================================================
    # STEP 1: Image Enhancement
    # ========================================================================
    if not args.skip_enhancement:
        # Create enhancer
        if args.enhance_model == 'pytorch':
            enhancer = LU2NetPyTorchEnhancer(args.pytorch_model)
        elif args.enhance_model == 'onnx':
            enhancer = LU2NetONNXEnhancer(args.onnx_model)

        # Enhance dataset
        test_images_dir = Path(args.test_dir) / 'images'
        enhance_dataset(enhancer, test_images_dir, enhanced_dir, copy_labels=True)
    else:
        print(f"\n{'='*70}")
        print(f"STEP 1: ENHANCEMENT (SKIPPED)")
        print(f"{'='*70}")
        print(f"Using existing enhanced images in: {enhanced_dir}\n")

    # ========================================================================
    # STEP 2: YOLO Detection on ENHANCED Images
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"STEP 2: DETECTION ON ENHANCED IMAGES")
    print(f"{'='*70}\n")

    enhanced_metrics = run_yolo_detection_with_visualization(
        yolo_model_path=args.yolo_model,
        data_yaml=args.data_yaml,
        test_dir=str(enhanced_dir),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=project_base,
        name='enhanced',
        save_viz=True
    )

    # ========================================================================
    # STEP 3: YOLO Detection on ORIGINAL Images
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"STEP 3: DETECTION ON ORIGINAL IMAGES")
    print(f"{'='*70}\n")

    original_metrics = run_yolo_detection_with_visualization(
        yolo_model_path=args.yolo_model,
        data_yaml=args.data_yaml,
        test_dir=str(args.test_dir),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=project_base,
        name='original',
        save_viz=True
    )

    # ========================================================================
    # STEP 4: Create 3-Way Visual Comparisons
    # ========================================================================
    if args.compare:
        print(f"\n{'='*70}")
        print(f"STEP 4: CREATING 3-WAY VISUAL COMPARISONS")
        print(f"{'='*70}\n")

        comparison_dir = Path(project_base) / 'comparisons'

        # Path to raw images and ground truth labels
        raw_images_dir = Path(args.test_dir) / 'images'
        labels_dir = Path(args.test_dir) / 'labels'

        # Load class names from data.yaml if available
        class_names = None
        if os.path.exists(args.data_yaml):
            try:
                with open(args.data_yaml, 'r') as f:
                    data_cfg = yaml.safe_load(f)
                    class_names = data_cfg.get('names', None)
                    if class_names:
                        print(f"[Info] Loaded {len(class_names)} class names from data.yaml")
            except Exception as e:
                print(f"[Warning] Could not load class names from {args.data_yaml}: {e}")

        create_comparison_images(
            raw_images_dir=str(raw_images_dir),
            labels_dir=str(labels_dir),
            enhanced_viz_dir=enhanced_metrics['visualization_dir'],
            original_viz_dir=original_metrics['visualization_dir'],
            output_dir=str(comparison_dir),
            class_names=class_names
        )

    # ========================================================================
    # STEP 5: Compare Metrics
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"STEP 5: METRICS COMPARISON")
    print(f"{'='*70}\n")

    comparison_file = Path(project_base) / 'metrics_comparison.yaml'
    compare_metrics(enhanced_metrics, original_metrics, comparison_file)

    # ========================================================================
    # Final Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"✓✓✓ PIPELINE COMPLETE! ✓✓✓")
    print(f"{'='*70}")
    print(f"\nResults Summary:")
    print(f"  Enhanced images:        {enhanced_dir}")
    print(f"  Enhanced detections:    {enhanced_metrics['output_dir']}")
    print(f"  Original detections:    {original_metrics['output_dir']}")
    if args.compare:
        print(f"  Comparison images:      {comparison_dir}")
    print(f"  Metrics comparison:     {comparison_file}")
    print(f"\nKey Improvements (Enhanced vs Original):")
    print(f"  mAP@0.5:0.95: {enhanced_metrics.get('mAP50-95', 0):.4f} vs {original_metrics.get('mAP50-95', 0):.4f}")
    print(f"  mAP@0.5:     {enhanced_metrics.get('mAP50', 0):.4f} vs {original_metrics.get('mAP50', 0):.4f}")
    print(f"  Precision:   {enhanced_metrics.get('precision', 0):.4f} vs {original_metrics.get('precision', 0):.4f}")
    print(f"  Recall:      {enhanced_metrics.get('recall', 0):.4f} vs {original_metrics.get('recall', 0):.4f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
