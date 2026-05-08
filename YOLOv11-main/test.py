import argparse
import os
from pathlib import Path
import yaml
from ultralytics import YOLO

# Import custom metrics module
import metrics


def load_data_cfg(data_yaml, test_dir=None):
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if test_dir:
        cfg["test"] = test_dir
    return cfg


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 test/inference on held-out test set")
    parser.add_argument(
        "--model",
        type=str,
        # default="/home/ndpthao/eject/IMPLEMENTATION/YOLOv11-main/urpc_train/yolov11n_finetune/weights/best.pt",
        default="/home/ndpthao/eject/IMPLEMENTATION/YOLOv11-main/yolo11n.pt",
        help="Checkpoint path (best.pt/last.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml",
        help="Data yaml file (will be used for names and paths)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/test",
        help="Override test images directory (requires matching labels subfolder)",
    )
    parser.add_argument("--imgsz", type=int, default=256, help="Inference image size")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for val")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id or 'cpu'")
    parser.add_argument("--project", type=str, default="urpc_test", help="Output project dir")
    parser.add_argument("--name", type=str, default="yolov11n_test", help="Run name")
    args = parser.parse_args()

    data_cfg = load_data_cfg(args.data, args.test_dir)

    # Load class names from data.yaml
    class_names = data_cfg.get('names', None)
    if class_names is None:
        class_names = []
    print(f"\nDataset: {args.data}")
    print(f"Classes: {class_names if class_names else 'Not specified'}")
    print(f"Number of classes: {len(class_names)}\n")

    # Load YOLO model
    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    print(f"✓ Model loaded successfully!\n")

    # Run validation on test split to get metrics
    print(f"{'='*70}")
    print(f"RUNNING YOLO VALIDATION ON TEST SET")
    print(f"{'='*70}")
    print(f"Test directory: {args.test_dir}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")

    # YOLO requires YAML file path, not dict
    # Save modified config to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_yaml:
        yaml.dump(data_cfg, tmp_yaml, default_flow_style=False)
        temp_yaml_path = tmp_yaml.name

    try:
        results = model.val(
            data=temp_yaml_path,
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            save_json=True,  # coco-style json if applicable
            project=args.project,
            name=args.name,
            exist_ok=True,
            verbose=True
        )
    finally:
        # Clean up temporary file
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)

    # Extract metrics using custom metrics module
    detection_metrics = metrics.extract_yolo_metrics(results)

    # Print metrics using custom pretty-print function
    metrics.print_detection_metrics(detection_metrics, class_names=class_names)

    # Save metrics to YAML file
    output_dir = Path(args.project) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / 'detection_metrics.yaml'

    with open(metrics_file, 'w') as f:
        yaml.dump(detection_metrics, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Metrics saved to: {metrics_file}")
    print(f"✓ Artifacts saved to: {output_dir}\n")

    # Print summary
    print(f"{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Test set: {args.test_dir}")
    print(f"Results directory: {output_dir}")
    print(f"\nKey Metrics:")
    print(f"  mAP@0.5:0.95 : {detection_metrics.get('mAP50-95', 0.0):.4f}")
    print(f"  mAP@0.5      : {detection_metrics.get('mAP50', 0.0):.4f}")
    print(f"  Precision    : {detection_metrics.get('precision', 0.0):.4f}")
    print(f"  Recall       : {detection_metrics.get('recall', 0.0):.4f}")
    print(f"  F1-Score     : {detection_metrics.get('f1', 0.0):.4f}")
    if 'fps' in detection_metrics:
        print(f"  FPS          : {detection_metrics['fps']:.2f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
