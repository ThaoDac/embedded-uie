#!/usr/bin/env python3
"""
UIE Deep Learning Models Evaluation Script
===========================================
This script runs multiple deep learning underwater image enhancement methods
and evaluates their performance on a given dataset.

Supported Methods:
- FUnIE-GAN (PyTorch): Fast Underwater Image Enhancement GAN
- FUnIE-GAN (ONNX): ONNX optimized version
- HCLR-Net: Hierarchical Context Learning and Refinement Network
- LU2Net (PyTorch): Lightweight U-Net for Underwater Images
- LU2Net (ONNX): ONNX optimized version
- PGHS (PyTorch): Physical-Guided Hybrid System
- PGHS (ONNX): ONNX optimized version
- UIR-PolyKernel: Underwater Image Restoration with Polynomial Kernels
- WaterFormer: Transformer-based Underwater Image Enhancement

Usage:
    python UIE_DL.py --input <input_dir> --output <output_dir> --gt <gt_dir>
"""

import os
import sys
import time
import argparse
import subprocess
import re
from pathlib import Path
from datetime import datetime


class DLModelEvaluator:
    """Evaluator for deep learning UIE models"""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.models = {
            'FUnIE-GAN-PyTorch': {
                'path': os.path.join(base_dir, 'FUnIE-GAN-main/app_console_pytorch.py'),
                'name': 'FUnIE-GAN-PyTorch',
                'full_name': 'Fast Underwater Image Enhancement GAN (PyTorch)',
                'type': 'Deep Learning - GAN',
                'framework': 'PyTorch'
            },
            'FUnIE-GAN-ONNX': {
                'path': os.path.join(base_dir, 'FUnIE-GAN-main/FUnIE-GAN-origin/PyTorch/inference_onnx.py'),
                'name': 'FUnIE-GAN-ONNX',
                'full_name': 'Fast Underwater Image Enhancement GAN (ONNX)',
                'type': 'Deep Learning - GAN',
                'framework': 'ONNX Runtime'
            },
            'HCLR-Net': {
                'path': os.path.join(base_dir, 'HCLR-Net-main/app_console.py'),
                'name': 'HCLR-Net',
                'full_name': 'Hierarchical Context Learning and Refinement Network',
                'type': 'Deep Learning - CNN',
                'framework': 'PyTorch'
            },
            'LU2Net-PyTorch': {
                'path': os.path.join(base_dir, 'LU2Net-master/app_console.py'),
                'name': 'LU2Net-PyTorch',
                'full_name': 'Lightweight U-Net for Underwater Images (PyTorch)',
                'type': 'Deep Learning - U-Net',
                'framework': 'PyTorch'
            },
            'LU2Net-ONNX': {
                'path': os.path.join(base_dir, 'LU2Net-master/inference_onnx.py'),
                'name': 'LU2Net-ONNX',
                'full_name': 'Lightweight U-Net for Underwater Images (ONNX)',
                'type': 'Deep Learning - U-Net',
                'framework': 'ONNX Runtime'
            },
            'PGHS-PyTorch': {
                'path': os.path.join(base_dir, 'PGHS-main/app_console.py'),
                'name': 'PGHS-PyTorch',
                'full_name': 'Physical-Guided Hybrid System (PyTorch)',
                'type': 'Deep Learning - Hybrid',
                'framework': 'PyTorch'
            },
            'PGHS-ONNX': {
                'path': os.path.join(base_dir, 'PGHS-main/inference_onnx.py'),
                'name': 'PGHS-ONNX',
                'full_name': 'Physical-Guided Hybrid System (ONNX)',
                'type': 'Deep Learning - Hybrid',
                'framework': 'ONNX Runtime'
            },
            'UIR-PolyKernel': {
                'path': os.path.join(base_dir, 'UIR-PolyKernel-main/app_console.py'),
                'name': 'UIR-PolyKernel',
                'full_name': 'Underwater Image Restoration with Polynomial Kernels',
                'type': 'Deep Learning - CNN',
                'framework': 'PyTorch'
            },
            'WaterFormer': {
                'path': os.path.join(base_dir, 'WaterFormer-master/app_console.py'),
                'name': 'WaterFormer',
                'full_name': 'Transformer-based Underwater Image Enhancement',
                'type': 'Deep Learning - Transformer',
                'framework': 'PyTorch'
            }
        }
        self.results = {}

    def check_models_exist(self):
        """Check if all model scripts exist"""
        missing = []
        available = []
        for model_key, model_info in self.models.items():
            if not os.path.exists(model_info['path']):
                missing.append(f"{model_key}: {model_info['path']}")
            else:
                available.append(model_key)

        if missing:
            print("WARNING: The following model scripts are missing and will be skipped:")
            for m in missing:
                print(f"  - {m}")
            print(f"\nAvailable models: {len(available)}/{len(self.models)}")

        return available

    def run_model(self, model_key, input_dir, output_dir, gt_dir=None):
        """
        Run a single model and capture its output

        Args:
            model_key: Key identifying the model
            input_dir: Input directory path
            output_dir: Output directory path
            gt_dir: Ground truth directory path (optional)

        Returns:
            dict: Parsed results from model execution
        """
        model_info = self.models[model_key]
        model_path = model_info['path']
        model_output_dir = os.path.join(output_dir, model_key)

        # Prepare command
        cmd = [
            sys.executable,
            model_path,
            '--input', input_dir,
            '--output', model_output_dir
        ]

        if gt_dir:
            cmd.extend(['--gt', gt_dir])

        print(f"\n{'='*80}")
        print(f"Running {model_info['name']} - {model_info['full_name']}")
        print(f"Framework: {model_info['framework']}")
        print(f"{'='*80}")
        print(f"Command: {' '.join(cmd)}")

        # Run the model
        # IMPORTANT: Change to model directory before running (some models need relative paths for config/weights)
        model_dir = os.path.dirname(model_path)
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout for DL models
                cwd=model_dir  # Run from model directory
            )
            elapsed_time = time.time() - start_time

            if result.returncode != 0:
                print(f"ERROR: {model_key} failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr[:1000]}")  # Print first 1000 chars of error
                return None

            # Parse output
            output_text = result.stdout
            print(output_text)  # Print to console for monitoring

            # Extract metrics from output
            metrics = self.parse_output(output_text, model_key)
            metrics['execution_time'] = elapsed_time
            metrics['model_name'] = model_info['name']
            metrics['full_name'] = model_info['full_name']
            metrics['type'] = model_info['type']
            metrics['framework'] = model_info['framework']

            return metrics

        except subprocess.TimeoutExpired:
            print(f"ERROR: {model_key} execution timed out (>2 hours)")
            return None
        except Exception as e:
            print(f"ERROR: Failed to run {model_key}: {str(e)}")
            return None

    def parse_output(self, output_text, model_key):
        """
        Parse model output to extract metrics

        Args:
            output_text: Raw output text from model execution
            model_key: Model identifier

        Returns:
            dict: Extracted metrics
        """
        metrics = {
            # Quality metrics
            'psnr': -1.0,
            'psnr_std': 0.0,
            'ssim': -1.0,
            'ssim_std': 0.0,
            'uciqe': -1.0,
            'uciqe_std': 0.0,
            'uiqm': -1.0,
            'uiqm_std': 0.0,
            'niqe': -1.0,
            'niqe_std': 0.0,

            # Performance metrics
            'avg_fps': 0.0,
            'min_fps': 0.0,
            'max_fps': 0.0,
            'inference_time': 0.0,
            'total_time': 0.0,
            'preprocess_time': 0.0,
            'postprocess_time': 0.0,

            # Resource metrics
            'model_size_mb': 0.0,
            'flops_gflops': 0.0,
            'macs_gmacs': 0.0,
            'params_m': 0.0,
            'memory_mb': 0.0,
            'ram_usage_mb': 0.0,
            'ram_usage_std': 0.0,
            'energy_joules': 0.0,
            'battery_wh': 0.0,

            # Processing info
            'total_images': 0,
            'processed_images': 0
        }

        # Regular expressions for metric extraction
        patterns = {
            # Quality metrics (with ±)
            'psnr': r'PSNR\s*:\s*([\d.]+)\s*±\s*([\d.]+)',
            'ssim': r'SSIM\s*:\s*([\d.]+)\s*±\s*([\d.]+)',
            'uciqe': r'UCIQE\s*:\s*([\d.]+)\s*±\s*([\d.]+)',
            'uiqm': r'UIQM\s*:\s*([\d.]+)\s*±\s*([\d.]+)',
            'niqe': r'NIQE\s*:\s*([\d.]+)\s*±\s*([\d.]+)',

            # FPS metrics
            'avg_fps': r'Avg FPS\s*:\s*([\d.]+)',
            'min_fps': r'Min FPS\s*:\s*([\d.]+)',
            'max_fps': r'Max FPS\s*:\s*([\d.]+)',

            # Timing metrics
            'inference_time': r'Avg Inference Time\s*:\s*([\d.]+)\s*s',
            'total_time': r'Avg Total Time\s*:\s*([\d.]+)\s*s',
            'preprocess_time': r'Preprocess\s*:\s*([\d.]+)\s*s',
            'postprocess_time': r'Postprocess\s*:\s*([\d.]+)\s*s',

            # Model complexity
            'model_size_mb': r'Model Size\s*:\s*([\d.]+)\s*MB',
            'flops_gflops': r'FLOPs \(GFLOPs\)\s*:\s*([\d.]+)',
            'macs_gmacs': r'MACs \(GMACs\)\s*:\s*([\d.]+)',
            'params_m': r'Parameters \(M\)\s*:\s*([\d.]+)',

            # Memory metrics
            'memory_mb': r'Model Memory \(MB\)\s*:\s*([\d.]+)',
            'ram_usage_mb': r'RAM Usage \(Profiler\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)',
            'energy_joules': r'Energy Consumption\s*:\s*([\d.]+)\s*J',
            'battery_wh': r'\((\d+\.\d+)\s*Wh\)',

            # Processing info
            'total_images': r'Total Images\s*:\s*(\d+)',
            'processed_images': r'Processed\s*:\s*(\d+)'
        }

        # Extract metrics
        for key, pattern in patterns.items():
            match = re.search(pattern, output_text)
            if match:
                if key in ['psnr', 'ssim', 'uciqe', 'uiqm', 'niqe', 'ram_usage_mb']:
                    # Metrics with standard deviation
                    metrics[key] = float(match.group(1))
                    metrics[f'{key}_std'] = float(match.group(2))
                else:
                    metrics[key] = float(match.group(1))

        return metrics

    def generate_markdown_report(self, output_path, input_dir, output_dir, gt_dir):
        """
        Generate comprehensive markdown report

        Args:
            output_path: Path to output markdown file
            input_dir: Input directory used for evaluation
            output_dir: Output directory used
            gt_dir: Ground truth directory used
        """
        report = []

        # Header
        report.append("# Deep Learning UIE Models Evaluation Report")
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("## Evaluation Configuration")
        report.append("")
        report.append(f"- **Input Directory:** `{input_dir}`")
        report.append(f"- **Output Directory:** `{output_dir}`")
        report.append(f"- **Ground Truth Directory:** `{gt_dir if gt_dir else 'N/A'}`")
        report.append("")

        # Summary table
        report.append("## Summary Table")
        report.append("")
        report.append("| Model | Full Name | Framework | Type | Images | Status |")
        report.append("|-------|-----------|-----------|------|--------|--------|")

        for model_key in self.models.keys():
            model_info = self.models[model_key]
            if model_key in self.results and self.results[model_key]:
                metrics = self.results[model_key]
                status = f"✓ ({metrics['processed_images']}/{metrics['total_images']})"
            else:
                status = "✗ Failed/Skipped"

            images = metrics.get('total_images', 0) if model_key in self.results and self.results[model_key] else 0
            report.append(f"| {model_info['name']} | {model_info['full_name']} | {model_info['framework']} | {model_info['type']} | {images} | {status} |")

        report.append("")

        # Detailed results for each model
        report.append("## Detailed Results")
        report.append("")

        for model_key in self.models.keys():
            if model_key not in self.results or not self.results[model_key]:
                continue

            metrics = self.results[model_key]
            model_info = self.models[model_key]

            report.append(f"### {model_info['name']} - {model_info['full_name']}")
            report.append("")
            report.append(f"**Type:** {model_info['type']} | **Framework:** {model_info['framework']}")
            report.append("")

            # Model Architecture
            report.append("#### Model Architecture")
            report.append("")
            report.append("| Metric | Value |")
            report.append("|--------|-------|")
            report.append(f"| Parameters | {metrics['params_m']:.3f} M |")
            report.append(f"| FLOPs | {metrics['flops_gflops']:.3f} GFLOPs |")
            report.append(f"| MACs | {metrics['macs_gmacs']:.3f} GMACs |")
            report.append(f"| Model Size | {metrics['model_size_mb']:.2f} MB |")
            report.append(f"| Model Memory | {metrics['memory_mb']:.2f} MB |")
            report.append("")

            # Image Quality Assessment (IQA)
            report.append("#### Image Quality Assessment (IQA)")
            report.append("")

            if gt_dir and metrics['psnr'] > 0:
                report.append("**Full-Reference Metrics:**")
                report.append("")
                report.append("| Metric | Value | Std Dev |")
                report.append("|--------|-------|---------|")
                report.append(f"| PSNR (dB) | {metrics['psnr']:.9f} | ± {metrics['psnr_std']:.9f} |")
                report.append(f"| SSIM | {metrics['ssim']:.9f} | ± {metrics['ssim_std']:.9f} |")
                report.append("")

            report.append("**No-Reference Metrics:**")
            report.append("")
            report.append("| Metric | Value | Std Dev | Note |")
            report.append("|--------|-------|---------|------|")

            if metrics['uciqe'] > 0:
                report.append(f"| UCIQE | {metrics['uciqe']:.9f} | ± {metrics['uciqe_std']:.9f} | Higher is better |")
            if metrics['uiqm'] > 0:
                report.append(f"| UIQM | {metrics['uiqm']:.9f} | ± {metrics['uiqm_std']:.9f} | Higher is better |")
            if metrics['niqe'] > 0:
                report.append(f"| NIQE | {metrics['niqe']:.9f} | ± {metrics['niqe_std']:.9f} | Lower is better |")

            report.append("")

            # Real-time Performance
            report.append("#### Real-Time Performance")
            report.append("")
            report.append("**FPS Metrics:**")
            report.append("")
            report.append("| Metric | Value |")
            report.append("|--------|-------|")
            report.append(f"| Average FPS | {metrics['avg_fps']:.2f} |")
            report.append(f"| Min FPS | {metrics['min_fps']:.2f} |")
            report.append(f"| Max FPS | {metrics['max_fps']:.2f} |")
            report.append("")

            report.append("**Timing Breakdown:**")
            report.append("")
            report.append("| Stage | Time (s) |")
            report.append("|-------|----------|")
            report.append(f"| Preprocessing | {metrics['preprocess_time']:.4f} |")
            report.append(f"| Inference | {metrics['inference_time']:.4f} |")
            report.append(f"| Postprocessing | {metrics['postprocess_time']:.4f} |")
            report.append(f"| **Total** | **{metrics['total_time']:.4f}** |")
            report.append("")

            # Device Performance
            report.append("#### Device Performance Metrics")
            report.append("")
            report.append("| Metric | Value |")
            report.append("|--------|-------|")
            report.append(f"| RAM Usage (Profiler) | {metrics['ram_usage_mb']:.2f} ± {metrics['ram_usage_std']:.2f} MB |")
            report.append(f"| Energy Consumption | {metrics['energy_joules']:.2f} J ({metrics['battery_wh']:.6f} Wh) |")
            report.append(f"| Total Execution Time | {metrics['execution_time']:.2f} s |")
            report.append("")

        # Comparative Analysis
        report.append("## Comparative Analysis")
        report.append("")

        # Quality comparison
        if gt_dir:
            report.append("### Quality Metrics Comparison (with Ground Truth)")
            report.append("")
            report.append("| Model | Framework | PSNR (dB) | SSIM | UCIQE | UIQM | NIQE |")
            report.append("|-------|-----------|-----------|------|-------|------|------|")

            for model_key in self.models.keys():
                if model_key in self.results and self.results[model_key]:
                    m = self.results[model_key]
                    psnr = f"{m['psnr']:.4f}" if m['psnr'] > 0 else "N/A"
                    ssim = f"{m['ssim']:.4f}" if m['ssim'] > 0 else "N/A"
                    uciqe = f"{m['uciqe']:.4f}" if m['uciqe'] > 0 else "N/A"
                    uiqm = f"{m['uiqm']:.4f}" if m['uiqm'] > 0 else "N/A"
                    niqe = f"{m['niqe']:.4f}" if m['niqe'] > 0 else "N/A"
                    report.append(f"| {m['model_name']} | {m['framework']} | {psnr} | {ssim} | {uciqe} | {uiqm} | {niqe} |")
            report.append("")

        # Model Architecture Comparison
        report.append("### Model Architecture Comparison")
        report.append("")
        report.append("| Model | Framework | Params (M) | FLOPs (G) | MACs (G) | Model Size (MB) |")
        report.append("|-------|-----------|------------|-----------|----------|-----------------|")

        for model_key in self.models.keys():
            if model_key in self.results and self.results[model_key]:
                m = self.results[model_key]
                report.append(f"| {m['model_name']} | {m['framework']} | {m['params_m']:.3f} | {m['flops_gflops']:.3f} | {m['macs_gmacs']:.3f} | {m['model_size_mb']:.2f} |")

        report.append("")

        # Performance comparison
        report.append("### Performance Metrics Comparison")
        report.append("")
        report.append("| Model | Framework | Avg FPS | Inference (s) | Total Time (s) | RAM (MB) | Energy (J) |")
        report.append("|-------|-----------|---------|---------------|----------------|----------|------------|")

        for model_key in self.models.keys():
            if model_key in self.results and self.results[model_key]:
                m = self.results[model_key]
                report.append(f"| {m['model_name']} | {m['framework']} | {m['avg_fps']:.2f} | {m['inference_time']:.4f} | {m['total_time']:.4f} | {m['ram_usage_mb']:.2f} | {m['energy_joules']:.2f} |")

        report.append("")

        # Conclusions
        report.append("## Conclusions")
        report.append("")
        report.append("### Best Performers")
        report.append("")

        # Find best performers
        valid_results = {k: v for k, v in self.results.items() if v}

        if valid_results:
            # Best quality (PSNR if available, else UCIQE)
            if any(m['psnr'] > 0 for m in valid_results.values()):
                best_psnr = max(valid_results.items(), key=lambda x: x[1]['psnr'] if x[1]['psnr'] > 0 else -999)
                report.append(f"- **Best PSNR:** {best_psnr[1]['model_name']} ({best_psnr[1]['psnr']:.4f} dB)")

            if any(m['uciqe'] > 0 for m in valid_results.values()):
                best_uciqe = max(valid_results.items(), key=lambda x: x[1]['uciqe'] if x[1]['uciqe'] > 0 else -999)
                report.append(f"- **Best UCIQE:** {best_uciqe[1]['model_name']} ({best_uciqe[1]['uciqe']:.4f})")

            # Best speed (FPS)
            if any(m['avg_fps'] > 0 for m in valid_results.values()):
                best_fps = max(valid_results.items(), key=lambda x: x[1]['avg_fps'])
                report.append(f"- **Fastest (FPS):** {best_fps[1]['model_name']} ({best_fps[1]['avg_fps']:.2f} FPS)")

            # Most efficient (lowest energy)
            if any(m['energy_joules'] > 0 for m in valid_results.values()):
                best_energy = min(valid_results.items(), key=lambda x: x[1]['energy_joules'] if x[1]['energy_joules'] > 0 else 999999)
                report.append(f"- **Most Energy Efficient:** {best_energy[1]['model_name']} ({best_energy[1]['energy_joules']:.2f} J)")

            # Smallest model
            if any(m['params_m'] > 0 for m in valid_results.values()):
                smallest = min(valid_results.items(), key=lambda x: x[1]['params_m'] if x[1]['params_m'] > 0 else 999999)
                report.append(f"- **Smallest Model:** {smallest[1]['model_name']} ({smallest[1]['params_m']:.3f} M params)")

            # Lowest memory
            if any(m['ram_usage_mb'] > 0 for m in valid_results.values()):
                best_ram = min(valid_results.items(), key=lambda x: x[1]['ram_usage_mb'] if x[1]['ram_usage_mb'] > 0 else 999999)
                report.append(f"- **Lowest RAM Usage:** {best_ram[1]['model_name']} ({best_ram[1]['ram_usage_mb']:.2f} MB)")

        report.append("")

        # Framework comparison
        report.append("### Framework Comparison (PyTorch vs ONNX)")
        report.append("")
        pytorch_models = [k for k, v in valid_results.items() if v['framework'] == 'PyTorch']
        onnx_models = [k for k, v in valid_results.items() if v['framework'] == 'ONNX Runtime']

        if pytorch_models and onnx_models:
            report.append(f"- **PyTorch Models:** {len(pytorch_models)}")
            avg_pytorch_fps = sum(valid_results[k]['avg_fps'] for k in pytorch_models) / len(pytorch_models)
            report.append(f"  - Average FPS: {avg_pytorch_fps:.2f}")

            report.append(f"- **ONNX Models:** {len(onnx_models)}")
            avg_onnx_fps = sum(valid_results[k]['avg_fps'] for k in onnx_models) / len(onnx_models)
            report.append(f"  - Average FPS: {avg_onnx_fps:.2f}")

            speedup = avg_onnx_fps / avg_pytorch_fps if avg_pytorch_fps > 0 else 0
            report.append(f"  - **ONNX Speedup:** {speedup:.2f}x")

        report.append("")
        report.append("---")
        report.append("")
        report.append("*This report was automatically generated by UIE_DL.py evaluation script.*")

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"\n{'='*80}")
        print(f"Report generated: {output_path}")
        print(f"{'='*80}")

    def run_all_models(self, input_dir, output_dir, gt_dir=None):
        """
        Run all available models sequentially

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            gt_dir: Ground truth directory path (optional)
        """
        print(f"\n{'='*80}")
        print("DEEP LEARNING UIE MODELS EVALUATION")
        print(f"{'='*80}")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Ground Truth: {gt_dir if gt_dir else 'Not provided'}")
        print(f"{'='*80}\n")

        # Check which models exist
        available_models = self.check_models_exist()

        if not available_models:
            print("\nERROR: No model scripts found!")
            return False

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Run each available model
        for model_key in available_models:
            result = self.run_model(model_key, input_dir, output_dir, gt_dir)
            self.results[model_key] = result

            if result:
                print(f"\n✓ {model_key} completed successfully")
            else:
                print(f"\n✗ {model_key} failed")

        return True


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Deep Learning UIE Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python UIE_DL.py --input test_images --output results
  python UIE_DL.py --input test_images --output results --gt ground_truth
  python UIE_DL.py --input test_images --output results --gt ground_truth --report custom_report.md
        '''
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Input image directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--gt', type=str, default=None,
                        help='Ground truth directory (optional, for full-reference metrics)')
    parser.add_argument('--report', type=str, default='UIE_DL.md',
                        help='Output markdown report filename (default: UIE_DL.md)')

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"ERROR: Input directory does not exist: {args.input}")
        sys.exit(1)

    # Validate ground truth directory if provided
    if args.gt and not os.path.isdir(args.gt):
        print(f"ERROR: Ground truth directory does not exist: {args.gt}")
        sys.exit(1)

    # Get base directory (IMPLEMENTATION)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Create evaluator
    evaluator = DLModelEvaluator(base_dir)

    # Run all models
    start_time = time.time()
    success = evaluator.run_all_models(args.input, args.output, args.gt)
    total_time = time.time() - start_time

    if not success:
        print("\nERROR: Evaluation failed")
        sys.exit(1)

    # Generate report
    report_path = os.path.join(args.output, args.report)
    evaluator.generate_markdown_report(report_path, args.input, args.output, args.gt)

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Report: {report_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
