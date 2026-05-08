#!/usr/bin/env python3
"""
UIE Non-Deep Learning Models Evaluation Script
===============================================
This script runs multiple non-deep learning underwater image enhancement methods
and evaluates their performance on a given dataset.

Supported Methods:
- CLAHE: Contrast Limited Adaptive Histogram Equalization
- DCP: Dark Channel Prior
- RGHS: Relative Global Histogram Stretching
- UDCP: Underwater Dark Channel Prior
- ULAP: Underwater Light Attenuation Prior

Usage:
    python UIE_nonDL.py --input <input_dir> --output <output_dir> --gt <gt_dir>
"""

import os
import sys
import time
import argparse
import subprocess
import re
from pathlib import Path
from datetime import datetime


class ModelEvaluator:
    """Evaluator for non-deep learning UIE models"""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.models = {
            'CLAHE': {
                'path': os.path.join(base_dir, 'CLAHE-main/app_console.py'),
                'name': 'CLAHE',
                'full_name': 'Contrast Limited Adaptive Histogram Equalization',
                'type': 'Traditional CV Algorithm'
            },
            'DCP': {
                'path': os.path.join(base_dir, 'DCP-main/app_console.py'),
                'name': 'DCP',
                'full_name': 'Dark Channel Prior',
                'type': 'Classical Algorithm'
            },
            'RGHS': {
                'path': os.path.join(base_dir, 'RGHS/app_console.py'),
                'name': 'RGHS',
                'full_name': 'Relative Global Histogram Stretching',
                'type': 'Traditional Algorithm'
            },
            'UDCP': {
                'path': os.path.join(base_dir, 'UDCP-main/app_console.py'),
                'name': 'UDCP',
                'full_name': 'Underwater Dark Channel Prior',
                'type': 'Classical Algorithm'
            },
            'ULAP': {
                'path': os.path.join(base_dir, 'ULAP-main/app_console.py'),
                'name': 'ULAP',
                'full_name': 'Underwater Light Attenuation Prior',
                'type': 'Traditional Algorithm'
            }
        }
        self.results = {}

    def check_models_exist(self):
        """Check if all model scripts exist"""
        missing = []
        for model_key, model_info in self.models.items():
            if not os.path.exists(model_info['path']):
                missing.append(f"{model_key}: {model_info['path']}")

        if missing:
            print("ERROR: The following model scripts are missing:")
            for m in missing:
                print(f"  - {m}")
            return False
        return True

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
        print(f"{'='*80}")
        print(f"Command: {' '.join(cmd)}")

        # Run the model
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            elapsed_time = time.time() - start_time

            if result.returncode != 0:
                print(f"ERROR: {model_key} failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
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

            return metrics

        except subprocess.TimeoutExpired:
            print(f"ERROR: {model_key} execution timed out (>1 hour)")
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
            'flops_mflops': 0.0,
            'flops_gflops': 0.0,
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

            # Resource metrics
            'model_size_mb': r'Model Size\s*:\s*([\d.]+)\s*MB',
            'flops_mflops': r'FLOPs\s*:\s*([\d.]+)\s*MFLOPs',
            'flops_gflops': r'FLOPs\s*:\s*([\d.]+)\s*GFLOPs',
            'memory_mb': r'Memory Usage\s*:\s*([\d.]+)\s*MB',
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

        # Convert FLOPs to consistent unit (MFLOPs)
        if metrics['flops_gflops'] > 0:
            metrics['flops_mflops'] = metrics['flops_gflops'] * 1000

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
        report.append("# Non-Deep Learning UIE Models Evaluation Report")
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
        report.append("| Model | Full Name | Type | Images | Status |")
        report.append("|-------|-----------|------|--------|--------|")

        for model_key in self.models.keys():
            model_info = self.models[model_key]
            if model_key in self.results and self.results[model_key]:
                metrics = self.results[model_key]
                status = f"✓ ({metrics['processed_images']}/{metrics['total_images']})"
            else:
                status = "✗ Failed"

            report.append(f"| {model_info['name']} | {model_info['full_name']} | {model_info['type']} | {metrics.get('total_images', 0) if model_key in self.results and self.results[model_key] else 0} | {status} |")

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
            report.append(f"**Type:** {model_info['type']}")
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
            report.append(f"| Model Size | {metrics['model_size_mb']:.2f} MB |")
            report.append(f"| FLOPs | {metrics['flops_mflops']:.2f} MFLOPs |")
            report.append(f"| Memory Usage | {metrics['memory_mb']:.2f} MB |")
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
            report.append("| Model | PSNR (dB) | SSIM | UCIQE | UIQM | NIQE |")
            report.append("|-------|-----------|------|-------|------|------|")

            for model_key in self.models.keys():
                if model_key in self.results and self.results[model_key]:
                    m = self.results[model_key]
                    psnr = f"{m['psnr']:.4f}" if m['psnr'] > 0 else "N/A"
                    ssim = f"{m['ssim']:.4f}" if m['ssim'] > 0 else "N/A"
                    uciqe = f"{m['uciqe']:.4f}" if m['uciqe'] > 0 else "N/A"
                    uiqm = f"{m['uiqm']:.4f}" if m['uiqm'] > 0 else "N/A"
                    niqe = f"{m['niqe']:.4f}" if m['niqe'] > 0 else "N/A"
                    report.append(f"| {m['model_name']} | {psnr} | {ssim} | {uciqe} | {uiqm} | {niqe} |")
            report.append("")

        # Performance comparison
        report.append("### Performance Metrics Comparison")
        report.append("")
        report.append("| Model | Avg FPS | Inference (s) | Total Time (s) | RAM (MB) | Energy (J) |")
        report.append("|-------|---------|---------------|----------------|----------|------------|")

        for model_key in self.models.keys():
            if model_key in self.results and self.results[model_key]:
                m = self.results[model_key]
                report.append(f"| {m['model_name']} | {m['avg_fps']:.2f} | {m['inference_time']:.4f} | {m['total_time']:.4f} | {m['ram_usage_mb']:.2f} | {m['energy_joules']:.2f} |")

        report.append("")

        # Resource comparison
        report.append("### Resource Metrics Comparison")
        report.append("")
        report.append("| Model | Model Size (MB) | FLOPs (MFLOPs) | Memory (MB) | Energy (Wh) |")
        report.append("|-------|-----------------|----------------|-------------|-------------|")

        for model_key in self.models.keys():
            if model_key in self.results and self.results[model_key]:
                m = self.results[model_key]
                report.append(f"| {m['model_name']} | {m['model_size_mb']:.2f} | {m['flops_mflops']:.2f} | {m['memory_mb']:.2f} | {m['battery_wh']:.6f} |")

        report.append("")

        # Conclusions
        report.append("## Conclusions")
        report.append("")
        report.append("### Best Performers")
        report.append("")

        # Find best performers
        valid_results = {k: v for k, v in self.results.items() if v}

        if valid_results:
            # Best quality (UCIQE)
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

            # Lowest memory
            if any(m['ram_usage_mb'] > 0 for m in valid_results.values()):
                best_ram = min(valid_results.items(), key=lambda x: x[1]['ram_usage_mb'] if x[1]['ram_usage_mb'] > 0 else 999999)
                report.append(f"- **Lowest RAM Usage:** {best_ram[1]['model_name']} ({best_ram[1]['ram_usage_mb']:.2f} MB)")

        report.append("")
        report.append("---")
        report.append("")
        report.append("*This report was automatically generated by UIE_nonDL.py evaluation script.*")

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"\n{'='*80}")
        print(f"Report generated: {output_path}")
        print(f"{'='*80}")

    def run_all_models(self, input_dir, output_dir, gt_dir=None):
        """
        Run all models sequentially

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            gt_dir: Ground truth directory path (optional)
        """
        print(f"\n{'='*80}")
        print("NON-DEEP LEARNING UIE MODELS EVALUATION")
        print(f"{'='*80}")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Ground Truth: {gt_dir if gt_dir else 'Not provided'}")
        print(f"{'='*80}\n")

        # Check all models exist
        if not self.check_models_exist():
            return False

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Run each model
        for model_key in self.models.keys():
            result = self.run_model(model_key, input_dir, output_dir, gt_dir)
            self.results[model_key] = result

            if result:
                print(f"\n✓ {model_key} completed successfully")
            else:
                print(f"\n✗ {model_key} failed")

        return True


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Non-Deep Learning UIE Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python UIE_nonDL.py --input test_images --output results
  python UIE_nonDL.py --input test_images --output results --gt ground_truth
  python UIE_nonDL.py --input test_images --output results --gt ground_truth --report custom_report.md
        '''
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Input image directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--gt', type=str, default=None,
                        help='Ground truth directory (optional, for full-reference metrics)')
    parser.add_argument('--report', type=str, default='UIE_nonDL.md',
                        help='Output markdown report filename (default: UIE_nonDL.md)')

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
    evaluator = ModelEvaluator(base_dir)

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
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Report: {report_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
