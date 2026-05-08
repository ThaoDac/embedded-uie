import os
import sys
import time
import numpy as np
from PIL import Image

try:
    from tf_keras.models import model_from_json
except ImportError:
    try:
        from keras.models import model_from_json
    except ImportError:
        from tensorflow.keras.models import model_from_json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TFKeras'))
from TFKeras.utils.data_utils import preprocess, deprocess

from metrics import (
    calculate_performance_metrics,
    evaluate_all_image_metrics,
    calculate_video_fps_metrics,
    measure_inference_memory
)


class FunieGANApp:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                'TFKeras/models/gen_p/model_15320_.h5'
            )

        self.model_path = model_path
        self.model = None
        self.load_model()

    # =============================
    # MODEL LOADING
    # =============================
    def load_model(self):
        model_dir = os.path.dirname(self.model_path)
        model_name = os.path.basename(self.model_path).replace('.h5', '')

        model_h5 = os.path.join(model_dir, f"{model_name}.h5")
        model_json = os.path.join(model_dir, f"{model_name}.json")

        if not os.path.exists(model_h5):
            raise FileNotFoundError(f"Model file not found: {model_h5}")
        if not os.path.exists(model_json):
            raise FileNotFoundError(f"Model JSON file not found: {model_json}")

        print(f"Loading model architecture from {model_json}...")
        print(f"Loading model weights from {model_h5}...")

        with open(model_json, "r") as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(model_h5)

        print(f"Model loaded successfully!\n")

    # =============================
    # IMAGE UTILS
    # =============================
    def read_and_resize(self, image_path, target_size=(256, 256)):
        img = Image.open(image_path).resize(target_size)
        if img.mode == 'L':
            img = img.convert('RGB')
        return np.array(img).astype(np.float32)

    # =============================
    # ENHANCEMENT FOR A BATCH OF IMAGES
    # =============================
    def enhance_batch(self, image_paths, output_dir, gt_dir=None, show_metrics=True,
                     measure_actual_memory=False):
        """
        Enhance a batch of images and calculate average metrics.

        Args:
            image_paths: List of image file paths
            output_dir: Directory to save enhanced images
            gt_dir: Path to directory containing ground truth images (optional)
            show_metrics: Whether to display average metrics
            measure_actual_memory: Whether to measure actual inference memory (slower but accurate)
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Enhancing {len(image_paths)} images...")
        if gt_dir:
            print(f"Using ground truth images from: {gt_dir}")
        if measure_actual_memory:
            print("⚠️  Measuring actual inference memory (this will be slower)")

        all_frame_times = []
        all_image_metrics = {'psnr': [], 'ssim': [], 'uiqm': [], 'uciqe': [], 'niqe': []}
        all_perf_metrics = {
            'latency': [], 'fps': [], 'model_size_mb': [], 'memory_mb': [],
            'flops_mflops': [], 'energy_joules': [], 'battery_wh': []
        }

        # Measure inference memory once for the first image if requested
        inference_memory_info = None
        if measure_actual_memory and len(image_paths) > 0:
            print("\n" + "=" * 70)
            print("MEASURING ACTUAL INFERENCE MEMORY (First image)")
            print("=" * 70)
            # Use first image for memory measurement
            first_img = self.read_and_resize(image_paths[0], (256, 256))
            first_img_preprocessed = preprocess(first_img)
            first_img_batch = np.expand_dims(first_img_preprocessed, axis=0)
            inference_memory_info = measure_inference_memory(
                model=self.model,
                input_data=first_img_batch,
                warmup_runs=2,
                measure_runs=1
            )
            print("=" * 70 + "\n")

        successful_images = 0

        for i, img_path in enumerate(image_paths):
            img_name = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f"enhanced_{img_name}")

            try:
                input_img = self.read_and_resize(img_path, (256, 256))
                img_preprocessed = preprocess(input_img)
                img_batch = np.expand_dims(img_preprocessed, axis=0)

                # ---- Model inference time (for FPS calculation)
                start_time = time.time()
                enhanced_batch = self.model.predict(img_batch, verbose=0)
                end_time = time.time()
                inference_time = end_time - start_time
                all_frame_times.append(inference_time)

                # enhanced_img = deprocess(enhanced_batch)[0]
                # Image.fromarray(enhanced_img).save(output_path)

                # Lấy kích thước gốc
                orig_img = Image.open(img_path)
                orig_w, orig_h = orig_img.size

                enhanced_img = deprocess(enhanced_batch)[0]

                # Resize lại đúng kích thước ban đầu
                enhanced_resized = Image.fromarray(enhanced_img).resize((orig_w, orig_h), Image.BICUBIC)

                enhanced_resized.save(output_path)


                # ---- Metrics (after inference only)
                if show_metrics:
                    if gt_dir:
                        gt_path = os.path.join(gt_dir, img_name)
                        if os.path.exists(gt_path):
                            gt_img = self.read_and_resize(gt_path, (256, 256))
                            ref_img = gt_img
                        else:
                            print(f"⚠️ GT not found for {img_name}, skipping PSNR/SSIM.")
                            ref_img = None
                    else:
                        ref_img = None

                    # Compute image metrics
                    image_metrics = evaluate_all_image_metrics(
                        input_img=ref_img.astype(np.uint8) if ref_img is not None else None,
                        output_img=enhanced_img,
                        output_tensor=None,
                        device=None
                    )

                    # Compute performance metrics (only inference time)
                    perf_metrics = calculate_performance_metrics(
                        start_time=start_time,
                        end_time=end_time,
                        image_shape=enhanced_img.shape,
                        model_path=self.model_path,
                        model=self.model,
                        inference_memory_info=inference_memory_info
                    )

                    # Collect metrics
                    for key in all_image_metrics.keys():
                        if key in image_metrics:
                            all_image_metrics[key].append(image_metrics[key])

                    for key in all_perf_metrics.keys():
                        if key in perf_metrics:
                            all_perf_metrics[key].append(perf_metrics[key])

                successful_images += 1
                print(f"[{i + 1}/{len(image_paths)}] {img_name} done ({inference_time * 1000:.2f} ms)")

            except Exception as e:
                print(f"[{i + 1}/{len(image_paths)}] Error processing {img_name}: {str(e)}")

        # ---- Summary
        if successful_images > 0 and show_metrics:
            fps_metrics = calculate_video_fps_metrics(all_frame_times)

            avg_image_metrics = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in all_image_metrics.items()}
            avg_perf_metrics = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in all_perf_metrics.items()}

            print("\n" + "=" * 70)
            print("BATCH PROCESSING RESULTS")
            print("=" * 70)
            print(f"  Total Images: {len(image_paths)} | Successful: {successful_images}")
            print("-" * 70)
            print("AVERAGE IMAGE QUALITY METRICS")
            print(f"  PSNR  : {avg_image_metrics['psnr']:.3f} dB")
            print(f"  SSIM  : {avg_image_metrics['ssim']:.3f}")
            print(f"  UIQM  : {avg_image_metrics['uiqm']:.3f}")
            print(f"  UCIQE : {avg_image_metrics['uciqe']:.3f}")
            print(f"  NIQE  : {avg_image_metrics['niqe']:.3f}")
            print("-" * 70)
            print("PERFORMANCE METRICS (Inference only)")
            print(f"  Avg Latency per image : {avg_perf_metrics['latency']:.4f} s")
            print(f"  Avg FPS (1/latency)   : {fps_metrics['avg_fps']:.2f}")
            print(f"  Model Size            : {avg_perf_metrics['model_size_mb']:.2f} MB")

            # Display detailed memory info if actual inference memory was measured
            if inference_memory_info is not None:
                print(f"  Memory Usage (Inference): {avg_perf_metrics['memory_mb']:.2f} MB")
                if 'baseline_memory_mb' in inference_memory_info:
                    print(f"    ├─ Baseline Memory    : {inference_memory_info['baseline_memory_mb']:.2f} MB")
                    print(f"    ├─ Peak Memory        : {inference_memory_info['peak_memory_mb']:.2f} MB")
                    print(f"    └─ Inference Memory   : {inference_memory_info['inference_memory_mb']:.2f} MB")
                if inference_memory_info.get('gpu_inference_mb', 0) > 0:
                    print(f"  GPU Memory (Inference)  : {inference_memory_info['gpu_inference_mb']:.2f} MB")
                    print(f"    ├─ GPU Baseline       : {inference_memory_info['gpu_baseline_mb']:.2f} MB")
                    print(f"    └─ GPU Peak           : {inference_memory_info['gpu_peak_mb']:.2f} MB")
            else:
                print(f"  Memory Usage (Model)  : {avg_perf_metrics['memory_mb']:.2f} MB")

            print(f"  Energy Consumption    : {avg_perf_metrics['energy_joules']:.2f} J ({avg_perf_metrics['battery_wh']:.6f} Wh)")
            print("=" * 70)
            print("DETAILED FPS METRICS (Inference only)")
            print("=" * 70)
            print(f"  Total Frames  : {fps_metrics['total_frames']}")
            print(f"  Total Time    : {fps_metrics['total_time']:.2f} s")
            print(f"  Avg FPS       : {fps_metrics['avg_fps']:.2f}")
            print(f"  Min FPS       : {fps_metrics['min_fps']:.2f}")
            print(f"  Max FPS       : {fps_metrics['max_fps']:.2f}")
            print("=" * 70)
        return None

    # =============================
    # ENHANCE VIDEO (no GT)
    # =============================
    def enhance_video(self, video_path, output_dir, show_metrics=True):
        import cv2

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        input_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {input_fps} FPS | {total_frames} frames")
        print(f"Output: {output_dir}")

        frame_times = []
        processed_frames = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb).resize((256, 256))
                frame_np = np.array(frame_pil).astype(np.float32)
                frame_preprocessed = preprocess(frame_np)
                frame_batch = np.expand_dims(frame_preprocessed, axis=0)

                start_time = time.time()
                enhanced_batch = self.model.predict(frame_batch, verbose=0)
                end_time = time.time()
                inference_time = end_time - start_time
                frame_times.append(inference_time)

                enhanced_frame = deprocess(enhanced_batch)[0]
                frame_filename = f"frame_{processed_frames:06d}.jpg"
                Image.fromarray(enhanced_frame).save(os.path.join(output_dir, frame_filename))
                processed_frames += 1

                if processed_frames % 30 == 0:
                    current_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0
                    print(f"Frame {processed_frames}/{total_frames} ({processed_frames / total_frames * 100:.1f}%) - {current_fps:.2f} FPS")

        finally:
            cap.release()

        fps_metrics = calculate_video_fps_metrics(frame_times)
        if show_metrics:
            print("\n" + "=" * 70)
            print("VIDEO INFERENCE PERFORMANCE")
            print("=" * 70)
            print(f"  Total Frames  : {fps_metrics['total_frames']}")
            print(f"  Total Time    : {fps_metrics['total_time']:.2f} s")
            print(f"  Avg FPS       : {fps_metrics['avg_fps']:.2f}")
            print(f"  Min FPS       : {fps_metrics['min_fps']:.2f}")
            print(f"  Max FPS       : {fps_metrics['max_fps']:.2f}")
            print("=" * 70)

        print(f"\nEnhanced frames saved to {output_dir}/")
        return fps_metrics


# =============================
# MAIN
# =============================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='FUnIE-GAN Enhancement (Images or Video)')
    parser.add_argument('--input', type=str, required=True, help='Path to input image folder or video file')
    parser.add_argument('--output', type=str, required=True, help='Path to save enhanced outputs')
    parser.add_argument('--gt', type=str, default=None, help='Path to Ground Truth folder (optional)')
    parser.add_argument('--model', type=str, default=None, help='Path to model .h5 file')
    parser.add_argument('--video', action='store_true', help='Flag to process as video')
    parser.add_argument('--measure-memory', action='store_true',
                       help='Measure actual inference memory (includes activations, slower but accurate)')

    args = parser.parse_args()

    app = FunieGANApp(model_path=args.model)

    if args.video:
        app.enhance_video(args.input, args.output, show_metrics=True)
    else:
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} must be a directory for image batch processing.")
            return
        from TFKeras.utils.data_utils import getPaths
        image_paths = getPaths(args.input)
        app.enhance_batch(image_paths, args.output, gt_dir=args.gt,
                         show_metrics=True, measure_actual_memory=args.measure_memory)


if __name__ == "__main__":
    main()
