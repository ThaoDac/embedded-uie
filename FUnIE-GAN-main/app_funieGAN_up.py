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
    calculate_video_fps_metrics
)


class FunieGANUpApp:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                'TFKeras/models/gen_up/model_35442_.h5'
            )

        self.model_path = model_path
        self.model = None
        self.load_model()

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

        print(f"Model loaded successfully!")

    def read_and_resize(self, image_path, target_size=(256, 256)):
        img = Image.open(image_path).resize(target_size)

        if img.mode == 'L':
            img = img.convert('RGB')

        return np.array(img).astype(np.float32)

    def enhance_image(self, image_path, output_path=None, show_metrics=False):
        input_img = self.read_and_resize(image_path, (256, 256))
        img_preprocessed = preprocess(input_img)
        img_batch = np.expand_dims(img_preprocessed, axis=0)  

        start_time = time.time()
        enhanced_batch = self.model.predict(img_batch, verbose=0)
        end_time = time.time()

        enhanced_img = deprocess(enhanced_batch)[0]

        if output_path:
            Image.fromarray(enhanced_img).save(output_path)
            print(f"Enhanced image saved to {output_path}")

        if show_metrics:
            image_metrics = evaluate_all_image_metrics(
                input_img=input_img.astype(np.uint8),
                output_img=enhanced_img,
                output_tensor=None,  
                device=None
            )

            perf_metrics = calculate_performance_metrics(
                start_time=start_time,
                end_time=end_time,
                image_shape=enhanced_img.shape,
                model_path=os.path.dirname(self.model_path)
            )

            self.display_results(image_metrics, perf_metrics)

        return enhanced_img

    def enhance_image_with_comparison(self, image_path, output_path):
        input_img = self.read_and_resize(image_path, (256, 256))
        img_preprocessed = preprocess(input_img)
        img_batch = np.expand_dims(img_preprocessed, axis=0)

        enhanced_batch = self.model.predict(img_batch)
        enhanced_img = deprocess(enhanced_batch)[0]

        comparison = np.hstack((input_img.astype('uint8'), enhanced_img)).astype('uint8')
        Image.fromarray(comparison).save(output_path)
        print(f"Comparison image saved to {output_path}")

        return input_img.astype('uint8'), enhanced_img

    def enhance_batch(self, image_paths, output_dir, show_metrics=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Enhancing {len(image_paths)} images...")

        # Collect metrics from all images
        all_frame_times = []
        all_image_metrics = {
            'psnr': [], 'ssim': [], 'uiqm': [], 'uciqe': [], 'niqe': []
        }
        all_perf_metrics = {
            'latency': [], 'fps': [], 'model_size_mb': [], 'memory_mb': [],
            'flops_mflops': [], 'energy_joules': [], 'battery_wh': []
        }

        successful_images = 0

        for i, img_path in enumerate(image_paths):
            img_name = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f"enhanced_{img_name}")

            try:
                input_img = self.read_and_resize(img_path, (256, 256))
                img_preprocessed = preprocess(input_img)
                img_batch = np.expand_dims(img_preprocessed, axis=0)

                start_time = time.time()
                enhanced_batch = self.model.predict(img_batch, verbose=0)
                end_time = time.time()

                enhanced_img = deprocess(enhanced_batch)[0]

                Image.fromarray(enhanced_img).save(output_path)

                frame_time = end_time - start_time
                all_frame_times.append(frame_time)

                if show_metrics:
                    image_metrics = evaluate_all_image_metrics(
                        input_img=input_img.astype(np.uint8),
                        output_img=enhanced_img,
                        output_tensor=None,
                        device=None
                    )

                    perf_metrics = calculate_performance_metrics(
                        start_time=start_time,
                        end_time=end_time,
                        image_shape=enhanced_img.shape,
                        model_path=os.path.dirname(self.model_path)
                    )

                    for key in all_image_metrics.keys():
                        if key in image_metrics:
                            all_image_metrics[key].append(image_metrics[key])

                    for key in all_perf_metrics.keys():
                        if key in perf_metrics:
                            all_perf_metrics[key].append(perf_metrics[key])

                successful_images += 1
                print(f"[{i+1}/{len(image_paths)}] Processed: {img_name} ({frame_time*1000:.2f}ms)")

            except Exception as e:
                print(f"[{i+1}/{len(image_paths)}] Error processing {img_name}: {str(e)}")

        if successful_images > 0 and show_metrics:
            fps_metrics = calculate_video_fps_metrics(all_frame_times)

            avg_image_metrics = {}
            for key, values in all_image_metrics.items():
                if len(values) > 0:
                    avg_image_metrics[key] = np.mean(values)
                else:
                    avg_image_metrics[key] = 0.0

            avg_perf_metrics = {}
            for key, values in all_perf_metrics.items():
                if len(values) > 0:
                    avg_perf_metrics[key] = np.mean(values)
                else:
                    avg_perf_metrics[key] = 0.0

            print("\n" + "=" * 70)
            print("BATCH PROCESSING RESULTS")
            print("=" * 70)
            print(f"  Total Images                        : {len(image_paths)}")
            print(f"  Successfully Processed              : {successful_images}")
            print(f"  Failed                              : {len(image_paths) - successful_images}")
            print()

            print("AVERAGE IMAGE QUALITY METRICS")
            print("-" * 70)
            print(f"  PSNR (Peak Signal-to-Noise Ratio)  : {avg_image_metrics.get('psnr', 0.0):>10.3f} dB")
            print(f"  SSIM (Structural Similarity Index) : {avg_image_metrics.get('ssim', 0.0):>10.3f}")
            print(f"  UIQM (Underwater Image Quality)    : {avg_image_metrics.get('uiqm', 0.0):>10.3f}")
            print(f"  UCIQE (Underwater Color Quality)   : {avg_image_metrics.get('uciqe', 0.0):>10.3f}")
            print(f"  NIQE (Natural Image Quality)       : {avg_image_metrics.get('niqe', 0.0):>10.3f}")
            print()

            print("AVERAGE PERFORMANCE METRICS")
            print("-" * 70)
            print(f"  Total Processing Time               : {fps_metrics['total_time']:.2f} seconds")
            print()
            print("  FPS Performance:")
            print(f"    Average FPS                       : {fps_metrics['avg_fps']:>10.2f} fps")
            print(f"    Minimum FPS (worst case)          : {fps_metrics['min_fps']:>10.2f} fps")
            print(f"    Maximum FPS (best case)           : {fps_metrics['max_fps']:>10.2f} fps")
            print()
            print("  Frame Processing Time:")
            print(f"    Average                           : {fps_metrics['avg_frame_time']*1000:>10.2f} ms")
            print(f"    Minimum                           : {fps_metrics['min_frame_time']*1000:>10.2f} ms")
            print(f"    Maximum                           : {fps_metrics['max_frame_time']*1000:>10.2f} ms")
            print(f"    Std Deviation                     : {fps_metrics['std_frame_time']*1000:>10.2f} ms")
            print(f"    Median                            : {fps_metrics['median_frame_time']*1000:>10.2f} ms")
            print()
            print(f"  Model Size                          : {avg_perf_metrics.get('model_size_mb', 0.0):>10.2f} MB")
            print(f"  Memory Usage (avg)                  : {avg_perf_metrics.get('memory_mb', 0.0):>10.2f} MB")
            print(f"  FLOPs (Est.)                        : {avg_perf_metrics.get('flops_mflops', 0.0):>10.2f} MFLOPs")
            print(f"  Energy Consumption (avg)            : {avg_perf_metrics.get('energy_joules', 0.0):>10.5f} Joules")
            print(f"  Battery Usage (avg)                 : {avg_perf_metrics.get('battery_wh', 0.0):>10.5f} Wh")
            print("=" * 70)

            return {
                'fps_metrics': fps_metrics,
                'image_metrics': avg_image_metrics,
                'perf_metrics': avg_perf_metrics,
                'total_images': len(image_paths),
                'successful': successful_images
            }

        return None

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

        print(f"Input video: {width}x{height} @ {input_fps} FPS, {total_frames} frames")
        print(f"Output directory: {output_dir}")

        frame_times = []
        processed_frames = 0

        print("\nProcessing video frames...")
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

                enhanced_frame = deprocess(enhanced_batch)[0]

                frame_filename = f"frame_{processed_frames:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                Image.fromarray(enhanced_frame).save(frame_path)

                frame_time = end_time - start_time
                frame_times.append(frame_time)
                processed_frames += 1

                if processed_frames % 30 == 0 or processed_frames == total_frames:
                    current_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0
                    print(f"Progress: {processed_frames}/{total_frames} frames ({processed_frames/total_frames*100:.1f}%) - Current FPS: {current_fps:.2f}")

        finally:
            cap.release()

        metrics = calculate_video_fps_metrics(frame_times)

        if show_metrics:
            print("\n" + "=" * 70)
            print("VIDEO PROCESSING FPS METRICS")
            print("=" * 70)
            print(f"  Input Video FPS                     : {input_fps:.2f} fps")
            print(f"  Total Frames Processed              : {metrics['total_frames']}")
            print(f"  Total Processing Time               : {metrics['total_time']:.2f} seconds")
            print()
            print("  FPS Performance:")
            print(f"    Average FPS                       : {metrics['avg_fps']:.2f} fps")
            print(f"    Minimum FPS (worst case)          : {metrics['min_fps']:.2f} fps")
            print(f"    Maximum FPS (best case)           : {metrics['max_fps']:.2f} fps")
            print()
            print("  Frame Processing Time:")
            print(f"    Average                           : {metrics['avg_frame_time']*1000:.2f} ms")
            print(f"    Minimum                           : {metrics['min_frame_time']*1000:.2f} ms")
            print(f"    Maximum                           : {metrics['max_frame_time']*1000:.2f} ms")
            print(f"    Std Deviation                     : {metrics['std_frame_time']*1000:.2f} ms")
            print(f"    Median                            : {metrics['median_frame_time']*1000:.2f} ms")
            print()
            print(f"  Real-time capability: {'YES' if metrics['avg_fps'] >= input_fps else 'NO'}")
            if metrics['avg_fps'] < input_fps:
                print(f"  Speed ratio: {metrics['avg_fps']/input_fps*100:.1f}% of real-time")
            print("=" * 70)

        print(f"\nEnhanced frames saved to {output_dir}/")
        return metrics

    def display_results(self, image_metrics, perf_metrics):
        print("IMAGE QUALITY METRICS")
        print("-" * 70)
        print(f"  PSNR (Peak Signal-to-Noise Ratio)  : {image_metrics.get('psnr', 0.0):>10.3f} dB")
        print(f"  SSIM (Structural Similarity Index) : {image_metrics.get('ssim', 0.0):>10.3f}")
        print(f"  UIQM (Underwater Image Quality)    : {image_metrics.get('uiqm', 0.0):>10.3f}")
        print(f"  UCIQE (Underwater Color Quality)   : {image_metrics.get('uciqe', 0.0):>10.3f}")
        print(f"  NIQE (Natural Image Quality)       : {image_metrics.get('niqe', 0.0):>10.3f}")
        
        print("PERFORMANCE METRICS")
        print("-" * 70)
        print(f"  Inference Latency                   : {perf_metrics.get('latency', 0.0):>10.3f} seconds")
        print(f"  Frames Per Second (FPS)             : {perf_metrics.get('fps', 0.0):>10.2f} fps")
        print(f"  Model Size                          : {perf_metrics.get('model_size_mb', 0.0):>10.2f} MB")
        print(f"  Memory Usage                        : {perf_metrics.get('memory_mb', 0.0):>10.2f} MB")
        print(f"  FLOPs (Est.)                        : {perf_metrics.get('flops_mflops', 0.0):>10.2f} MFLOPs")
        print(f"  Energy Consumption                  : {perf_metrics.get('energy_joules', 0.0):>10.5f} Joules")
        print(f"  Battery Usage                       : {perf_metrics.get('battery_wh', 0.0):>10.5f} Wh")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='FUnIE-GAN-UP Image/Video Enhancement')
    parser.add_argument('--input', type=str, required=True, help='Input image/video path or directory')
    parser.add_argument('--output', type=str, required=True, help='Output image/video path or directory')
    parser.add_argument('--model', type=str, default=None, help='Path to model .h5 file')
    parser.add_argument('--comparison', action='store_true', help='Save side-by-side comparison (images only)')
    parser.add_argument('--metrics', action='store_true', help='Calculate and display quality metrics')
    parser.add_argument('--video', action='store_true', help='Process video file')

    args = parser.parse_args()

    app = FunieGANUpApp(model_path=args.model)

    if args.video:
        if os.path.isfile(args.input):
            app.enhance_video(args.input, args.output, show_metrics=True)
        else:
            print(f"Error: {args.input} is not a valid video file")

    elif os.path.isfile(args.input):
        if args.comparison:
            app.enhance_image_with_comparison(args.input, args.output)
        else:
            app.enhance_image(args.input, args.output, show_metrics=args.metrics)
    elif os.path.isdir(args.input):
        
        from TFKeras.utils.data_utils import getPaths
        image_paths = getPaths(args.input)
        app.enhance_batch(image_paths, args.output)
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()
