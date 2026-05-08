# FUnIE-GAN - Underwater Image Enhancement

Fast Underwater Image Enhancement for Real-time Applications using Generative Adversarial Networks (GANs).

#### Install All Dependencies:
```bash
pip install -r requirements.txt
```

#### Process Single Image:
```bash
python app_console_pytorch.py input_image.jpg
```

The script expects images in `static/uploads/` folder by default. 
Output will be saved to `static/results/enh_your_image.jpg`

#### Process Video (Frame-by-frame):
```bash
python app_console_pytorch.py video.mp4 --video
```

Process only first 100 frames:
```bash
python app_console_pytorch.py video.mp4 --video --max-frames 100
```
pytho
### 2. Using Keras FUnIE-GAN Application (Paired Model)

```bash
# Single image
python app_funieGAN.py --input input_image.jpg --output enhanced_image.jpg

# Single image with comparison (side-by-side)
python app_funieGAN.py --input input_image.jpg --output comparison.jpg --comparison

# Batch processing (folder)
python funiegan_app.py --input ./dataset/input/ --output ./results/enhanced/ --gt ./dataset/ 


# Use custom model
python app_funieGAN.py --input image.jpg --output output.jpg --model /path/to/model.h5
```

### 3. Using Keras FUnIE-GAN-UP Application (Unpaired Model)

```bash
# Single image
python app_funieGAN_up.py --input input_image.jpg --output enhanced_image.jpg

# Single video
python app_funieGAN.py --input /path/to/video.mp4 --output /path/to/output_frames --video --metrics

#folder images
python app_funieGAN.py --input /path/input --output /path/to/output_frames --metrics
```

