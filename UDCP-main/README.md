# UDCP - Underwater Dark Channel Prior

## Introduction

This project implements the **UDCP (Underwater Dark Channel Prior)** algorithm for underwater image enhancement. UDCP is a physics-based method specifically designed for underwater images by removing the effects of water (light absorption and scattering).

## UDCP vs DCP

**Key Difference:**
- **UDCP**: Dark Channel = min(G, B) - optimized for underwater images (excludes Red channel)
- **DCP**: Dark Channel = min(R, G, B) - designed for atmospheric haze/fog

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Flask
- Pillow
- scikit-image
- scipy
- psutil

## Installation

1. Navigate to the project directory:
```bash
cd IMPLEMENTATION/UDCP-main
```

2. Install dependencies:
```bash
pip install flask numpy opencv-python scikit-image scipy psutil pillow
```

3. (Optional) Install PyTorch for advanced NIQE metric:
```bash
# CPU version
pip install torch torchvision

# GPU version (if CUDA available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

4. (Optional) Install PyIQA for more accurate NIQE:
```bash
pip install pyiqa
```

## Usage

### Web Interface

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload an underwater image and click "Enhance Image" to process it.

4. View results including:
   - Input and output images
   - Pixel matrices comparison
   - Dark channel visualization
   - Transmission maps (raw and refined)
   - Heatmap visualization
   - Quality and performance metrics

