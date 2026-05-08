# DCP - Dark Channel Prior for Image Dehazing

## Introduction

This project implements the **Dark Channel Prior (DCP)** algorithm for single image dehazing/defogging. The algorithm removes haze, fog, or turbidity from images to restore clear visibility and natural colors.

**Dark Channel Prior** is a classical computer vision technique that works by:
1. Computing the dark channel (minimum intensity across RGB channels)
2. Estimating atmospheric light
3. Calculating transmission map
4. Refining transmission with guided filter
5. Recovering the clear image using the image formation model

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Flask
- Pillow
- scikit-image
- psutil

## Installation

1. Clone the repository:
```bash
cd IMPLEMENTATION/DCP-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
### Web Interface

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload an image and click "Dehaze" to process it.

