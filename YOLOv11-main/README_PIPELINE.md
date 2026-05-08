# finetune
yolo detect train model=yolo11n.pt data=/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml epochs=100 imgsz=512 batch=16 lr0=0.003 lrf=0.05 optimizer=AdamW weight_decay=0.0003 freeze=4 rect=True mosaic=0.8 mixup=0.2 degrees=10 translate=0.1 scale=0.5 pretrained=True

yolo detect train model=yolo11s.pt data=/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml epochs=100 imgsz=640 batch=16 optimizer=AdamW lr0=0.0015 lrf=0.1 weight_decay=0.0005 rect=False freeze=0 mosaic=0.7 mixup=0.08 degrees=10 translate=0.1 scale=0.5 pretrained=True

yolo detect train model=yolo11s.pt data=/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml epochs=150 imgsz=960 batch=8 device=0 optimizer=AdamW lr0=0.0013 lrf=0.012 warmup_epochs=5 warmup_momentum=0.6 warmup_bias_lr=0.05 weight_decay=0.00075 momentum=0.937 mosaic=0.7 mixup=0.05 copy_paste=0.1 close_mosaic=10 degrees=5 translate=0.08 scale=0.5 shear=2 perspective=0.0005 hsv_h=0.015 hsv_s=0.6 hsv_v=0.4 label_smoothing=0.02 pretrained=True
Validating /home/ndpthao/eject/runs/detect/train10/weights/best.pt...
Ultralytics 8.3.49 🚀 Python-3.12.3 torch-2.3.1+cu121 CUDA:0 (NVIDIA GeForce RTX 3060, 11908MiB)
YOLO11s summary (fused): 238 layers, 9,414,348 parameters, 0 gradients, 21.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 75/75 [00:12<00:00,  5.91it/s]
                   all       1200       9390      0.787      0.726      0.787      0.464
           holothurian        255        461      0.748      0.739      0.734      0.432
               echinus        816       4233      0.792      0.881      0.898      0.501
               scallop        315       3196      0.813      0.502      0.682       0.41
              starfish        558       1500      0.797       0.78      0.836      0.514
Speed: 0.4ms preprocess, 4.5ms inference, 0.0ms loss, 1.1ms postprocess per image
Results saved to /home/ndpthao/eject/runs/detect/train10

yolo detect train model=yolo11s.pt data=/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml epochs=70 imgsz=768 batch=12 optimizer=AdamW lr0=0.0015 lrf=0.1 weight_decay=0.0005 rect=False freeze=0 mosaic=0.65 mixup=0.04 degrees=10 translate=0.1 scale=0.4 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 label_smoothing=0.05 pretrained=True

ảnh origin
YOLO11s summary (fused): 238 layers, 9,414,348 parameters, 0 gradients, 21.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 50/50 [00:10<00:00,  4.62it/s]
                   all       1200       9390      0.791      0.679      0.755      0.434
           holothurian        255        461      0.718      0.678       0.68      0.405
               echinus        816       4233      0.812      0.848      0.896      0.486
               scallop        315       3196      0.818      0.449      0.625      0.358
              starfish        558       1500      0.817      0.739       0.82      0.487
Speed: 0.3ms preprocess, 2.9ms inference, 0.0ms loss, 1.2ms postprocess per image

ảnh enhanced
Ultralytics 8.3.49 🚀 Python-3.12.3 torch-2.3.1+cu121 CUDA:0 (NVIDIA GeForce RTX 3060, 11908MiB)
YOLO11s summary (fused): 238 layers, 9,414,348 parameters, 0 gradients, 21.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 50/50 [00:11<00:00,  4.45it/s]
                   all       1200       9390      0.778      0.657      0.722      0.407
           holothurian        255        461      0.749      0.648      0.663      0.383
               echinus        816       4233       0.79      0.827       0.87      0.464
               scallop        315       3196      0.791       0.45      0.603      0.347
              starfish        558       1500      0.783      0.702      0.754      0.437
Speed: 0.3ms preprocess, 2.7ms inference, 0.0ms loss, 1.4ms postprocess per image

origin
yolo detect train model=yolo11n.pt data=/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml epochs=150 imgsz=960 batch=12 device=0 optimizer=AdamW lr0=0.0013 lrf=0.012 warmup_epochs=5 warmup_momentum=0.6 warmup_bias_lr=0.05 weight_decay=0.00075 momentum=0.937 mosaic=0.7 mixup=0.05 copy_paste=0.1 close_mosaic=10 degrees=5 translate=0.08 scale=0.5 shear=2 perspective=0.0005 hsv_h=0.015 hsv_s=0.6 hsv_v=0.4 label_smoothing=0.02 pretrained=True

Validating /home/ndpthao/eject/runs/detect/train8/weights/best.pt...
Ultralytics 8.3.49 🚀 Python-3.12.3 torch-2.3.1+cu121 CUDA:0 (NVIDIA GeForce RTX 3060, 11908MiB)
YOLO11n summary (fused): 238 layers, 2,582,932 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 50/50 [00:14<00:00,  3.52it/s]
                   all       1200       9390      0.795      0.683      0.768      0.445
           holothurian        255        461      0.751      0.695      0.706      0.413
               echinus        816       4233       0.81      0.859      0.901      0.497
               scallop        315       3196      0.831      0.458      0.653      0.386
              starfish        558       1500       0.79      0.721      0.813      0.486
Speed: 0.4ms preprocess, 2.1ms inference, 0.0ms loss, 1.8ms postprocess per image
Results saved to /home/ndpthao/eject/runs/detect/train8

yolo detect train model=yolo11n.pt data=/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml epochs=150 imgsz=1024 batch=10 device=0 optimizer=AdamW lr0=0.0008 lrf=0.008 warmup_epochs=10 warmup_momentum=0.6 warmup_bias_lr=0.05 weight_decay=0.00075 momentum=0.937 mosaic=0.9 mixup=0.05 copy_paste=0.1 close_mosaic=25 degrees=5 translate=0.12 scale=0.6 shear=2 perspective=0.0005 hsv_h=0.015 hsv_s=0.6 hsv_v=0.4 label_smoothing=0.02 iou=0.7 pretrained=True

Validating /home/ndpthao/eject/runs/detect/train7/weights/best.pt...
Ultralytics 8.3.49 🚀 Python-3.12.3 torch-2.3.1+cu121 CUDA:0 (NVIDIA GeForce RTX 3060, 11908MiB)
YOLO11n summary (fused): 238 layers, 2,582,932 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 60/60 [00:16<00:00,  3.75it/s]
                   all       1200       9390      0.792       0.69      0.777       0.45
           holothurian        255        461      0.749      0.675      0.704      0.417
               echinus        816       4233      0.807      0.863      0.901      0.494
               scallop        315       3196      0.816      0.491      0.683      0.399
              starfish        558       1500      0.794      0.731      0.818      0.489
Speed: 0.5ms preprocess, 2.5ms inference, 0.0ms loss, 2.0ms postprocess per image
Results saved to /home/ndpthao/eject/runs/detect/train7

yolo detect train model=yolo11n.pt data=/home/ndpthao/eject/IMPLEMENTATION/Dataset/RUOD/data.yaml epochs=150 imgsz=1024 batch=10 device=0 optimizer=AdamW lr0=0.0008 lrf=0.008 warmup_epochs=10 warmup_momentum=0.6 warmup_bias_lr=0.05 weight_decay=0.00075 momentum=0.937 mosaic=0.9 mixup=0.05 copy_paste=0.1 close_mosaic=25 degrees=5 translate=0.12 scale=0.6 shear=2 perspective=0.0005 hsv_h=0.015 hsv_s=0.6 hsv_v=0.4 label_smoothing=0.02 iou=0.7 pretrained=True

Validating /home/ndpthao/eject/runs/detect/train11/weights/best.pt...
Ultralytics 8.3.49 🚀 Python-3.12.3 torch-2.3.1+cu121 CUDA:0 (NVIDIA GeForce RTX 3060, 11908MiB)
YOLO11n summary (fused): 238 layers, 2,584,102 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 66/66 [00:23<00:00,  2.79it/s]
                   all       1317       7242      0.858      0.802       0.87      0.619
                corals        236        742      0.791      0.652      0.733      0.526
            cuttlefish        235        459      0.938      0.919      0.963      0.795
                 diver        287        617      0.912      0.889      0.934      0.709
               echinus        283       1183      0.865      0.856      0.915      0.553
                  fish        310       1230      0.832      0.717      0.834      0.574
           holothurian        252        809      0.858      0.773      0.856      0.537
             jellyfish         96        239      0.782      0.675      0.768      0.617
               scallop        144        836      0.829      0.744      0.839      0.534
              starfish        272        747      0.817      0.835      0.882      0.564
                turtle        266        380      0.955      0.955      0.979      0.782
Speed: 0.5ms preprocess, 3.2ms inference, 0.0ms loss, 2.6ms postprocess per image
Results saved to /home/ndpthao/eject/runs/detect/train11

dataset origin
Validating /home/ndpthao/eject/runs/detect/train/weights/best.pt...
Ultralytics 8.3.49 🚀 Python-3.12.3 torch-2.3.1+cu121 CUDA:0 (NVIDIA GeForce RTX 3060, 11908MiB)
YOLO11n summary (fused): 238 layers, 2,582,932 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 50/50 [00:13<00:00,  3.59it/s]
                   all       1200       9390      0.808       0.67      0.761      0.439
           holothurian        255        461      0.772      0.657      0.695      0.406
               echinus        816       4233      0.821      0.851      0.903      0.498
               scallop        315       3196      0.832      0.457      0.642      0.378
              starfish        558       1500      0.807      0.715      0.804      0.473
Speed: 0.4ms preprocess, 1.9ms inference, 0.0ms loss, 1.8ms postprocess per image
Results saved to /home/ndpthao/eject/runs/detect/train

yolo detect train model=yolo11n.pt data=/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml epochs=150 imgsz=896 batch=12 device=0 optimizer=AdamW lr0=0.001 lrf=0.01 warmup_epochs=8 warmup_momentum=0.65 warmup_bias_lr=0.05 weight_decay=0.0005 momentum=0.95 mosaic=1.0 mixup=0.15 copy_paste=0.3 close_mosaic=20 degrees=10 translate=0.15 scale=0.7 shear=3 perspective=0.001 hsv_h=0.02 hsv_s=0.7 hsv_v=0.5 flipud=0.2 fliplr=0.5 label_smoothing=0.0 pretrained=True



dataset enhanced (loại bỏ các thamố chỉnh mau hsv_h=0.015 hsv_s=0.6 hsv_v=0.4 )
YOLO11s summary (fused): 238 layers, 9,414,348 parameters, 0 gradients, 21.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 50/50 [00:11<00:00,  4.35it/s]
                   all       1200       9390       0.76       0.66      0.719      0.406
           holothurian        255        461      0.717      0.666       0.66      0.383
               echinus        816       4233      0.779      0.832       0.87      0.463
               scallop        315       3196      0.769      0.445      0.595      0.341
              starfish        558       1500      0.773      0.696      0.752      0.437
Speed: 0.2ms preprocess, 2.7ms inference, 0.0ms loss, 1.6ms postprocess per image
Results saved to /home/ndpthao/eject/runs/detect/train5

dataset enhanced (KHÔNG loại bỏ các thamố chỉnh mau hsv_h=0.015 hsv_s=0.6 hsv_v=0.4 )


Bộ tham số không có hsv
yolo detect train model=yolo11n.pt data=/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml epochs=100 imgsz=896 batch=12 device=0 optimizer=AdamW lr0=0.0013 lrf=0.012 warmup_epochs=5 warmup_momentum=0.6 warmup_bias_lr=0.05 weight_decay=0.00075 momentum=0.937 mosaic=0.7 mixup=0.05 copy_paste=0.1 close_mosaic=10 degrees=5 translate=0.08 scale=0.5 shear=2 perspective=0.0005 label_smoothing=0.02 pretrained=True


dataset origin

dataset enhanced

yolo detect train model=yolo11n.pt data=/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml epochs=100 imgsz=768 batch=12 device=0 optimizer=AdamW lr0=0.0015 lrf=0.008 weight_decay=0.0003 warmup_epochs=15 warmup_momentum=0.85 warmup_bias_lr=0.15 mosaic=1.0 mixup=0.25 copy_paste=0.5 close_mosaic=20 degrees=10 translate=0.15 scale=0.6 shear=3.0 perspective=0.0008 flipud=0.5 fliplr=0.5 hsv_h=0.0 hsv_s=0.0 hsv_v=0.0 label_smoothing=0.08 dropout=0.15 multi_scale=True pretrained=True

-> chạy bộ này bị out of memory, cần giảm imgsz = 768 và epoch 100 (cũ là epochs=250 imgsz=1024)
dataset origin

dataset enhanced



dataset origin

dataset enhanced


dataset origin

dataset enhanced
# inference
yolo detect predict model=/home/ndpthao/eject/runs/detect/train/weights/best.pt source=/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/test/images 

# inference metrics
yolo detect val model=/home/ndpthao/eject/runs/detect/train/weights/best.pt data=/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml split=test save=True

Ultralytics 8.3.49 🚀 Python-3.12.3 torch-2.3.1+cu121 CUDA:0 (NVIDIA GeForce RTX 3060, 11908MiB)
YOLO11n summary (fused): 238 layers, 2,582,932 parameters, 0 gradients, 6.3 GFLOPs
val: Scanning /home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/test/labels.cache... 800 images, 25 backgrounds, 0 corrupt: 100%|██████████| 800/800 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 50/50 [00:12<00:00,  4.14it/s]
                   all        800       6581      0.814      0.712      0.803      0.467
           holothurian        206        373      0.825      0.706      0.773      0.461
               echinus        466       2048      0.794      0.865        0.9       0.49
               scallop        304       3237      0.824      0.525      0.704      0.424
              starfish        343        923      0.811      0.751      0.835      0.494
Speed: 0.9ms preprocess, 4.0ms inference, 0.0ms loss, 2.1ms postprocess per image
Results saved to /home/ndpthao/eject/runs/detect/val

yolo detect val model=/home/ndpthao/eject/runs/detect/train5/weights/best.pt data=/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml split=test save=True

YOLO11s summary (fused): 238 layers, 9,414,348 parameters, 0 gradients, 21.3 GFLOPs
val: Scanning /home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/test_enh_lu2net/labels.cache... 800 images, 25 backgrounds, 0 corrupt: 100%|██████████| 800/800 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 50/50 [00:12<00:00,  3.95it/s]
                   all        800       6581      0.799      0.665      0.755      0.426
           holothurian        206        373      0.802      0.675      0.721      0.425
               echinus        466       2048      0.782      0.812      0.863      0.455
               scallop        304       3237      0.797        0.5      0.654      0.375
              starfish        343        923      0.815      0.673       0.78       0.45
Speed: 0.6ms preprocess, 5.9ms inference, 0.0ms loss, 1.7ms postprocess per image
Results saved to /home/ndpthao/eject/runs/detect/val2

train4/weights/best.pt
Metric          Enhanced     Original     Improvement 
-------------------------------------------------------
mAP50-95        0.1271       0.0934       +36.04%     
mAP50           0.2941       0.2197       +33.89%     
mAP75           0.0885       0.0621       +42.50%     
precision       0.5206       0.5097       +2.14%      
recall          0.2893       0.2037       +42.00%     
f1              0.3719       0.2911       +27.76%  

train5/weights/best/pt
Metric          Enhanced     Original     Improvement 
-------------------------------------------------------
mAP50-95        0.1284       0.1159       +10.76%     
mAP50           0.2950       0.2541       +16.10%     
mAP75           0.0925       0.0925       -0.00%      
precision       0.5739       0.5698       +0.72%      
recall          0.2929       0.2396       +22.27%     
f1              0.3879       0.3373       +14.99%     

Train6
Metric          Enhanced     Original     Improvement 
-------------------------------------------------------
mAP50-95        0.1596       0.1247       +28.02%     
mAP50           0.3581       0.2798       +27.97%     
mAP75           0.1196       0.0897       +33.37%     
precision       0.6017       0.5582       +7.80%      
recall          0.3335       0.2636       +26.50%     
f1              0.4291       0.3581       +19.83%     

# Pipeline: Image Enhancement + YOLO Detection

Pipeline tích hợp tăng cường chất lượng ảnh underwater và phát hiện đối tượng bằng YOLO.

## Quy trình

```
Input Images (URPC2020/test)
    ↓
[1] Image Enhancement (LU2Net PyTorch/ONNX)
    ↓
Enhanced Images (test_enh_modelx/)
    ↓
[2] YOLO Object Detection
    ↓
Metrics: mAP@0.5, mAP@0.5:0.95, Precision, Recall, IoU
```

## Cài đặt

```bash
cd /home/ndpthao/eject/IMPLEMENTATION/YOLOv11-main

# Kiểm tra dependencies
pip install ultralytics onnxruntime pyyaml opencv-python numpy torch
```

## Sử dụng

### 1. Chạy pipeline đầy đủ với LU2Net PyTorch

```bash
python pipeline_enhance_detect.py \
    --enhance-model pytorch \
    --model-name lu2net \
    --test-dir /home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/test \
    --yolo-model /home/ndpthao/eject/IMPLEMENTATION/YOLOv11-main/urpc_train/yolov11n_finetune/weights/best.pt \
    --imgsz 256 \
    --device 0
```

**Kết quả:**
- Enhanced images: `/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/test_enh_lu2net/images/`
- Labels (copied): `/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/test_enh_lu2net/labels/`
- Detection results: `urpc_test_enh_lu2net/yolov11n_lu2net/`
- Metrics file: `urpc_test_enh_lu2net/yolov11n_lu2net/metrics.yaml`

### 2. Chạy pipeline với LU2Net ONNX (nhanh hơn, CPU-optimized)

```bash
python pipeline_enhance_detect.py \
    --enhance-model onnx \
    --model-name lu2net_onnx \
    --onnx-model /home/ndpthao/eject/IMPLEMENTATION/LU2Net-master/LightUNet_170.onnx \
    --test-dir /home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/test \
    --yolo-model /home/ndpthao/eject/IMPLEMENTATION/YOLOv11-main/urpc_train/yolov11n_finetune/weights/best.pt \
    --device 0
```

**Kết quả:**
- Enhanced images: `test_enh_lu2net_onnx/`

### 3. Chạy YOLO trên ảnh gốc (không enhancement)

```bash
python pipeline_enhance_detect.py \
    --enhance-model none \
    --model-name original \
    --test-dir /home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/test \
    --yolo-model /home/ndpthao/eject/IMPLEMENTATION/YOLOv11-main/urpc_train/yolov11n_finetune/weights/best.pt
```

### 4. Chỉ tăng cường ảnh (không chạy YOLO)

```bash
python pipeline_enhance_detect.py \
    --enhance-model pytorch \
    --model-name lu2net \
    --skip-detection
```

### 5. So sánh nhiều enhancement models

```bash
# Model 1: LU2Net PyTorch
python pipeline_enhance_detect.py --enhance-model pytorch --model-name lu2net --device 0

# Model 2: LU2Net ONNX
python pipeline_enhance_detect.py --enhance-model onnx --model-name lu2net_onnx --device 0

# Baseline: No enhancement
python pipeline_enhance_detect.py --enhance-model none --model-name original --device 0
```

## Kết quả Metrics

Pipeline tự động tính toán các metrics sau:

### YOLO Detection Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **mAP@0.5** | Mean Average Precision ở IoU threshold = 0.5 | 0-1 (cao hơn = tốt hơn) |
| **mAP@0.5:0.95** | Mean Average Precision trung bình từ IoU 0.5 đến 0.95 | 0-1 (cao hơn = tốt hơn) |
| **mAP@0.75** | Mean Average Precision ở IoU threshold = 0.75 | 0-1 (cao hơn = tốt hơn) |
| **Precision** | Tỷ lệ detection đúng / tổng detection | 0-1 (cao hơn = ít false positive) |
| **Recall** | Tỷ lệ phát hiện được / tổng ground truth | 0-1 (cao hơn = ít false negative) |
| **F1 Score** | Harmonic mean của Precision và Recall | 0-1 (cao hơn = cân bằng tốt) |
| **IoU** | Intersection over Union (độ chồng lấp bbox) | 0-1 (cao hơn = bbox chính xác hơn) |

### Speed Metrics

- **Preprocess time**: Thời gian tiền xử lý (ms/image)
- **Inference time**: Thời gian inference (ms/image)
- **Postprocess time**: Thời gian hậu xử lý (ms/image)
- **FPS**: Frames per second

## Cấu trúc thư mục output

```
URPC2020/
├── test/                          # Original test set
│   ├── images/
│   └── labels/
├── test_enh_lu2net/              # Enhanced by LU2Net PyTorch
│   ├── images/
│   └── labels/                   # Copied from original
├── test_enh_lu2net_onnx/         # Enhanced by LU2Net ONNX
│   ├── images/
│   └── labels/
└── ...

YOLOv11-main/
├── urpc_test_enh_lu2net/         # Detection results for LU2Net
│   └── yolov11n_lu2net/
│       ├── metrics.yaml          # All metrics in YAML
│       ├── confusion_matrix.png
│       ├── F1_curve.png
│       ├── PR_curve.png
│       ├── P_curve.png
│       └── R_curve.png
├── urpc_test_enh_lu2net_onnx/   # Detection results for ONNX
│   └── ...
└── urpc_test_enh_original/       # Detection results for no enhancement
    └── ...
```

## Ví dụ metrics.yaml

```yaml
mAP50-95: 0.6234      # mAP@0.5:0.95
mAP50: 0.8456         # mAP@0.5 (IoU=0.5)
mAP75: 0.6891         # mAP@0.75 (IoU=0.75)
precision: 0.8234     # Precision
recall: 0.7891        # Recall
f1: 0.8056            # F1 Score
per_class_mAP:        # Per-class mAP
  - 0.85              # Class 0
  - 0.78              # Class 1
  - 0.92              # Class 2
speed:
  preprocess: 0.5     # ms
  inference: 2.3      # ms
  postprocess: 0.3    # ms
```

## Tham số Command Line

### Enhancement Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enhance-model` | `pytorch` | Model type: `pytorch`, `onnx`, `none` |
| `--model-name` | `lu2net` | Tên model cho output directory |
| `--pytorch-model` | `LightUNet_170.pth` | Path to PyTorch model |
| `--onnx-model` | `LightUNet_170.onnx` | Path to ONNX model |

### Dataset Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--test-dir` | `URPC2020/test` | Test dataset directory |
| `--output-base` | `URPC2020/` | Base directory for enhanced images |

### YOLO Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--yolo-model` | `best.pt` | Path to YOLO model |
| `--data-yaml` | `data.yaml` | Path to data configuration |
| `--imgsz` | `256` | Image size for inference |
| `--batch` | `1` | Batch size |
| `--device` | `0` | Device (0, cpu, etc.) |

### Pipeline Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--skip-enhancement` | `False` | Skip enhancement (use existing images) |
| `--skip-detection` | `False` | Skip YOLO detection |

## So sánh Enhancement Models

Sau khi chạy pipeline với nhiều models, so sánh metrics:

```bash
# So sánh mAP@0.5
echo "=== mAP@0.5 Comparison ==="
grep "mAP50:" urpc_test_enh_*/yolov11n_*/metrics.yaml

# So sánh Precision/Recall
echo "=== Precision/Recall Comparison ==="
grep -E "(precision|recall):" urpc_test_enh_*/yolov11n_*/metrics.yaml
```

## Troubleshooting

### Lỗi: CUDA out of memory
```bash
# Giảm batch size
python pipeline_enhance_detect.py --batch 1 --device 0

# Hoặc chạy trên CPU
python pipeline_enhance_detect.py --device cpu
```

### Lỗi: Label files not found
```bash
# Kiểm tra labels directory
ls /home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/test/labels/

# Labels phải cùng tên với images (*.txt)
```

### Lỗi: ONNX model not found
```bash
# Chuyển PyTorch sang ONNX trước
cd /home/ndpthao/eject/IMPLEMENTATION/LU2Net-master
python convert_to_onnx.py --pth LightUNet_170.pth --output LightUNet_170.onnx
```

## Best Practices

1. **Chạy baseline trước**: Luôn chạy với `--enhance-model none` để có baseline metrics
2. **So sánh nhiều models**: Test cả PyTorch và ONNX để chọn model tốt nhất
3. **Lưu metrics**: Copy `metrics.yaml` vào folder riêng để so sánh sau
4. **Visualize results**: Xem confusion matrix và PR curves trong output directory

## Citation

```
LU2Net: Lightweight Underwater Image Enhancement Network
YOLO: You Only Look Once - Object Detection
URPC2020: Underwater Robot Professional Competition Dataset
```
(venv) ndpthao@ndpthao-MS-7A44:~/eject/IMPLEMENTATION$ bash UIE_YOLO.sh
================================================================================
UIE + YOLO OBJECT DETECTION EVALUATION PIPELINE
================================================================================
Start time: Fri Jan 23 08:29:28 AM +07 2026
Input images: /home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/test/images
YOLO weights: /home/ndpthao/eject/IMPLEMENTATION/YOLOv11-main/best.pt
Output: /home/ndpthao/eject/IMPLEMENTATION/UIE_YOLO_Results/run_20260123_082928
================================================================================

================================================================================
Processing: Original (No Enhancement)
================================================================================
Ultralytics 8.3.49 🚀 Python-3.12.3 torch-2.9.1+cpu CPU (Intel Xeon E3-1230 v5 3.40GHz)
YOLO11n summary (fused): 238 layers, 2,582,932 parameters, 0 gradients, 6.3 GFLOPs
val: Scanning /home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/test/labels.cache... 800 images, 25 backgrounds, 0 corrupt: 100%|██████████| 800/800 [00:00<?, ?it/s]
/home/ndpthao/eject/IMPLEMENTATION/venv/lib/python3.12/site-packages/torch/utils/data/dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 50/50 [02:28<00:00,  2.97s/it]
                   all        800       6581      0.812      0.728      0.815      0.473
           holothurian        206        373      0.807      0.739      0.779       0.45
               echinus        466       2048      0.787      0.874      0.906      0.499
               scallop        304       3237      0.825      0.543      0.729      0.435
              starfish        343        923       0.83      0.755      0.844      0.509
Speed: 3.2ms preprocess, 153.2ms inference, 0.0ms loss, 0.8ms postprocess per image
Results saved to /home/ndpthao/eject/IMPLEMENTATION/UIE_YOLO_Results/run_20260123_082928/yolo_Original/results
💡 Learn more at https://docs.ultralytics.com/modes/val
VS Code: view Ultralytics VS Code Extension ⚡ at https://docs.ultralytics.com/integrations/vscode

======================================================================
DCP CONSOLE APPLICATION - DARK CHANNEL PRIOR
======================================================================
Input Directory  : /home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/test/images
Output Directory : /home/ndpthao/eject/IMPLEMENTATION/UIE_YOLO_Results/run_20260123_082928/temp_DCP/images
======================================================================

Dehazing images from: /home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/test/images

======================================================================
DCP PROCESSING PIPELINE
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 
                   all        800       6581      0.764      0.601      0.682      0.381
           holothurian        206        373      0.764      0.627      0.679      0.379
               echinus        466       2048      0.778      0.738      0.802      0.411
               scallop        304       3237      0.753      0.375      0.524      0.311
              starfish        343        923       0.76      0.664      0.723      0.423
======================================================================

Funie-GAN
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 
                   all        800       6581      0.605      0.229      0.327      0.174
           holothurian        206        373      0.411      0.303      0.306      0.147
               echinus        466       2048      0.783       0.29      0.453      0.226
               scallop        304       3237      0.405     0.0568      0.122     0.0704
              starfish        343        923       0.82      0.266      0.427      0.252

======================================================================
HCLR-net PROCESSING PIPELINE
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 
                   all        800       6581      0.678      0.461      0.558      0.307
           holothurian        206        373      0.468      0.649      0.619      0.341
               echinus        466       2048      0.798       0.54      0.673      0.342
               scallop        304       3237      0.655      0.158      0.318      0.188
              starfish        343        923      0.793      0.497      0.623      0.357
======================================================================
Lu2net PROCESSING PIPELINE
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 
                   all        800       6581      0.647      0.434      0.514      0.281
           holothurian        206        373       0.45      0.582      0.558      0.307
               echinus        466       2048      0.772        0.5      0.613      0.302
               scallop        304       3237      0.569      0.142       0.27       0.16
              starfish        343        923      0.798      0.511      0.616      0.356
======================================================================
RGHS PROCESSING PIPELINE
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 
                   all        800       6581      0.744      0.495      0.597      0.334
           holothurian        206        373      0.617      0.579      0.612      0.348
               echinus        466       2048      0.779      0.557      0.661      0.331
               scallop        304       3237      0.771      0.282      0.457      0.271
              starfish        343        923       0.81      0.563      0.656      0.384
======================================================================
UDCP PROCESSING PIPELINE
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 
                   all        800       6581      0.693      0.522        0.6      0.332
           holothurian        206        373      0.621      0.614      0.636      0.359
               echinus        466       2048      0.744      0.607      0.694      0.341
               scallop        304       3237      0.674      0.341      0.468      0.274
              starfish        343        923      0.733      0.526        0.6      0.356
======================================================================
UDCP PROCESSING PIPELINE
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 
                   all        800       6581      0.693      0.522        0.6      0.332
           holothurian        206        373      0.621      0.614      0.636      0.359
               echinus        466       2048      0.744      0.607      0.694      0.341
               scallop        304       3237      0.674      0.341      0.468      0.274
              starfish        343        923      0.733      0.526        0.6      0.356

======================================================================
ULAP PROCESSING PIPELINE
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 
                   all        800       6581      0.718      0.549      0.623      0.343
           holothurian        206        373      0.641      0.579      0.613      0.332
               echinus        466       2048      0.723      0.759      0.778      0.389
               scallop        304       3237      0.762      0.282      0.469      0.282
              starfish        343        923      0.746      0.578      0.633      0.369

======================================================================
CLAHE PROCESSING PIPELINE
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 
                   all        800       6581      0.713      0.498      0.592      0.332
           holothurian        206        373      0.579      0.625      0.641      0.358
               echinus        466       2048       0.78      0.592      0.697      0.362
               scallop        304       3237      0.731      0.185      0.362      0.218
              starfish        343        923      0.764      0.589      0.667       0.39

======================================================================
WaterFormer PROCESSING PIPELINE
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 
                   all        800       6581      0.655      0.469      0.552      0.303
           holothurian        206        373      0.438      0.568      0.536      0.294
               echinus        466       2048      0.767      0.531      0.647      0.324
               scallop        304       3237      0.648        0.2      0.358      0.209
              starfish        343        923      0.769      0.576      0.667      0.385

======================================================================
UIR-polykernel PROCESSING PIPELINE
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):
                   all        800       6581      0.702      0.441       0.55       0.31
           holothurian        206        373       0.51       0.63      0.623      0.354
               echinus        466       2048      0.807      0.486      0.663      0.344
               scallop        304       3237      0.656      0.208      0.351      0.208
              starfish        343        923      0.833       0.44      0.564      0.334

======================================================================
PGHS PROCESSING PIPELINE
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):
                   all        800       6581      0.744       0.54      0.643      0.364
           holothurian        206        373      0.557       0.66      0.648      0.375
               echinus        466       2048       0.81      0.607      0.725      0.375
               scallop        304       3237      0.793      0.302      0.499      0.301
              starfish        343        923      0.815       0.59      0.699      0.403         