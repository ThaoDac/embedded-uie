# Tối ưu Training Parameters cho Enhanced Images (LU2Net + YOLO)

## Quan trọng: Khi đã dùng Image Enhancement

Vì bạn đã dùng **LU2Net enhancement** trước khi train YOLO, các **color/brightness augmentations NÊN GIẢM MẠNH** hoặc **TẮT HẲN**!

### Lý do:

1. ✅ **Enhanced images đã có màu sắc/lighting chuẩn hóa**
   - LU2Net đã fix color distortion, lighting, contrast
   - HSV augmentation sẽ **phá vỡ** sự chuẩn hóa này

2. ✅ **Geometric augmentation vẫn cần**
   - Rotation, scale, flip vẫn quan trọng
   - Không ảnh hưởng đến color enhancement

3. ✅ **Mosaic/Mixup vẫn cần cho class balancing**
   - Giúp xử lý class imbalance
   - Không conflict với enhancement

## Bộ tham số TỐI ƯU cho Enhanced Dataset

### Version 1: Balanced (KHUYẾN NGHỊ cho enhanced images)

```bash
yolo detect train \
  model=yolo11n.pt \
  data=/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml \
  epochs=200 \
  imgsz=896 \
  batch=16 \
  device=0 \
  \
  # === Optimizer ===
  optimizer=AdamW \
  lr0=0.002 \
  lrf=0.01 \
  weight_decay=0.0005 \
  momentum=0.937 \
  \
  # === Warmup ===
  warmup_epochs=10 \
  warmup_momentum=0.8 \
  warmup_bias_lr=0.1 \
  \
  # === Mosaic & Mixup (QUAN TRỌNG cho class balance!) ===
  mosaic=1.0 \
  mixup=0.15 \
  copy_paste=0.3 \
  close_mosaic=15 \
  \
  # === Geometric Augmentation (GIỮ NGUYÊN - không ảnh hưởng enhancement) ===
  degrees=15 \
  translate=0.2 \
  scale=0.7 \
  shear=5.0 \
  perspective=0.001 \
  flipud=0.5 \
  fliplr=0.5 \
  \
  # === Color Augmentation (TẮT hoặc RẤT NHẸ cho enhanced images!) ===
  hsv_h=0.0 \
  hsv_s=0.0 \
  hsv_v=0.0 \
  \
  # === Regularization ===
  label_smoothing=0.05 \
  dropout=0.1 \
  \
  # === Other ===
  multi_scale=True \
  rect=False \
  pretrained=True \
  patience=50 \
  save_period=10 \
  project=urpc_enhanced_v1 \
  name=yolo11n_lu2net_balanced
```

### Version 2: Small Objects Focus (cho enhanced images)

Nếu scallop/holothurian nhỏ:

```bash
yolo detect train \
  model=yolo11n.pt \
  data=/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml \
  epochs=250 \
  imgsz=1024 \
  batch=12 \
  device=0 \
  \
  # === Optimizer ===
  optimizer=AdamW \
  lr0=0.0015 \
  lrf=0.008 \
  weight_decay=0.0003 \
  \
  # === Warmup ===
  warmup_epochs=15 \
  warmup_momentum=0.85 \
  warmup_bias_lr=0.15 \
  \
  # === Strong Augmentation cho small objects ===
  mosaic=1.0 \
  mixup=0.25 \
  copy_paste=0.5 \
  close_mosaic=20 \
  \
  # === Geometric (moderate - preserve small objects) ===
  degrees=10 \
  translate=0.15 \
  scale=0.6 \
  shear=3.0 \
  perspective=0.0008 \
  flipud=0.5 \
  fliplr=0.5 \
  \
  # === Color Aug: TẮT cho enhanced images! ===
  hsv_h=0.0 \
  hsv_s=0.0 \
  hsv_v=0.0 \
  \
  # === Regularization ===
  label_smoothing=0.08 \
  dropout=0.15 \
  \
  multi_scale=True \
  pretrained=True \
  patience=80 \
  project=urpc_enhanced_v2 \
  name=yolo11n_lu2net_small_obj
```

### Version 3: Conservative (ít aug hơn, stable training)

```bash
yolo detect train \
  model=yolo11n.pt \
  data=/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml \
  epochs=180 \
  imgsz=896 \
  batch=16 \
  device=0 \
  \
  # === Optimizer ===
  optimizer=AdamW \
  lr0=0.001 \
  lrf=0.015 \
  weight_decay=0.0008 \
  \
  # === Warmup ===
  warmup_epochs=8 \
  warmup_momentum=0.75 \
  warmup_bias_lr=0.08 \
  \
  # === Moderate Augmentation ===
  mosaic=0.85 \
  mixup=0.12 \
  copy_paste=0.25 \
  close_mosaic=12 \
  \
  # === Geometric ===
  degrees=12 \
  translate=0.15 \
  scale=0.6 \
  shear=4.0 \
  perspective=0.0007 \
  flipud=0.5 \
  fliplr=0.5 \
  \
  # === Color Aug: TẮT! ===
  hsv_h=0.0 \
  hsv_s=0.0 \
  hsv_v=0.0 \
  \
  # === Regularization ===
  label_smoothing=0.04 \
  dropout=0.08 \
  \
  pretrained=True \
  multi_scale=True \
  patience=60 \
  project=urpc_enhanced_v3 \
  name=yolo11n_lu2net_conservative
```

## So sánh: Original vs Enhanced Dataset Training

| Parameter | Original Images | Enhanced Images (LU2Net) | Lý do |
|-----------|-----------------|--------------------------|-------|
| **hsv_h** | 0.015-0.03 | **0.0** (TẮT) | Enhanced đã chuẩn hóa color |
| **hsv_s** | 0.6-0.9 | **0.0** (TẮT) | Enhanced đã fix saturation |
| **hsv_v** | 0.4-0.7 | **0.0** (TẮT) | Enhanced đã normalize brightness |
| **mosaic** | 0.7-1.0 | **1.0** (TĂNG) | Vẫn cần cho class balance |
| **mixup** | 0.05-0.25 | **0.15-0.25** | Vẫn cần |
| **copy_paste** | 0.1-0.5 | **0.3-0.5** | Vẫn cần cho minority classes |
| **degrees** | 15 | 15 | Không đổi |
| **translate** | 0.2 | 0.2 | Không đổi |
| **scale** | 0.7 | 0.7 | Không đổi |
| **flipud** | 0.5 | 0.5 | Không đổi |
| **fliplr** | 0.5 | 0.5 | Không đổi |

## Giải thích chi tiết

### 1. TẠI SAO tắt HSV augmentation?

```python
# LU2Net đã làm gì:
- Chuẩn hóa color distribution
- Fix underwater color cast (blue/green tint)
- Normalize lighting/contrast
- Enhance clarity

# Nếu dùng HSV aug trên enhanced images:
hsv_s=0.8 → Thay đổi saturation → PHÁ VỠ color normalization của LU2Net
hsv_v=0.6 → Thay đổi brightness → PHÁ VỠ lighting normalization
hsv_h=0.025 → Shift hue → PHÁ VỠ color correction

→ Model sẽ học cả ảnh enhanced TỐT và ảnh bị HSV aug PHÁ HOẠI
→ Kết quả inference kém hơn!
```

### 2. GIỮ NGUYÊN geometric augmentation

```python
# Geometric augs KHÔNG ảnh hưởng color/lighting enhancement:
- degrees (rotation): OK ✅
- translate (shift): OK ✅
- scale (zoom): OK ✅
- shear: OK ✅
- flip: OK ✅
- perspective: OK ✅

→ Vẫn giúp model robust với pose/position/size variations
→ KHÔNG conflict với LU2Net enhancement
```

### 3. TĂNG mosaic/mixup cho class balance

```python
# Dataset imbalance:
holothurian: 461 samples (5%)  ← Rất ít!
scallop: 3196 samples (34%)
starfish: 1500 samples (16%)
echinus: 4233 samples (45%)

# Giải pháp:
mosaic=1.0     → Kết hợp 4 ảnh → balance classes trong 1 batch
mixup=0.15-0.25 → Blend 2 ảnh → augment minority classes
copy_paste=0.3-0.5 → Paste holothurians vào ảnh khác → tăng samples
```

## Command Line KHUYẾN NGHỊ (Ready to run)

### V1 - Balanced (CHẠY NÀY TRƯỚC!)

```bash
cd /home/ndpthao/eject/IMPLEMENTATION/YOLOv11-main

yolo detect train \
  model=yolo11n.pt \
  data=/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml \
  epochs=200 \
  imgsz=896 \
  batch=16 \
  device=0 \
  optimizer=AdamW \
  lr0=0.002 \
  lrf=0.01 \
  weight_decay=0.0005 \
  momentum=0.937 \
  warmup_epochs=10 \
  warmup_momentum=0.8 \
  warmup_bias_lr=0.1 \
  mosaic=1.0 \
  mixup=0.15 \
  copy_paste=0.3 \
  close_mosaic=15 \
  degrees=15 \
  translate=0.2 \
  scale=0.7 \
  shear=5.0 \
  perspective=0.001 \
  flipud=0.5 \
  fliplr=0.5 \
  hsv_h=0.0 \
  hsv_s=0.0 \
  hsv_v=0.0 \
  label_smoothing=0.05 \
  dropout=0.1 \
  multi_scale=True \
  pretrained=True \
  patience=50 \
  save_period=10 \
  project=urpc_enhanced_v1 \
  name=yolo11n_lu2net_balanced
```

### V2 - Small Objects (nếu scallop vẫn thấp sau V1)

```bash
yolo detect train \
  model=yolo11n.pt \
  data=/home/ndpthao/eject/IMPLEMENTATION/Dataset/URPC2020/data.yaml \
  epochs=250 \
  imgsz=1024 \
  batch=12 \
  device=0 \
  optimizer=AdamW \
  lr0=0.0015 \
  lrf=0.008 \
  weight_decay=0.0003 \
  warmup_epochs=15 \
  warmup_momentum=0.85 \
  warmup_bias_lr=0.15 \
  mosaic=1.0 \
  mixup=0.25 \
  copy_paste=0.5 \
  close_mosaic=20 \
  degrees=10 \
  translate=0.15 \
  scale=0.6 \
  shear=3.0 \
  perspective=0.0008 \
  flipud=0.5 \
  fliplr=0.5 \
  hsv_h=0.0 \
  hsv_s=0.0 \
  hsv_v=0.0 \
  label_smoothing=0.08 \
  dropout=0.15 \
  multi_scale=True \
  pretrained=True \
  patience=80 \
  project=urpc_enhanced_v2 \
  name=yolo11n_lu2net_small_obj
```

## Kết quả kỳ vọng (với Enhanced Images)

### Với V1 (Balanced):
```
Current (original params + enhanced):
- Overall mAP50: 0.761, mAP50-95: 0.439
- Scallop Recall: 0.457 (PROBLEM!)
- Holothurian mAP50: 0.695

Expected with V1 optimized:
- Overall mAP50: 0.80-0.83 (+5-9%)
- Overall mAP50-95: 0.48-0.52 (+9-18%)
- Scallop Recall: 0.65-0.72 (+42-58%) ← Main improvement!
- Holothurian mAP50: 0.75-0.80 (+8-15%)
```

### Với V2 (Small Objects):
```
Expected (if scallops are small):
- Overall mAP50: 0.83-0.86 (+9-13%)
- Overall mAP50-95: 0.52-0.57 (+18-30%)
- Scallop Recall: 0.72-0.78 (+58-71%)
- Holothurian mAP50: 0.78-0.82 (+12-18%)
```

## Thay đổi chính so với config cũ

| Parameter | Config cũ | Config mới (V1) | Thay đổi |
|-----------|-----------|-----------------|----------|
| **epochs** | 150 | **200** | +33% |
| **batch** | 12 | **16** | +33% (faster training) |
| **lr0** | 0.0013 | **0.002** | +54% (faster convergence) |
| **lrf** | 0.012 | **0.01** | -17% |
| **warmup_epochs** | 5 | **10** | +100% (better stability) |
| **mosaic** | 0.7 | **1.0** | +43% (class balance!) |
| **mixup** | 0.05 | **0.15** | +200% |
| **copy_paste** | 0.1 | **0.3** | +200% (cho holothurian!) |
| **degrees** | 5 | **15** | +200% |
| **translate** | 0.08 | **0.2** | +150% |
| **scale** | 0.5 | **0.7** | +40% |
| **shear** | 2 | **5** | +150% |
| **flipud** | 0 | **0.5** | NEW! (underwater important) |
| **hsv_h** | 0.015 | **0.0** | TẮT (enhanced images!) |
| **hsv_s** | 0.6 | **0.0** | TẮT |
| **hsv_v** | 0.4 | **0.0** | TẮT |
| **label_smoothing** | 0.02 | **0.05** | +150% |
| **dropout** | - | **0.1** | NEW! |
| **multi_scale** | - | **True** | NEW! |

## QUAN TRỌNG: Nếu muốn thử nhẹ HSV

Nếu bạn lo model quá fit với enhanced style, có thể dùng **HSV RẤT NHẸ**:

```bash
# Thay vì hsv=0.0, dùng:
hsv_h=0.005  # RẤT nhẹ (thay vì 0.015-0.025)
hsv_s=0.1    # RẤT nhẹ (thay vì 0.6-0.8)
hsv_v=0.1    # RẤT nhẹ (thay vì 0.4-0.6)
```

Nhưng theo kinh nghiệm, **TẮT HOÀN TOÀN (0.0) tốt hơn** với enhanced images!

## Monitoring Training

Sau mỗi 10 epochs, kiểm tra:

```bash
# Xem tensorboard
tensorboard --logdir urpc_enhanced_v1/yolo11n_lu2net_balanced

# Hoặc xem results.csv
cat urpc_enhanced_v1/yolo11n_lu2net_balanced/results.csv
```

Metrics cần theo dõi:
- **val/scallop_recall** → Target: >0.65
- **val/holothurian_mAP50** → Target: >0.75
- **val/mAP50-95** → Target: >0.48

## Timeline

Với RTX 3060 (11GB):
- V1 (200 epochs, imgsz=896, batch=16): ~6-7 hours
- V2 (250 epochs, imgsz=1024, batch=12): ~10-12 hours

## Chiến lược

1. **Chạy V1 trước** (balanced, stable)
2. **Nếu scallop recall vẫn <0.65** → Chạy V2 (small objects focus)
3. **So sánh results**
4. **Pick best model** hoặc ensemble

## Notes

- ✅ **Đã tắt HSV aug** vì enhanced images
- ✅ **Tăng mosaic/mixup/copy_paste** cho class balance
- ✅ **Giữ nguyên geometric aug**
- ✅ **Thêm flipud** cho underwater
- ✅ **Thêm dropout** chống overfit
- ✅ **Multi-scale training**
