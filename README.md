# Embedded Implementation of Underwater Image Enhancement Algorithms 

The project is split into two phases:

- **Phase 1 — Image Enhancement Benchmark.** Run all 11 UIE models (5 non-DL + 6 DL) on a common underwater image set and compare image quality metrics (PSNR, SSIM, UCIQE, UIQM, NIQE) together with system metrics (FPS, RAM, energy).
- **Phase 2 — UIE → YOLOv11 Object Detection.** Fine-tune YOLOv11 on URPC2020 / RUOD / DUO, then compare detection performance when feeding
  YOLO with **original** images vs **images enhanced** by each UIE model.

---

## 1. Repository layout

```
embedded-uie/│
│ ── Phase 1: 11 UIE models ───────────────────────────────────
├── CLAHE-main/             # 1. CLAHE (non-DL)
├── DCP-main/               # 2. Dark Channel Prior (non-DL)
├── RGHS/                   # 3. Relative Global Histogram Stretching (non-DL)
├── UDCP-main/              # 4. Underwater DCP (non-DL)
├── ULAP-main/              # 5. Underwater Light Attenuation Prior (non-DL)
├── FUnIE-GAN-main/         # 6. Fast Underwater GAN (DL)
├── HCLR-Net-main/          # 7. Hierarchical Context Learning & Refinement (DL)
├── LU2Net-master/          # 8. Lightweight U2-Net (DL)
├── PGHS-main/              # 9. Physics-Guided Hybrid System (DL)
├── UIR-PolyKernel-main/    # 10. Polynomial-Kernel UIR (DL)
├── WaterFormer-master/     # 11. Transformer-based UIE (DL)
│
│ ── Phase 2: YOLOv11 ─────────────────────────────────────────
├── YOLOv11-main/           # train + test + UIE→YOLO pipelines
│
├── UIE_nonDL.py            # run the 5 non-DL UIE models + write MD report
├── UIE_DL.py               # run the 6 DL UIE models + write MD report
├── UIE_YOLO.sh             # workflow: UIE → YOLO over all 11 models
├── UIE_YOLO_PGHS_UIRPoly.sh# subset (PGHS, UIR-PolyKernel) — quick debug
├── analyze_underwater_image.py # per-image histogram / sharpness analysis
└── UIE_YOLO_Results/       # outputs from Phase 2 runs
```

---

## 2. Environment setup

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Each UIE folder ships its own `requirements.txt`, but the dependencies overlap heavily. Install the shared stack first, then add per-model extras only if something is missing:

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install ultralytics==8.3.49 onnx onnxruntime
pip install opencv-python numpy scipy scikit-image Pillow tqdm pyyaml
pip install matplotlib pandas seaborn
pip install pyiqa pytorch_msssim einops
pip install psutil memory_profiler
pip install thop fvcore           # FLOPs / params counters
```

Pretrained weights required before running Phase 1 (all git-ignored):

| Model           | Weight path                                              |
|-----------------|----------------------------------------------------------|
| FUnIE-GAN       | `FUnIE-GAN-main/PyTorch/models/...`                      |
| HCLR-Net        | `HCLR-Net-main/checkpoints/best_psnr/...`                |
| LU2Net          | `LU2Net-master/checkpoints/LightUNet_170.pth`            |
| PGHS            | `PGHS-main/models/uw_epoch_79.pth`                       |
| UIR-PolyKernel  | `UIR-PolyKernel-main/models/UIR_PolyKernel_epoch_311.pth`|
| WaterFormer     | `WaterFormer-master/checkpoints/weights.pth`             |
| YOLOv11         | `YOLOv11-main/yolo11n.pt`, `YOLOv11-main/best.pt`        |

---

## 3. Phase 1 — run the 11 UIE models and compare

### 3.1. Non-DL models (CLAHE, DCP, RGHS, UDCP, ULAP)

```bash
python UIE_nonDL.py \
    --input  Dataset/URPC2020/test/images \
    --output results/phase1_nonDL \
    --gt     Dataset/URPC2020/test/images \
    --report UIE_nonDL.md
```

`--gt` is only needed for full-reference metrics (PSNR/SSIM). Without ground truth the script reports UCIQE / UIQM / NIQE only.

### 3.2. Deep learning models (the remaining 6)

```bash
python UIE_DL.py \
    --input  Dataset/URPC2020/test/images \
    --output results/phase1_DL \
    --gt     Dataset/URPC2020/test/images \
    --report UIE_DL.md
```

Each model can also be invoked individually:

```bash
# Example: run LU2Net
python LU2Net-master/app_console.py \
    --input  Dataset/URPC2020/test/images \
    --output results/lu2net

# Example: run WaterFormer (resize 256 for low-end machines)
python WaterFormer-master/app_console.py \
    --input  Dataset/URPC2020/test/images \
    --output results/waterformer \
    --device cpu --resize 256
```

Result: each `results/phase1_*` directory contains the enhanced images plus a markdown report with the full 11-model comparison.

---

## 4. Phase 2 — fine-tune YOLOv11 and the UIE → Detection pipeline

### 4.1. Fine-tune YOLOv11 on URPC2020 / RUOD

Before training, edit the paths inside `Dataset/<DS>/data.yaml` to match your machine.

```bash
# Option 1: Python script (URPC2020, 4 classes)
python YOLOv11-main/train.py

# Option 2: Ultralytics CLI (recommended — more flexible)
yolo detect train \
    model=YOLOv11-main/yolo11n.pt \
    data=Dataset/URPC2020/data.yaml \
    epochs=150 imgsz=960 batch=12 device=0 \
    optimizer=AdamW lr0=0.0013 lrf=0.012 \
    mosaic=0.7 mixup=0.05 close_mosaic=10 \
    project=YOLOv11-main/urpc_train name=yolov11n_finetune
```

Best weights will be written to
`YOLOv11-main/urpc_train/yolov11n_finetune/weights/best.pt`.
A pre-trained copy can be placed at `YOLOv11-main/best.pt` so the
scripts below can pick it up directly.

### 4.2. Validate the baseline (original images)

```bash
python YOLOv11-main/test.py \
    --model    YOLOv11-main/best.pt \
    --data     Dataset/URPC2020/data.yaml \
    --test-dir Dataset/URPC2020/test \
    --imgsz 640 --device 0
```

### 4.3. Full workflow: UIE → YOLO over the 11 models

The bash script runs sequentially: (1) enhance the test images with each
UIE model → (2) call `yolo detect val` on the enhanced images →
(3) aggregate metrics into
`UIE_YOLO_Results/run_<timestamp>/UIE_YOLO_Report.md`, then clean up
temporary files.

```bash
# Linux / macOS
bash UIE_YOLO.sh

# Subset (PGHS + UIR-PolyKernel only — quick smoke test)
bash UIE_YOLO_PGHS_UIRPoly.sh
```

> On Windows, run these scripts through **Git Bash** or **WSL** because
> they rely on `bash` / `ln -sf`.

Before running, update `BASE_DIR` at the top of each `.sh` to your real
local path (the defaults are Linux paths).

### 4.4. Python pipeline (3-way visualization with bbox overlays)

```bash
# Detailed comparison Original vs each UIE model on YOLO,
# with side-by-side images: [Raw + GT] | [Enhanced + Det] | [Original + Det]
python YOLOv11-main/pipeline_all_uie_models.py

# Single model (e.g. LU2Net PyTorch)
python YOLOv11-main/pipeline_enhance_detect.py \
    --enhance-model pytorch --model-name lu2net \
    --test-dir Dataset/URPC2020/test \
    --yolo-model YOLOv11-main/best.pt \
    --imgsz 640 --device 0
```

Outputs include enhanced images, detection visualizations,
`metrics.yaml`, and a `UIE_YOLO_Report.md` comparing mAP@50,
mAP@50-95, Precision and Recall.

---

## 5. Datasets — original download links

All datasets are listed in `.gitignore` because of their size. After
cloning the repo, download them and place them under `Dataset/`
following the structure below.

### 5.1. Object Detection (Phase 2)

| Dataset | Description | #Classes | Original link |
|---------|-------------|----------|---------------|
| **URPC2020** | Underwater Robot Picking Contest 2020. Main dataset used in the paper. 5,543 train / 1,200 val / 800 test. | 4 (holothurian, echinus, scallop, starfish) | Roboflow mirrors: <https://universe.roboflow.com/search?q=urpc2020> |
| **RUOD** | Real-world Underwater Object Detection (v2 on Roboflow). 13,161 images with corrected annotations. | 10 (corals, cuttlefish, diver, echinus, fish, holothurian, jellyfish, scallop, starfish, turtle) | <https://universe.roboflow.com/marcofarrugia/ruod-tcoz3/dataset/2> · paper / dataset: <https://github.com/dlut-dimt/RUOD> |
| **DUO** | Detecting Underwater Objects, de-duplicated using a perceptual hash. 6,671 train / 1,111 val / 1,111 test. | 4 (holothurian, echinus, scallop, starfish) | <https://github.com/chongweiliu/DUO> |

Each dataset must follow the standard YOLO layout:

```
Dataset/<Name>/
├── data.yaml          # train/val/test paths + nc + names
├── train/{images,labels}
├── valid/{images,labels}
└── test/{images,labels}
```

After downloading, **update the paths inside `data.yaml`** to match your
local machine.

### 5.2. Image Enhancement (UIE training — Phase 1)

The DL UIE models (FUnIE-GAN, HCLR-Net, LU2Net, PGHS, UIR-PolyKernel,
WaterFormer) are trained on paired underwater datasets
(input ↔ reference):

| Dataset | Used by | Original link |
|---------|---------|---------------|
| **EUVP** (Enhancement of Underwater Visual Perception) — 12K paired + 8K unpaired | FUnIE-GAN, UIR-PolyKernel, PGHS, ... | <https://irvlab.cs.umn.edu/resources/euvp-dataset> |
| **UIEB** (Underwater Image Enhancement Benchmark) — 890 raw / 60 challenging | HCLR-Net, LU2Net, WaterFormer, ... | <https://li-chongyi.github.io/proj_benchmark.html> |
| **LSUI** (Large-Scale Underwater Image) — 4,279 paired | WaterFormer, U-Shape Transformer | <https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement> |
| **UFO-120** | FUnIE-GAN-UP (super-resolution variant) | <https://irvlab.cs.umn.edu/resources/ufo-120-dataset> |
| **UVEB** (Underwater Video Enhancement Benchmark) | UVE-Net (extra) | <https://github.com/yzbouc/UVEB> |
