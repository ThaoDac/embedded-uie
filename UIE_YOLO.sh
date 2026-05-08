#!/bin/bash
################################################################################
# UIE + YOLO Object Detection Evaluation Pipeline
################################################################################
# Evaluates the impact of underwater image enhancement (UIE) on object detection
# Each UIE model enhances images -> YOLO detects -> saves metrics -> cleans up
################################################################################

set -e

BASE_DIR="/home/ndpthao/eject/IMPLEMENTATION"
DATASET_DIR="${BASE_DIR}/Dataset/URPC2020"
INPUT_DIR="${DATASET_DIR}/test/images"
LABELS_DIR="${DATASET_DIR}/test/labels"
YOLO_WEIGHTS="${BASE_DIR}/YOLOv11-main/best.pt"
OUTPUT_BASE_DIR="${BASE_DIR}/UIE_YOLO_Results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${OUTPUT_BASE_DIR}/run_${TIMESTAMP}"
REPORT_FILE="${RUN_DIR}/UIE_YOLO_Report.md"

mkdir -p "${RUN_DIR}"

echo "================================================================================"
echo "UIE + YOLO OBJECT DETECTION EVALUATION PIPELINE"
echo "================================================================================"
echo "Start time: $(date)"
echo "Input images: ${INPUT_DIR}"
echo "YOLO weights: ${YOLO_WEIGHTS}"
echo "Output: ${RUN_DIR}"
echo "================================================================================"

# Initialize report
cat > "${REPORT_FILE}" << 'EOF'
# UIE + YOLO Object Detection Evaluation Report

This report evaluates the impact of Underwater Image Enhancement (UIE) methods on YOLOv11 object detection performance.

## Evaluation Configuration

- **Dataset:** URPC2020 Test Set
- **YOLO Model:** YOLOv11-nano (best.pt)
- **Classes:** holothurian, echinus, scallop, starfish (4 classes)

## Results

| UIE Method | AP@50 | mAP50-95 | Precision | Recall |
|------------|-------|----------|-----------|--------|
EOF

# Function to run UIE and YOLO evaluation
run_uie_yolo() {
    local UIE_NAME=$1
    local UIE_CMD=$2

    echo ""
    echo "================================================================================"
    echo "Processing: ${UIE_NAME}"
    echo "================================================================================"

    TEMP_DIR="${RUN_DIR}/temp_${UIE_NAME}"
    TEMP_IMAGES="${TEMP_DIR}/images"
    TEMP_LABELS="${TEMP_DIR}/labels"

    mkdir -p "${TEMP_IMAGES}"
    ln -sf "${LABELS_DIR}"/* "${TEMP_DIR}/" 2>/dev/null || cp -r "${LABELS_DIR}" "${TEMP_DIR}/"
    mkdir -p "${TEMP_LABELS}"
    ln -sf "${LABELS_DIR}"/* "${TEMP_LABELS}/" 2>/dev/null || cp "${LABELS_DIR}"/* "${TEMP_LABELS}/"

    # Run UIE enhancement
    echo "Running UIE: ${UIE_NAME}..."
    eval ${UIE_CMD}

    # Create temp data.yaml
    TEMP_YAML="${TEMP_DIR}/data.yaml"
    cat > "${TEMP_YAML}" << YAML
train: ${DATASET_DIR}/train/images
val: ${DATASET_DIR}/valid/images
test: ${TEMP_IMAGES}

nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
YAML

    # Run YOLO detection
    echo "Running YOLO detection..."
    YOLO_OUTPUT="${RUN_DIR}/yolo_${UIE_NAME}"

    yolo detect val \
        model="${YOLO_WEIGHTS}" \
        data="${TEMP_YAML}" \
        split=test \
        save=False \
        project="${YOLO_OUTPUT}" \
        name="results" \
        exist_ok=True \
        verbose=False 2>&1 | tee "${YOLO_OUTPUT}_log.txt"

    # Extract metrics from YOLO output
    AP50=$(grep -oP "all\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+\K[\d.]+" "${YOLO_OUTPUT}_log.txt" | head -1 || echo "N/A")
    MAP50_95=$(grep -oP "all\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\K[\d.]+" "${YOLO_OUTPUT}_log.txt" | head -1 || echo "N/A")
    PRECISION=$(grep -oP "all\s+\d+\s+\d+\s+\K[\d.]+" "${YOLO_OUTPUT}_log.txt" | head -1 || echo "N/A")
    RECALL=$(grep -oP "all\s+\d+\s+\d+\s+[\d.]+\s+\K[\d.]+" "${YOLO_OUTPUT}_log.txt" | head -1 || echo "N/A")

    # Parse metrics from results
    if [ -f "${YOLO_OUTPUT}/results/results.csv" ]; then
        # Get last line metrics
        METRICS_LINE=$(tail -1 "${YOLO_OUTPUT}/results/results.csv")
        # Extract values (format depends on YOLO version)
    fi

    # Append to report
    echo "| ${UIE_NAME} | ${AP50:-N/A} | ${MAP50_95:-N/A} | ${PRECISION:-N/A} | ${RECALL:-N/A} |" >> "${REPORT_FILE}"

    echo "Completed: ${UIE_NAME}"
    echo "Cleaning up temporary files..."
    rm -rf "${TEMP_DIR}"
    rm -f "${YOLO_OUTPUT}_log.txt"

    echo "Done with ${UIE_NAME}"
}

# ===============================================================================
# Run Original (No Enhancement) as Baseline
# ===============================================================================
echo ""
echo "================================================================================"
echo "Processing: Original (No Enhancement)"
echo "================================================================================"

TEMP_YAML_ORIG="${RUN_DIR}/data_original.yaml"
cat > "${TEMP_YAML_ORIG}" << YAML
train: ${DATASET_DIR}/train/images
val: ${DATASET_DIR}/valid/images
test: ${INPUT_DIR}

nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
YAML

YOLO_OUTPUT_ORIG="${RUN_DIR}/yolo_Original"
yolo detect val \
    model="${YOLO_WEIGHTS}" \
    data="${TEMP_YAML_ORIG}" \
    split=test \
    save=False \
    project="${YOLO_OUTPUT_ORIG}" \
    name="results" \
    exist_ok=True 2>&1 | tee "${RUN_DIR}/original_log.txt"

echo "| Original | - | - | - | - |" >> "${REPORT_FILE}"

# ===============================================================================
# DCP
# ===============================================================================
TEMP_DCP="${RUN_DIR}/temp_DCP/images"
mkdir -p "${TEMP_DCP}"
mkdir -p "${RUN_DIR}/temp_DCP/labels"
ln -sf "${LABELS_DIR}"/* "${RUN_DIR}/temp_DCP/labels/"

cd "${BASE_DIR}/DCP-main"
python3 app_console.py --input "${INPUT_DIR}" --output "${TEMP_DCP}" 2>/dev/null || true

TEMP_YAML_DCP="${RUN_DIR}/temp_DCP/data.yaml"
cat > "${TEMP_YAML_DCP}" << YAML
train: ${DATASET_DIR}/train/images
val: ${DATASET_DIR}/valid/images
test: ${TEMP_DCP}
nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
YAML

yolo detect val model="${YOLO_WEIGHTS}" data="${TEMP_YAML_DCP}" split=test save=False project="${RUN_DIR}/yolo_DCP" name="results" exist_ok=True 2>&1 | tee "${RUN_DIR}/DCP_log.txt"
echo "| DCP | - | - | - | - |" >> "${REPORT_FILE}"
rm -rf "${RUN_DIR}/temp_DCP"

# ===============================================================================
# FUnIE-GAN
# ===============================================================================
TEMP_FUNIEGAN="${RUN_DIR}/temp_FUnIE-GAN/images"
mkdir -p "${TEMP_FUNIEGAN}"
mkdir -p "${RUN_DIR}/temp_FUnIE-GAN/labels"
ln -sf "${LABELS_DIR}"/* "${RUN_DIR}/temp_FUnIE-GAN/labels/"

cd "${BASE_DIR}/FUnIE-GAN-main"
python3 app_console_pytorch.py --input "${INPUT_DIR}" --output "${TEMP_FUNIEGAN}" --device cpu 2>/dev/null || true

TEMP_YAML_FUNIEGAN="${RUN_DIR}/temp_FUnIE-GAN/data.yaml"
cat > "${TEMP_YAML_FUNIEGAN}" << YAML
train: ${DATASET_DIR}/train/images
val: ${DATASET_DIR}/valid/images
test: ${TEMP_FUNIEGAN}
nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
YAML

yolo detect val model="${YOLO_WEIGHTS}" data="${TEMP_YAML_FUNIEGAN}" split=test save=False project="${RUN_DIR}/yolo_FUnIE-GAN" name="results" exist_ok=True 2>&1 | tee "${RUN_DIR}/FUnIE-GAN_log.txt"
echo "| FUnIE-GAN | - | - | - | - |" >> "${REPORT_FILE}"
rm -rf "${RUN_DIR}/temp_FUnIE-GAN"

# ===============================================================================
# HCLR-Net
# ===============================================================================
TEMP_HCLR="${RUN_DIR}/temp_HCLR-Net/images"
mkdir -p "${TEMP_HCLR}"
mkdir -p "${RUN_DIR}/temp_HCLR-Net/labels"
ln -sf "${LABELS_DIR}"/* "${RUN_DIR}/temp_HCLR-Net/labels/"

cd "${BASE_DIR}/HCLR-Net-main"
python3 app_console.py --input "${INPUT_DIR}" --output "${TEMP_HCLR}" --device cpu 2>/dev/null || true

TEMP_YAML_HCLR="${RUN_DIR}/temp_HCLR-Net/data.yaml"
cat > "${TEMP_YAML_HCLR}" << YAML
train: ${DATASET_DIR}/train/images
val: ${DATASET_DIR}/valid/images
test: ${TEMP_HCLR}
nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
YAML

yolo detect val model="${YOLO_WEIGHTS}" data="${TEMP_YAML_HCLR}" split=test save=False project="${RUN_DIR}/yolo_HCLR-Net" name="results" exist_ok=True 2>&1 | tee "${RUN_DIR}/HCLR-Net_log.txt"
echo "| HCLR-Net | - | - | - | - |" >> "${REPORT_FILE}"
rm -rf "${RUN_DIR}/temp_HCLR-Net"

# ===============================================================================
# LU2Net
# ===============================================================================
TEMP_LU2NET="${RUN_DIR}/temp_LU2Net/images"
mkdir -p "${TEMP_LU2NET}"
mkdir -p "${RUN_DIR}/temp_LU2Net/labels"
ln -sf "${LABELS_DIR}"/* "${RUN_DIR}/temp_LU2Net/labels/"

cd "${BASE_DIR}/LU2Net-master"
python3 app_console.py --input "${INPUT_DIR}" --output "${TEMP_LU2NET}" --device cpu 2>/dev/null || true

TEMP_YAML_LU2NET="${RUN_DIR}/temp_LU2Net/data.yaml"
cat > "${TEMP_YAML_LU2NET}" << YAML
train: ${DATASET_DIR}/train/images
val: ${DATASET_DIR}/valid/images
test: ${TEMP_LU2NET}
nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
YAML

yolo detect val model="${YOLO_WEIGHTS}" data="${TEMP_YAML_LU2NET}" split=test save=False project="${RUN_DIR}/yolo_LU2Net" name="results" exist_ok=True 2>&1 | tee "${RUN_DIR}/LU2Net_log.txt"
echo "| LU2Net | - | - | - | - |" >> "${REPORT_FILE}"
rm -rf "${RUN_DIR}/temp_LU2Net"

# ===============================================================================
# PGHS
# ===============================================================================
TEMP_PGHS="${RUN_DIR}/temp_PGHS/images"
mkdir -p "${TEMP_PGHS}"
mkdir -p "${RUN_DIR}/temp_PGHS/labels"
ln -sf "${LABELS_DIR}"/* "${RUN_DIR}/temp_PGHS/labels/"

cd "${BASE_DIR}/PGHS-main"
python3 app_console.py --input "${INPUT_DIR}" --output "${TEMP_PGHS}" --device cpu 2>/dev/null || true

TEMP_YAML_PGHS="${RUN_DIR}/temp_PGHS/data.yaml"
cat > "${TEMP_YAML_PGHS}" << YAML
train: ${DATASET_DIR}/train/images
val: ${DATASET_DIR}/valid/images
test: ${TEMP_PGHS}
nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
YAML

yolo detect val model="${YOLO_WEIGHTS}" data="${TEMP_YAML_PGHS}" split=test save=False project="${RUN_DIR}/yolo_PGHS" name="results" exist_ok=True 2>&1 | tee "${RUN_DIR}/PGHS_log.txt"
echo "| PGHS | - | - | - | - |" >> "${REPORT_FILE}"
rm -rf "${RUN_DIR}/temp_PGHS"

# ===============================================================================
# RGHS
# ===============================================================================
TEMP_RGHS="${RUN_DIR}/temp_RGHS/images"
mkdir -p "${TEMP_RGHS}"
mkdir -p "${RUN_DIR}/temp_RGHS/labels"
ln -sf "${LABELS_DIR}"/* "${RUN_DIR}/temp_RGHS/labels/"

cd "${BASE_DIR}/RGHS"
python3 app_console.py --input "${INPUT_DIR}" --output "${TEMP_RGHS}" 2>/dev/null || true

TEMP_YAML_RGHS="${RUN_DIR}/temp_RGHS/data.yaml"
cat > "${TEMP_YAML_RGHS}" << YAML
train: ${DATASET_DIR}/train/images
val: ${DATASET_DIR}/valid/images
test: ${TEMP_RGHS}
nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
YAML

yolo detect val model="${YOLO_WEIGHTS}" data="${TEMP_YAML_RGHS}" split=test save=False project="${RUN_DIR}/yolo_RGHS" name="results" exist_ok=True 2>&1 | tee "${RUN_DIR}/RGHS_log.txt"
echo "| RGHS | - | - | - | - |" >> "${REPORT_FILE}"
rm -rf "${RUN_DIR}/temp_RGHS"

# ===============================================================================
# UDCP
# ===============================================================================
TEMP_UDCP="${RUN_DIR}/temp_UDCP/images"
mkdir -p "${TEMP_UDCP}"
mkdir -p "${RUN_DIR}/temp_UDCP/labels"
ln -sf "${LABELS_DIR}"/* "${RUN_DIR}/temp_UDCP/labels/"

cd "${BASE_DIR}/UDCP-main"
python3 app_console.py --input "${INPUT_DIR}" --output "${TEMP_UDCP}" 2>/dev/null || true

TEMP_YAML_UDCP="${RUN_DIR}/temp_UDCP/data.yaml"
cat > "${TEMP_YAML_UDCP}" << YAML
train: ${DATASET_DIR}/train/images
val: ${DATASET_DIR}/valid/images
test: ${TEMP_UDCP}
nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
YAML

yolo detect val model="${YOLO_WEIGHTS}" data="${TEMP_YAML_UDCP}" split=test save=False project="${RUN_DIR}/yolo_UDCP" name="results" exist_ok=True 2>&1 | tee "${RUN_DIR}/UDCP_log.txt"
echo "| UDCP | - | - | - | - |" >> "${REPORT_FILE}"
rm -rf "${RUN_DIR}/temp_UDCP"

# ===============================================================================
# UIR-PolyKernel
# ===============================================================================
TEMP_UIRPOLY="${RUN_DIR}/temp_UIR-PolyKernel/images"
mkdir -p "${TEMP_UIRPOLY}"
mkdir -p "${RUN_DIR}/temp_UIR-PolyKernel/labels"
ln -sf "${LABELS_DIR}"/* "${RUN_DIR}/temp_UIR-PolyKernel/labels/"

cd "${BASE_DIR}/UIR-PolyKernel-main"
python3 app_console.py --input "${INPUT_DIR}" --output "${TEMP_UIRPOLY}" --device cpu 2>/dev/null || true

TEMP_YAML_UIRPOLY="${RUN_DIR}/temp_UIR-PolyKernel/data.yaml"
cat > "${TEMP_YAML_UIRPOLY}" << YAML
train: ${DATASET_DIR}/train/images
val: ${DATASET_DIR}/valid/images
test: ${TEMP_UIRPOLY}
nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
YAML

yolo detect val model="${YOLO_WEIGHTS}" data="${TEMP_YAML_UIRPOLY}" split=test save=False project="${RUN_DIR}/yolo_UIR-PolyKernel" name="results" exist_ok=True 2>&1 | tee "${RUN_DIR}/UIR-PolyKernel_log.txt"
echo "| UIR-PolyKernel | - | - | - | - |" >> "${REPORT_FILE}"
rm -rf "${RUN_DIR}/temp_UIR-PolyKernel"

# ===============================================================================
# ULAP
# ===============================================================================
TEMP_ULAP="${RUN_DIR}/temp_ULAP/images"
mkdir -p "${TEMP_ULAP}"
mkdir -p "${RUN_DIR}/temp_ULAP/labels"
ln -sf "${LABELS_DIR}"/* "${RUN_DIR}/temp_ULAP/labels/"

cd "${BASE_DIR}/ULAP-main"
python3 app_console.py --input "${INPUT_DIR}" --output "${TEMP_ULAP}" 2>/dev/null || true

TEMP_YAML_ULAP="${RUN_DIR}/temp_ULAP/data.yaml"
cat > "${TEMP_YAML_ULAP}" << YAML
train: ${DATASET_DIR}/train/images
val: ${DATASET_DIR}/valid/images
test: ${TEMP_ULAP}
nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
YAML

yolo detect val model="${YOLO_WEIGHTS}" data="${TEMP_YAML_ULAP}" split=test save=False project="${RUN_DIR}/yolo_ULAP" name="results" exist_ok=True 2>&1 | tee "${RUN_DIR}/ULAP_log.txt"
echo "| ULAP | - | - | - | - |" >> "${REPORT_FILE}"
rm -rf "${RUN_DIR}/temp_ULAP"

# ===============================================================================
# WaterFormer
# ===============================================================================
TEMP_WATERFORMER="${RUN_DIR}/temp_WaterFormer/images"
mkdir -p "${TEMP_WATERFORMER}"
mkdir -p "${RUN_DIR}/temp_WaterFormer/labels"
ln -sf "${LABELS_DIR}"/* "${RUN_DIR}/temp_WaterFormer/labels/"

cd "${BASE_DIR}/WaterFormer-master"
python3 app_console.py --input "${INPUT_DIR}" --output "${TEMP_WATERFORMER}" --device cpu --resize 256 2>/dev/null || true

TEMP_YAML_WATERFORMER="${RUN_DIR}/temp_WaterFormer/data.yaml"
cat > "${TEMP_YAML_WATERFORMER}" << YAML
train: ${DATASET_DIR}/train/images
val: ${DATASET_DIR}/valid/images
test: ${TEMP_WATERFORMER}
nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
YAML

yolo detect val model="${YOLO_WEIGHTS}" data="${TEMP_YAML_WATERFORMER}" split=test save=False project="${RUN_DIR}/yolo_WaterFormer" name="results" exist_ok=True 2>&1 | tee "${RUN_DIR}/WaterFormer_log.txt"
echo "| WaterFormer | - | - | - | - |" >> "${REPORT_FILE}"
rm -rf "${RUN_DIR}/temp_WaterFormer"

# ===============================================================================
# CLAHE
# ===============================================================================
TEMP_CLAHE="${RUN_DIR}/temp_CLAHE/images"
mkdir -p "${TEMP_CLAHE}"
mkdir -p "${RUN_DIR}/temp_CLAHE/labels"
ln -sf "${LABELS_DIR}"/* "${RUN_DIR}/temp_CLAHE/labels/"

cd "${BASE_DIR}/CLAHE-main"
python3 app_console.py --input "${INPUT_DIR}" --output "${TEMP_CLAHE}" 2>/dev/null || true

TEMP_YAML_CLAHE="${RUN_DIR}/temp_CLAHE/data.yaml"
cat > "${TEMP_YAML_CLAHE}" << YAML
train: ${DATASET_DIR}/train/images
val: ${DATASET_DIR}/valid/images
test: ${TEMP_CLAHE}
nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
YAML

yolo detect val model="${YOLO_WEIGHTS}" data="${TEMP_YAML_CLAHE}" split=test save=False project="${RUN_DIR}/yolo_CLAHE" name="results" exist_ok=True 2>&1 | tee "${RUN_DIR}/CLAHE_log.txt"
echo "| CLAHE | - | - | - | - |" >> "${REPORT_FILE}"
rm -rf "${RUN_DIR}/temp_CLAHE"

# ===============================================================================
# Generate Final Report with Parsed Metrics
# ===============================================================================
echo ""
echo "================================================================================"
echo "Generating Final Report with Parsed Metrics..."
echo "================================================================================"

cd "${BASE_DIR}"

python3 << 'PYTHON_SCRIPT'
import os
import re
from pathlib import Path

run_dir = os.environ.get('RUN_DIR', '.')
report_file = os.path.join(run_dir, 'UIE_YOLO_Report.md')

uie_methods = [
    'Original', 'DCP', 'FUnIE-GAN', 'HCLR-Net', 'LU2Net', 'PGHS',
    'RGHS', 'UDCP', 'UIR-PolyKernel', 'ULAP', 'WaterFormer', 'CLAHE'
]

results = {}

for method in uie_methods:
    log_file = os.path.join(run_dir, f'{method}_log.txt')
    if method == 'Original':
        log_file = os.path.join(run_dir, 'original_log.txt')

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            content = f.read()

        # Parse metrics from YOLO output
        # Looking for line like: "all    1234    5678    0.123    0.456    0.789    0.012"
        match = re.search(r'all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content)
        if match:
            precision = float(match.group(1))
            recall = float(match.group(2))
            ap50 = float(match.group(3))
            map50_95 = float(match.group(4))
            results[method] = {
                'AP50': ap50,
                'mAP50-95': map50_95,
                'Precision': precision,
                'Recall': recall
            }
        else:
            results[method] = {'AP50': 'N/A', 'mAP50-95': 'N/A', 'Precision': 'N/A', 'Recall': 'N/A'}
    else:
        results[method] = {'AP50': 'N/A', 'mAP50-95': 'N/A', 'Precision': 'N/A', 'Recall': 'N/A'}

# Write final report
with open(report_file, 'w') as f:
    f.write('# UIE + YOLO Object Detection Evaluation Report\n\n')
    f.write('This report evaluates the impact of Underwater Image Enhancement (UIE) methods on YOLOv11 object detection performance.\n\n')
    f.write('## Evaluation Configuration\n\n')
    f.write('- **Dataset:** URPC2020 Test Set\n')
    f.write('- **YOLO Model:** YOLOv11-nano (best.pt)\n')
    f.write('- **Classes:** holothurian, echinus, scallop, starfish (4 classes)\n\n')
    f.write('## Results\n\n')
    f.write('| UIE Method | AP@50 | mAP50-95 | Precision | Recall |\n')
    f.write('|------------|-------|----------|-----------|--------|\n')

    for method in uie_methods:
        if method in results:
            r = results[method]
            ap50 = f"{r['AP50']:.4f}" if isinstance(r['AP50'], float) else r['AP50']
            map50_95 = f"{r['mAP50-95']:.4f}" if isinstance(r['mAP50-95'], float) else r['mAP50-95']
            precision = f"{r['Precision']:.4f}" if isinstance(r['Precision'], float) else r['Precision']
            recall = f"{r['Recall']:.4f}" if isinstance(r['Recall'], float) else r['Recall']
            f.write(f'| {method} | {ap50} | {map50_95} | {precision} | {recall} |\n')

    f.write('\n## Conclusions\n\n')

    # Find best performer
    best_ap50 = ('N/A', 0)
    for method, r in results.items():
        if isinstance(r['AP50'], float) and r['AP50'] > best_ap50[1]:
            best_ap50 = (method, r['AP50'])

    if best_ap50[0] != 'N/A':
        f.write(f'- **Best AP@50:** {best_ap50[0]} ({best_ap50[1]:.4f})\n')

    f.write('\n---\n')
    f.write('*Generated by UIE_YOLO.sh pipeline*\n')

print(f"Report saved to: {report_file}")
PYTHON_SCRIPT

export RUN_DIR

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETED"
echo "================================================================================"
echo "End time: $(date)"
echo "Report: ${REPORT_FILE}"
echo "================================================================================"
