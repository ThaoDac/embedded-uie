#!/bin/bash
################################################################################
# UIE + YOLO for PGHS and UIR-PolyKernel
################################################################################

set -e

BASE_DIR="/home/ndpthao/eject/IMPLEMENTATION"
DATASET_DIR="${BASE_DIR}/Dataset/URPC2020"
INPUT_DIR="${DATASET_DIR}/test/images"
LABELS_DIR="${DATASET_DIR}/test/labels"
YOLO_WEIGHTS="${BASE_DIR}/YOLOv11-main/best.pt"
OUTPUT_BASE_DIR="${BASE_DIR}/UIE_YOLO_Results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${OUTPUT_BASE_DIR}/run_PGHS_UIRPoly_${TIMESTAMP}"
REPORT_FILE="${RUN_DIR}/UIE_YOLO_Report.md"

mkdir -p "${RUN_DIR}"

echo "================================================================================"
echo "UIE + YOLO: PGHS & UIR-PolyKernel"
echo "================================================================================"
echo "Start time: $(date)"
echo "Output: ${RUN_DIR}"
echo "================================================================================"

# Initialize report
cat > "${REPORT_FILE}" << 'EOF'
# UIE + YOLO Object Detection Evaluation Report (PGHS & UIR-PolyKernel)

## Evaluation Configuration

- **Dataset:** URPC2020 Test Set
- **YOLO Model:** YOLOv11-nano (best.pt)
- **Classes:** holothurian, echinus, scallop, starfish (4 classes)

## Results

| UIE Method | AP@50 | mAP50-95 | Precision | Recall |
|------------|-------|----------|-----------|--------|
EOF

# ===============================================================================
# PGHS
# ===============================================================================
echo ""
echo "================================================================================"
echo "Processing: PGHS"
echo "================================================================================"

TEMP_PGHS="${RUN_DIR}/temp_PGHS"
TEMP_PGHS_IMAGES="${TEMP_PGHS}/images"
TEMP_PGHS_LABELS="${TEMP_PGHS}/labels"

mkdir -p "${TEMP_PGHS_IMAGES}"
mkdir -p "${TEMP_PGHS_LABELS}"

# Copy labels
echo "Copying labels..."
cp "${LABELS_DIR}"/*.txt "${TEMP_PGHS_LABELS}/"

# Run PGHS enhancement
echo "Running PGHS enhancement..."
cd "${BASE_DIR}/PGHS-main"
python3 app_console.py --input "${INPUT_DIR}" --output "${TEMP_PGHS_IMAGES}" --device cpu

# Rename enhanced files to match label names (remove enhanced_ prefix)
echo "Renaming enhanced files to match labels..."
cd "${TEMP_PGHS_IMAGES}"
for f in enhanced_*; do
    if [ -f "$f" ]; then
        newname="${f#enhanced_}"
        mv "$f" "$newname"
    fi
done

# Check counts
echo "Enhanced images count: $(ls -1 *.jpg *.png 2>/dev/null | wc -l)"
echo "Labels count: $(ls -1 ${TEMP_PGHS_LABELS}/*.txt 2>/dev/null | wc -l)"

# Create data.yaml
TEMP_YAML_PGHS="${TEMP_PGHS}/data.yaml"
cat > "${TEMP_YAML_PGHS}" << YAML
train: ${DATASET_DIR}/train/images
val: ${DATASET_DIR}/valid/images
test: ${TEMP_PGHS_IMAGES}

nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
YAML

# Run YOLO detection
echo "Running YOLO detection for PGHS..."
yolo detect val \
    model="${YOLO_WEIGHTS}" \
    data="${TEMP_YAML_PGHS}" \
    split=test \
    save=False \
    project="${RUN_DIR}/yolo_PGHS" \
    name="results" \
    exist_ok=True 2>&1 | tee "${RUN_DIR}/PGHS_log.txt"

# Cleanup PGHS
rm -rf "${TEMP_PGHS}"
echo "PGHS completed and cleaned up."

# ===============================================================================
# UIR-PolyKernel
# ===============================================================================
echo ""
echo "================================================================================"
echo "Processing: UIR-PolyKernel"
echo "================================================================================"

TEMP_UIRPOLY="${RUN_DIR}/temp_UIR-PolyKernel"
TEMP_UIRPOLY_IMAGES="${TEMP_UIRPOLY}/images"
TEMP_UIRPOLY_LABELS="${TEMP_UIRPOLY}/labels"

mkdir -p "${TEMP_UIRPOLY_IMAGES}"
mkdir -p "${TEMP_UIRPOLY_LABELS}"

# Copy labels
echo "Copying labels..."
cp "${LABELS_DIR}"/*.txt "${TEMP_UIRPOLY_LABELS}/"

# Run UIR-PolyKernel enhancement
echo "Running UIR-PolyKernel enhancement..."
cd "${BASE_DIR}/UIR-PolyKernel-main"
python3 app_console.py --input "${INPUT_DIR}" --output "${TEMP_UIRPOLY_IMAGES}" --device cpu

# Rename enhanced files to match label names (remove enhanced_ prefix)
echo "Renaming enhanced files to match labels..."
cd "${TEMP_UIRPOLY_IMAGES}"
for f in enhanced_*; do
    if [ -f "$f" ]; then
        newname="${f#enhanced_}"
        mv "$f" "$newname"
    fi
done

# Check counts
echo "Enhanced images count: $(ls -1 *.jpg *.png 2>/dev/null | wc -l)"
echo "Labels count: $(ls -1 ${TEMP_UIRPOLY_LABELS}/*.txt 2>/dev/null | wc -l)"

# Create data.yaml
TEMP_YAML_UIRPOLY="${TEMP_UIRPOLY}/data.yaml"
cat > "${TEMP_YAML_UIRPOLY}" << YAML
train: ${DATASET_DIR}/train/images
val: ${DATASET_DIR}/valid/images
test: ${TEMP_UIRPOLY_IMAGES}

nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
YAML

# Run YOLO detection
echo "Running YOLO detection for UIR-PolyKernel..."
yolo detect val \
    model="${YOLO_WEIGHTS}" \
    data="${TEMP_YAML_UIRPOLY}" \
    split=test \
    save=False \
    project="${RUN_DIR}/yolo_UIR-PolyKernel" \
    name="results" \
    exist_ok=True 2>&1 | tee "${RUN_DIR}/UIR-PolyKernel_log.txt"

# Cleanup UIR-PolyKernel
rm -rf "${TEMP_UIRPOLY}"
echo "UIR-PolyKernel completed and cleaned up."

# ===============================================================================
# Generate Final Report
# ===============================================================================
echo ""
echo "================================================================================"
echo "Generating Final Report..."
echo "================================================================================"

cd "${BASE_DIR}"

export RUN_DIR

python3 << 'PYTHON_SCRIPT'
import os
import re

run_dir = os.environ.get('RUN_DIR', '.')
report_file = os.path.join(run_dir, 'UIE_YOLO_Report.md')

uie_methods = ['PGHS', 'UIR-PolyKernel']
results = {}

for method in uie_methods:
    log_file = os.path.join(run_dir, f'{method}_log.txt')

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            content = f.read()

        # Parse metrics from YOLO output
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
    f.write('# UIE + YOLO Object Detection Evaluation Report (PGHS & UIR-PolyKernel)\n\n')
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

    f.write('\n---\n')
    f.write('*Generated by UIE_YOLO_PGHS_UIRPoly.sh*\n')

print(f"Report saved to: {report_file}")

# Print results
for method, r in results.items():
    print(f"\n{method}:")
    for k, v in r.items():
        print(f"  {k}: {v}")
PYTHON_SCRIPT

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETED"
echo "================================================================================"
echo "End time: $(date)"
echo "Report: ${REPORT_FILE}"
cat "${REPORT_FILE}"
echo "================================================================================"
