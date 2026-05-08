#!/bin/bash
# WaterFormer UIE Training Script with Config Support
# Trains WaterFormer model on PGHS underwater image dataset

set -e  # Exit on error

cd "$(dirname "$0")"

echo "=========================================="
echo "WaterFormer UIE Training Script"
echo "=========================================="
echo ""

# Check if config exists
CONFIG_FILE="config_train.yaml"
OPT_FILE="configs/uie_waterformer.yml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Unified config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$OPT_FILE" ]; then
    echo "ERROR: WaterFormer opt file not found: $OPT_FILE"
    exit 1
fi

echo "Configuration files:"
echo "  Unified Config : $CONFIG_FILE"
echo "  WaterFormer Opt: $OPT_FILE"
echo ""

# Dataset path
TRAIN_INPUT="/home/ndpthao/eject/IMPLEMENTATION/PGHS-main/Dataset/train/input"
TRAIN_GT="/home/ndpthao/eject/IMPLEMENTATION/PGHS-main/Dataset/train/target"

# Check if dataset exists
if [ ! -d "$TRAIN_INPUT" ] || [ ! -d "$TRAIN_GT" ]; then
    echo "ERROR: Dataset directories not found"
    echo "  Expected: $TRAIN_INPUT and $TRAIN_GT"
    exit 1
fi

# Count images
INPUT_COUNT=$(ls "$TRAIN_INPUT" | wc -l)
GT_COUNT=$(ls "$TRAIN_GT" | wc -l)
echo "Dataset Statistics:"
echo "  Input images  : $INPUT_COUNT"
echo "  Target images : $GT_COUNT"
echo ""

# Create output directories
mkdir -p work_dirs
mkdir -p tb_logger

echo "Training Configuration (from unified config):"
echo "  Config        : $CONFIG_FILE"
echo "  Dataset       : PGHS"
echo "  Epochs        : 300 (from config)"
echo "  Batch Size    : 1 (from config)"
echo "  Learning Rate : 0.0005 (from config)"
echo "  Num Workers   : 4 (from config)"
echo "  Random Seed   : 42 (from config)"
echo ""

echo "Starting training..."
echo "Press Ctrl+C to stop"
echo ""

# Run training with unified config
# Note: WaterFormer requires both --opt (original config) and --config (unified config)
python waterformer/train.py \
  --opt $OPT_FILE \
  --config $CONFIG_FILE \
  --launcher none

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Checkpoints saved in: ./work_dirs/UW_WaterFormer/"
echo "Logs saved in: ./work_dirs/UW_WaterFormer/"
echo ""
