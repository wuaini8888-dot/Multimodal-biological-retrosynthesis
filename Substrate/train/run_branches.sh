#!/bin/bash

# Ensure the script stops immediately if any command fails
set -e

ORIGINAL_DIR="/data/stu1/ml_project/bioec_retro1/FF/Finger3"
CURRENT_DIR="/data/stu1/ml_project/bioec_retro1/train1"

echo "=================================================="
echo "Starting training for branch model 2 (Seed: 20260205)"
echo "=================================================="
python $ORIGINAL_DIR/train.py \
  --config1 $ORIGINAL_DIR/train_stage1.yaml \
  --config2 $CURRENT_DIR/train_stage2_seed2.yaml

echo "=================================================="
echo "Starting training for branch model 3 (Seed: 20260527)"
echo "=================================================="
python $ORIGINAL_DIR/train.py \
  --config1 $ORIGINAL_DIR/train_stage1.yaml \
  --config2 $CURRENT_DIR/train_stage2_seed3.yaml

echo "=================================================="
echo "Starting training for branch model 4 (Seed: 20260305)"
echo "=================================================="
python $ORIGINAL_DIR/train.py \
  --config1 $ORIGINAL_DIR/train_stage1.yaml \
  --config2 $CURRENT_DIR/train_stage2_seed4.yaml

echo "All branch models have finished fine-tuning successfully!"