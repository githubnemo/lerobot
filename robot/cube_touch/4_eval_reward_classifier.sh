#!/bin/bash
# Step 4: Evaluate the trained reward classifier on the dataset
# Shows confusion matrix and performance metrics

set -e

export HF_HOME="/home/nemo/.cache/pysandbox-lerobot/huggingface"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=============================================="
echo "    CUBE TOUCH - REWARD CLASSIFIER EVALUATION"
echo "=============================================="
echo ""

# Default threshold, can be overridden by passing argument
THRESHOLD=${1:-0.5}

python "${SCRIPT_DIR}/../helpers/eval_classifier_on_dataset.py" \
    --classifier-path "outputs/reward_classifier/cube_touch" \
    --dataset-repo-id "nemo/cube_touch_dataset" \
    --dataset-root "./data/cube_touch_dataset" \
    --threshold "$THRESHOLD"
