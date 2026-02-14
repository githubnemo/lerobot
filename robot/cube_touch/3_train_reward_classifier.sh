#!/bin/bash
# Step 3: Train a reward classifier on the collected data
# The classifier learns to recognize "cube touched" vs "not touched" from images

set -e

export HF_HOME="/home/nemo/.cache/pysandbox-lerobot/huggingface"
export DISPLAY=:0.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/train_reward_classifier_config.json"
OUTPUT_DIR="outputs/reward_classifier/cube_touch"

echo "=============================================="
echo "    CUBE TOUCH - REWARD CLASSIFIER TRAINING"
echo "=============================================="
echo ""
echo "Config: $CONFIG_PATH"
echo ""

# Check if output directory exists
if [ -d "$OUTPUT_DIR" ]; then
    echo "WARNING: Output directory already exists: $OUTPUT_DIR"
    echo ""
    echo "Options:"
    echo "  [d] Delete and retrain from scratch"
    echo "  [r] Resume training (if checkpoint exists)"
    echo "  [q] Quit"
    echo ""
    read -p "Choose [d/r/q]: " choice
    case "$choice" in
        d|D)
            echo "Deleting existing output directory..."
            rm -rf "$OUTPUT_DIR"
            RESUME_FLAG=""
            ;;
        r|R)
            echo "Resuming training..."
            RESUME_FLAG="--resume=true"
            ;;
        *)
            echo "Exiting."
            exit 0
            ;;
    esac
else
    RESUME_FLAG=""
fi

echo ""
echo "This will train a vision model to recognize:"
echo "  - Class 0: Cube NOT touched (reward=0)"
echo "  - Class 1: Cube touched (reward=1)"
echo ""
echo "Model will be saved to: $OUTPUT_DIR"
echo ""

python -m lerobot.scripts.lerobot_train --config_path "$CONFIG_PATH" $RESUME_FLAG

