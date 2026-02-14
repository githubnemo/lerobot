#!/bin/bash
# Step 5a: Start the RL Learner (run this FIRST, then actor in another terminal)
# Uses both the reward classifier AND offline demonstrations

set -e

export HF_HOME="/home/nemo/.cache/pysandbox-lerobot/huggingface"
export DISPLAY=:0.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/rl_config.json"
CLASSIFIER_OUTPUT_DIR="outputs/reward_classifier/cube_touch"

# Find the latest reward classifier checkpoint
if [ -L "$CLASSIFIER_OUTPUT_DIR/checkpoints/last" ]; then
    CLASSIFIER_PATH="$CLASSIFIER_OUTPUT_DIR/checkpoints/last/pretrained_model"
else
    LATEST=$(ls -d "$CLASSIFIER_OUTPUT_DIR/checkpoints"/[0-9]* 2>/dev/null | sort -V | tail -1)
    if [ -z "$LATEST" ]; then
        echo "ERROR: No reward classifier checkpoint found in $CLASSIFIER_OUTPUT_DIR"
        echo "Please run ./3_train_reward_classifier.sh first"
        exit 1
    fi
    CLASSIFIER_PATH="$LATEST/pretrained_model"
fi

# Verify checkpoint exists
if [ ! -f "$CLASSIFIER_PATH/config.json" ]; then
    echo "ERROR: config.json not found in $CLASSIFIER_PATH"
    exit 1
fi

echo "=============================================="
echo "    CUBE TOUCH - RL LEARNER"
echo "=============================================="
echo ""
echo "Config: $CONFIG_PATH"
echo "Reward classifier: $CLASSIFIER_PATH"
echo ""
echo "The learner will:"
echo "  1. Load demonstrations into offline buffer"
echo "  2. Wait for actor to connect"
echo "  3. Train SAC policy mixing online + offline data"
echo ""
echo "Start the actor in another terminal: ./6_rl_actor.sh"
echo ""

# Create temp config with correct classifier path
TMP_CONFIG="/tmp/cube_touch_rl_config.json"
sed "s|\"pretrained_path\": \"outputs/reward_classifier/cube_touch\"|\"pretrained_path\": \"$CLASSIFIER_PATH\"|g" "$CONFIG_PATH" > "$TMP_CONFIG"

python -m lerobot.rl.learner --config_path "$TMP_CONFIG"