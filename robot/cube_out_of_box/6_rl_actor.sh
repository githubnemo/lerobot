#!/bin/bash
# Step 5b: Start the RL Actor (run AFTER learner is ready)
# The robot will explore and learn to take cube out of box

set -e

export HF_HOME="/home/nemo/.cache/pysandbox-lerobot/huggingface"
export DISPLAY=:0.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/rl_config.json"
CLASSIFIER_OUTPUT_DIR="outputs/reward_classifier/cube_out_of_box"

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
echo "    CUBE OUT OF BOX - RL ACTOR"
echo "=============================================="
echo ""
echo "Config: $CONFIG_PATH"
echo "Reward classifier: $CLASSIFIER_PATH"
echo ""
echo "=== KEYBOARD CONTROLS ==="
echo "  0-9:   Give human reward (0.0 to 0.9)"
echo "  '-':   Give punishment (-0.5)"
echo "  'q':   End episode (neutral)"
echo "  's':   End episode (success +1.0)"
echo "  'r':   Discard episode"
echo "  WASD:  Intervene (take control)"
echo "=========================="
echo ""
echo "The reward classifier will auto-detect success."
echo "You can also give additional human feedback with number keys."
echo ""

# Create temp config with correct classifier path
TMP_CONFIG="/tmp/cube_out_of_box_rl_config.json"
sed "s|\"pretrained_path\": \"outputs/reward_classifier/cube_out_of_box\"|\"pretrained_path\": \"$CLASSIFIER_PATH\"|g" "$CONFIG_PATH" > "$TMP_CONFIG"

python -m lerobot.rl.actor --config_path "$TMP_CONFIG"