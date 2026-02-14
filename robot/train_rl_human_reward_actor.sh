#!/bin/bash
# RL training with human rewards only (no reward classifier, no dataset)
# Run the learner first, then this actor script

set -e

# Use same HF cache as teleop (where calibration files are stored)
export HF_HOME="/home/nemo/.cache/pysandbox-lerobot/huggingface"
export DISPLAY=:0.0

CONFIG_PATH="$(dirname "$0")/train_rl_human_reward_config.json"

echo "Starting RL Actor with human rewards..."
echo "Config: $CONFIG_PATH"
echo ""
echo "=== HUMAN REWARD KEYS ==="
echo "  0-9: Give reward 0.0 to 0.9"
echo "  (Press during policy execution)"
echo "========================="
echo ""

python -m lerobot.rl.actor --config_path "$CONFIG_PATH"

