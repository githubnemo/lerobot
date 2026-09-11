#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/anton/lerobot-video-vam
source scripts/video_vam/cosmos_cuda_env.sh
export PYTHONPATH=".:src"
export PATH="/home/anton/.local/bin:$PATH"
PYTHON="/home/anton/lerobot-video-vam/.venv/bin/python"

LOG_FILE="/home/anton/lerobot-video-vam/outputs/logs/dual_rollouts_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Rollout 1: Cosmos 3 Edge Episode 35 ==="
"$PYTHON" src/lerobot/policies/vam/rollout_harness.py \
    --backbone cosmos3_edge \
    --lora-checkpoint outputs/train/cosmos3-edge-video-lora/best_lora.safetensors \
    --episodes 35

echo "=== Rollout 2: Cosmos 2B Episodes 32 and 35 ==="
"$PYTHON" src/lerobot/policies/vam/rollout_harness.py \
    --backbone cosmos2b \
    --lora-checkpoint outputs/train/cosmos-video-lora-step6000/best_lora.safetensors \
    --episodes 32 35

echo "=== ALL DUAL ROLLOUTS COMPLETED SUCCESSFULLY ==="
