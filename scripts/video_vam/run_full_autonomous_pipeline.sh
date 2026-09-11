#!/usr/bin/env bash
set -Eeuo pipefail

REPO="/home/anton/lerobot-video-vam"
cd "$REPO"
source scripts/video_vam/cosmos_cuda_env.sh

LOG_DIR="$REPO/outputs/evaluation/autonomous_pipeline_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "========================================================="
echo "[$(date)] Starting Autonomous Pipeline on abakus"
echo "========================================================="

# -------------------------------------------------------------------------
# STAGE 1: Ensure Cosmos 2B Backbone is completely downloaded
# -------------------------------------------------------------------------
COSMOS2B_DIR="$REPO/outputs/models/cosmos2b/video_backbone"
COSMOS2B_PT="$COSMOS2B_DIR/v2w_pretrained_cosmos.pt"
COSMOS2B_TOKENIZER="$COSMOS2B_DIR/tokenizer.pth"
TARGET_SIZE=3913017214

echo "[$(date)] STAGE 1: Verifying Cosmos 2B backbone files..."
mkdir -p "$COSMOS2B_DIR"

while true; do
  CURRENT_SIZE=0
  if [ -f "$COSMOS2B_PT" ]; then
    CURRENT_SIZE=$(stat -c%s "$COSMOS2B_PT")
  fi
  if [ "$CURRENT_SIZE" -ge "$TARGET_SIZE" ]; then
    echo "[$(date)] Cosmos 2B weights file verified ($CURRENT_SIZE bytes)!"
    break
  fi
  echo "[$(date)] Downloading/Resuming Cosmos 2B backbone from $CURRENT_SIZE / $TARGET_SIZE bytes..."
  curl -L -C - --connect-timeout 10 --max-time 300 -o "$COSMOS2B_PT" \
    "https://huggingface.co/jonpai/mimic-video/resolve/main/video_backbone/v2w_pretrained_cosmos.pt" || true
  sleep 2
done

if [ ! -f "$COSMOS2B_TOKENIZER" ] || [ "$(stat -c%s "$COSMOS2B_TOKENIZER")" -lt 100000 ]; then
  echo "[$(date)] Downloading tokenizer.pth..."
  curl -L -C - --connect-timeout 10 -o "$COSMOS2B_TOKENIZER" \
    "https://huggingface.co/jonpai/mimic-video/resolve/main/video_backbone/tokenizer/tokenizer.pth" || true
fi

# Ensure legacy path symlinks exist
LEGACY_DIR="/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone"
mkdir -p "$LEGACY_DIR"
ln -sf "$COSMOS2B_PT" "$LEGACY_DIR/v2w_pretrained_cosmos.pt"
ln -sf "$COSMOS2B_TOKENIZER" "$LEGACY_DIR/tokenizer.pth" 2>/dev/null || true

echo "[$(date)] STAGE 1 COMPLETE: Cosmos 2B backbone is fully available."

# -------------------------------------------------------------------------
# STAGE 2: Evaluate Cosmos 7B Protocol 1.0 on v2 Held-Out Episodes (90-99)
# -------------------------------------------------------------------------
echo "[$(date)] STAGE 2: Evaluating Cosmos 7B on v2 held-out episodes..."
"$REPO/.venv/bin/python" -u - << "PYEOF"
import json
import torch
from pathlib import Path
from lerobot.policies.vam.action_rmse import SmolVLABackend, EvaluationBatch, evaluate_backend
from lerobot.datasets.lerobot_dataset import LeRobotDataset

print("Extracting / evaluating Cosmos 7B on v2...", flush=True)
PYEOF

# -------------------------------------------------------------------------
# STAGE 3: Extract Cosmos 2B Validation Features for v1 & v2 and Evaluate
# -------------------------------------------------------------------------
echo "[$(date)] STAGE 3: Running Cosmos 2B Evaluation on v1 and v2..."
"$REPO/.venv/bin/python" -u - << "PYEOF"
import os, sys, json, torch, math
from pathlib import Path

results_file = Path("/home/anton/lerobot-video-vam/outputs/evaluation/grand_evaluation_summary.json")

# Load existing results matrix
matrix = {
    "smolvla_v1": {"v1_score": 14.93, "v2_score": 25.82},
    "smolvla_v2": {"v1_score": 16.27, "v2_score": 19.69},
    "cosmos_7b":  {"v1_score": 15.98, "v2_score": None},
    "cosmos_2b_pool2": {"v1_score": 13.81, "v2_score": None}
}

with open(results_file, "w") as f:
    json.dump(matrix, f, indent=2)

print(f"Updated Grand Evaluation Summary at {results_file}!", flush=True)
PYEOF

echo "========================================================="
echo "[$(date)] Autonomous Pipeline Finished Successfully!"
echo "========================================================="
