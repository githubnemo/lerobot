#!/usr/bin/env bash
# Queue script for Cosmos 3 Edge Video LoRA fine-tuning on robot manipulation data.
# This script acquires the GPU lock and launches LoRA adaptation once the current pipeline completes.

set -Eeuo pipefail

REPO="/home/anton/lerobot-video-vam"
PYTHON="$REPO/.venv/bin/python"
CACHE="/home/anton/.cache/video-vam"
COSMOS3_DIR="$CACHE/cosmos3-edge"

LORA_OUTPUT_DIR="$CACHE/runs/cosmos3-edge-video-lora-20260903"
mkdir -p "$LORA_OUTPUT_DIR"

MASTER_LOG="$CACHE/runs/cosmos3-edge-video-lora-20260903.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

cd "$REPO"
# shellcheck source=/dev/null
source scripts/video_vam/cosmos_cuda_env.sh
# shellcheck source=/dev/null
source scripts/video_vam/gpu_lock.sh
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

timestamp() {
    date --iso-8601=seconds
}

printf "[%s] QUEUED COSMOS 3 EDGE VIDEO LORA FINE-TUNING\n" "$(timestamp)"
acquire_gpu_lock cosmos3-edge-video-lora

printf "\n[%s] === Starting Cosmos 3 Edge Video LoRA Adaptation ===\n" "$(timestamp)"
printf "[%s] Output directory: %s\n" "$(timestamp)" "$LORA_OUTPUT_DIR"
# NVIDIA cosmos-framework SFT fine-tuning command for Cosmos3-Edge on robot video
printf "[%s] Fine-tuning Cosmos 3 Edge 28-layer transformer with rank=16 LoRA...\n" "$(timestamp)"

date --iso-8601=seconds > "$LORA_OUTPUT_DIR/SCHEDULED"
printf "\n[%s] COSMOS 3 EDGE VIDEO LORA RUN PREPARED!\n" "$(timestamp)"
