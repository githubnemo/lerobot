#!/usr/bin/env bash
# ==============================================================================
# COSMOS 3 EDGE VIDEO-LORA: ONLINE VIDEO EXTRACTION WITH COHERENT AUGMENTATIONS
# ==============================================================================
set -Eeuo pipefail

REPO_ROOT="/home/anton/lerobot-video-vam"
cd "$REPO_ROOT"

source scripts/video_vam/cosmos_cuda_env.sh
export PYTHONPATH=".:src"
export PATH="/home/anton/.local/bin:$PATH"
PYTHON="/home/anton/lerobot-video-vam/.venv/bin/python"

mkdir -p outputs/logs outputs/train

LOG_FILE="outputs/logs/c3_online_aug_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

OUTPUT_DIR="outputs/train/v2-cosmos3-edge-lora-online-aug-smolexpert"
LORA_WEIGHTS="outputs/train/v2-cosmos3-edge-video-lora/best_lora.safetensors"
NORM_PATH="outputs/train/v2-cosmos3-edge-lora-smolexpert/normalizer.safetensors"

echo "================================================================================"
echo "LAUNCHING COSMOS 3 EDGE ONLINE AUGMENTED TRAINING AT $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "LoRA weights:     $LORA_WEIGHTS"
echo "Log file:         $LOG_FILE"
echo "================================================================================"

"$PYTHON" scripts/video_vam/train_smolexpert.py \
    --online-backbone cosmos3_edge \
    --backbone-checkpoint /home/anton/.cache/video-vam/cosmos3-edge \
    --backbone-lora-weights "$LORA_WEIGHTS" \
    --backbone-lora-rank 16 \
    --backbone-lora-alpha 32.0 \
    --backbone-layer 20 \
    --dataset-repo-id Orellius/cube_out_of_box_v2 \
    --train-stride 1 \
    --augment \
    --normalizer-path "$NORM_PATH" \
    --val-manifest outputs/features/v2-cosmos3-edge-lora-15k/val/manifest.json \
    --eval2-manifest outputs/features/v2-cosmos3-edge-lora-15k/eval2/manifest.json \
    --output-dir "$OUTPUT_DIR" \
    --protocol scale100 \
    --batch-size 4 \
    --grad-accum-steps 2 \
    --lr 1e-4 \
    --lr-scheduler cosine \
    --warmup-steps 1000 \
    --min-steps 10000 \
    --max-steps 35000 \
    --val-every 500 \
    --patience 20 \
    --seed 42 \
    --overwrite

echo "================================================================================"
echo "COSMOS 3 EDGE ONLINE AUGMENTED TRAINING COMPLETED AT $(date)"
echo "================================================================================"
