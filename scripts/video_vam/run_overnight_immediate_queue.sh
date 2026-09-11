#!/usr/bin/env bash
# Master queue for overnight and immediate VAM tasks:
# 1. Render Cosmos 3 Edge side-by-side video (Base vs LoRA vs GT)
# 2. Download Cosmos 7B shards & evaluate on V2 held-out
# 3. Extract Cosmos 3 Edge LoRA features & train SmolExpert (dual eval V1+V2)
# 4. Extract Cosmos 2B Layers 16+20 multi-layer features & train SmolExpert (dual eval V1+V2)

set -Eeuo pipefail

REPO_ROOT="/home/anton/lerobot-video-vam"
cd "$REPO_ROOT"

source scripts/video_vam/cosmos_cuda_env.sh
export PYTHONPATH=".:src"
export PATH="/home/anton/.local/bin:$PATH"
PYTHON="/home/anton/lerobot-video-vam/.venv/bin/python"

mkdir -p outputs/logs outputs/evaluation outputs/features outputs/train

LOG_FILE="outputs/logs/queue_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "LAUNCHING MASTER VAM OVERNIGHT QUEUE AT $(date)"
echo "Log file: $LOG_FILE"
echo "================================================================================"

run_task() {
    local task_name="$1"
    shift
    echo ""
    echo "================================================================================"
    echo ">>> STARTING TASK: $task_name at $(date)"
    echo "================================================================================"
    set +e
    "$@"
    local status=$?
    set -e
    if [ $status -eq 0 ]; then
        echo ">>> TASK SUCCEEDED: $task_name at $(date)"
    else
        echo ">>> TASK FAILED with exit code $status: $task_name at $(date)"
    fi
    return 0
}

# ------------------------------------------------------------------------------
# Task 1: Render Cosmos 3 Edge Side-by-Side Video Comparison
# ------------------------------------------------------------------------------
run_task "Render Cosmos 3 Edge Side-by-Side Rollout" \
    "$PYTHON" scripts/video_vam/render_cosmos3_edge_rollout.py \
    --checkpoint-dir /home/anton/.cache/video-vam/cosmos3-edge \
    --dataset-root /home/anton/.cache/video-vam/cube-out-of-box-dataset \
    --lora-path outputs/train/cosmos3-edge-video-lora/best_lora.safetensors \
    --output-dir outputs/evaluation \
    --episode 32 \
    --start-frame 4820 \
    --num-frames 17 \
    --steps 25 \
    --seed 42

# ------------------------------------------------------------------------------
# Task 2: Download Cosmos 7B Transformer Shards & Evaluate on V2 Held-Out
# ------------------------------------------------------------------------------
run_task "Download Cosmos 7B Shards and Evaluate V2 Held-Out" \
    "$PYTHON" scripts/video_vam/download_and_eval_c7b.py

# ------------------------------------------------------------------------------
# Task 3: Cosmos 3 Edge Video-LoRA Features Extraction & SmolExpert Policy
# ------------------------------------------------------------------------------
C3_LORA_FEAT="outputs/features/cosmos3-edge-lora"
mkdir -p "$C3_LORA_FEAT"

if [ ! -f "$C3_LORA_FEAT/train/manifest.json" ]; then
    run_task "Extract Cosmos 3 Edge LoRA Train Features" \
        "$PYTHON" scripts/video_vam/extract_cosmos3_edge_pure_vision.py \
        --episodes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
        --stride 3 \
        --lora-checkpoint outputs/train/cosmos3-edge-video-lora/best_lora.safetensors \
        --output-dir "$C3_LORA_FEAT/train"
fi

if [ ! -f "$C3_LORA_FEAT/val/manifest.json" ]; then
    run_task "Extract Cosmos 3 Edge LoRA Val Features" \
        "$PYTHON" scripts/video_vam/extract_cosmos3_edge_pure_vision.py \
        --episodes 32 33 34 35 36 37 38 39 \
        --stride 20 \
        --lora-checkpoint outputs/train/cosmos3-edge-video-lora/best_lora.safetensors \
        --output-dir "$C3_LORA_FEAT/val"
fi

if [ -f "$C3_LORA_FEAT/train/manifest.json" ] && [ -f "$C3_LORA_FEAT/val/manifest.json" ]; then
    run_task "Train SmolExpert on Cosmos 3 Edge LoRA Features" \
        "$PYTHON" scripts/video_vam/train_smolexpert.py \
        --backbone cosmos3-edge \
        --protocol protocol1 \
        --train-manifest "$C3_LORA_FEAT/train/manifest.json" \
        --val-manifest "$C3_LORA_FEAT/val/manifest.json" \
        --eval2-manifest /home/anton/.cache/video-vam/flux2-klein-scale100-cache/eval2/manifest.json \
        --output-dir outputs/train/cosmos3-edge-lora-smolexpert \
        --patience 20 \
        --min-steps 10000 \
        --max-steps 50000 \
        --seed 0
fi

# ------------------------------------------------------------------------------
# Task 4: Cosmos 2B Multi-Layer (16+20) Features Extraction & SmolExpert Policy
# ------------------------------------------------------------------------------
C2B_ML_FEAT="outputs/features/cosmos2b_layers16_20_unpooled"
mkdir -p "$C2B_ML_FEAT"

if [ ! -f "$C2B_ML_FEAT/train/manifest.json" ]; then
    run_task "Extract Cosmos 2B Layers 16+20 Train Features" \
        "$PYTHON" scripts/video_vam/build_cosmos_feature_cache.py \
        --root /home/anton/.cache/video-vam/cube-out-of-box-dataset \
        --episodes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
        --stride 3 \
        --state-t 2 \
        --hidden-layers "16,20" \
        --checkpoint /home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt \
        --tokenizer /home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth \
        --lora-weights outputs/train/cosmos-video-lora-step6000/best_lora.safetensors \
        --output-dir "$C2B_ML_FEAT/train"
fi

if [ ! -f "$C2B_ML_FEAT/val/manifest.json" ]; then
    run_task "Extract Cosmos 2B Layers 16+20 Val Features" \
        "$PYTHON" scripts/video_vam/build_cosmos_feature_cache.py \
        --root /home/anton/.cache/video-vam/cube-out-of-box-dataset \
        --episodes 32 33 34 35 36 37 38 39 \
        --stride 20 \
        --state-t 2 \
        --hidden-layers "16,20" \
        --checkpoint /home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt \
        --tokenizer /home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth \
        --lora-weights outputs/train/cosmos-video-lora-step6000/best_lora.safetensors \
        --output-dir "$C2B_ML_FEAT/val"
fi

if [ -f "$C2B_ML_FEAT/train/manifest.json" ] && [ -f "$C2B_ML_FEAT/val/manifest.json" ]; then
    run_task "Train SmolExpert on Cosmos 2B Layers 16+20" \
        "$PYTHON" scripts/video_vam/train_smolexpert.py \
        --backbone cosmos \
        --protocol protocol1 \
        --train-manifest "$C2B_ML_FEAT/train/manifest.json" \
        --val-manifest "$C2B_ML_FEAT/val/manifest.json" \
        --eval2-manifest /home/anton/.cache/video-vam/flux2-klein-scale100-cache/eval2/manifest.json \
        --output-dir outputs/train/cosmos2b-layers16-20-smolexpert \
        --patience 20 \
        --min-steps 10000 \
        --max-steps 50000 \
        --seed 0
fi

echo ""
echo "================================================================================"
echo "MASTER OVERNIGHT QUEUE COMPLETED AT $(date)"
echo "================================================================================"
