#!/usr/bin/env bash
# Robust 3-task pipeline runner:
# 1. Train SmolExpert Action Head on Cosmos 3 Edge LoRA Features
# 2. Download Cosmos 7B Transformer Shards & Evaluate on V2 Held-Out
# 3. Extract Cosmos 2B Canonical Multi-Depth Features (4,8,12,16,18,20) & Train SmolExpert

set -Eeuo pipefail
REPO_ROOT="/home/anton/lerobot-video-vam"
cd "$REPO_ROOT"

source scripts/video_vam/cosmos_cuda_env.sh
export PYTHONPATH=".:src"
export PATH="/home/anton/.local/bin:$PATH"
PYTHON="/home/anton/lerobot-video-vam/.venv/bin/python"

mkdir -p outputs/logs outputs/train outputs/features outputs/evaluation

LOG_FILE="outputs/logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "LAUNCHING 3-TASK VAM QUEUE AT $(date)"
echo "Log file: $LOG_FILE"
echo "================================================================================"

run_task() {
    local task_name="$1"
    shift
    echo ""
    echo "================================================================================"
    echo ">>> STARTING: $task_name at $(date)"
    echo "================================================================================"
    set +e
    "$@"
    local status=$?
    set -e
    if [ $status -eq 0 ]; then
        echo ">>> SUCCEEDED: $task_name at $(date)"
    else
        echo ">>> FAILED (exit code $status): $task_name at $(date)"
    fi
    return 0
}

# ------------------------------------------------------------------------------
# 1. Cosmos 3 Edge LoRA SmolExpert Policy Training
# ------------------------------------------------------------------------------
C3_RUN_DIR="outputs/train/cosmos3-edge-lora-smolexpert"
mkdir -p "$C3_RUN_DIR"

run_task "Train SmolExpert on Cosmos 3 Edge LoRA" \
    "$PYTHON" scripts/video_vam/train_smolexpert.py \
    --backbone cosmos3-edge \
    --protocol protocol1 \
    --train-manifest outputs/features/cosmos3-edge-lora/train/manifest.json \
    --val-manifest outputs/features/cosmos3-edge-lora/val/manifest.json \
    --output-dir "$C3_RUN_DIR" \
    --patience 20 \
    --min-steps 10000 \
    --max-steps 50000 \
    --seed 0 \
    --overwrite

# Evaluate Cosmos 3 Edge LoRA on V2 held-out
run_task "Evaluate Cosmos 3 Edge LoRA on V2 Held-Out" \
    "$PYTHON" -c "
import json, subprocess
res = subprocess.run([
    '/home/anton/lerobot-video-vam/.venv/bin/python',
    'scripts/video_vam/eval_v2_heldout_models.py'
])
print('V2 held-out evaluation exit code:', res.returncode)
"

# ------------------------------------------------------------------------------
# 2. Cosmos 7B Download & V2 Held-Out Evaluation
# ------------------------------------------------------------------------------
run_task "Download Cosmos 7B Shards & Evaluate V2" \
    "$PYTHON" scripts/video_vam/download_and_eval_c7b.py

# ------------------------------------------------------------------------------
# 3. Cosmos 2B Multi-Depth Cache & Policy Training
# ------------------------------------------------------------------------------
C2B_DIR="outputs/features/cosmos2b_multidepth"
mkdir -p "$C2B_DIR/train" "$C2B_DIR/val"

if [ ! -f "$C2B_DIR/train/manifest.json" ]; then
    run_task "Extract Cosmos 2B Multi-Depth Train Features" \
        "$PYTHON" scripts/video_vam/build_cosmos_feature_cache.py \
        --root /home/anton/.cache/video-vam/cube-out-of-box-dataset \
        --episodes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
        --stride 3 \
        --state-t 2 \
        --hidden-layers "4,8,12,16,18,20" \
        --checkpoint /home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt \
        --tokenizer /home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth \
        --lora-weights outputs/train/cosmos-video-lora-step6000/best_lora.safetensors \
        --output-dir "$C2B_DIR/train"
fi

if [ ! -f "$C2B_DIR/val/manifest.json" ]; then
    run_task "Extract Cosmos 2B Multi-Depth Val Features" \
        "$PYTHON" scripts/video_vam/build_cosmos_feature_cache.py \
        --root /home/anton/.cache/video-vam/cube-out-of-box-dataset \
        --episodes 32 33 34 35 36 37 38 39 \
        --stride 20 \
        --state-t 2 \
        --hidden-layers "4,8,12,16,18,20" \
        --checkpoint /home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt \
        --tokenizer /home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth \
        --lora-weights outputs/train/cosmos-video-lora-step6000/best_lora.safetensors \
        --output-dir "$C2B_DIR/val"
fi

if [ -f "$C2B_DIR/train/manifest.json" ] && [ -f "$C2B_DIR/val/manifest.json" ]; then
    run_task "Train SmolExpert on Cosmos 2B Multi-Depth" \
        "$PYTHON" scripts/video_vam/train_smolexpert.py \
        --backbone cosmos \
        --protocol protocol1 \
        --train-manifest "$C2B_DIR/train/manifest.json" \
        --val-manifest "$C2B_DIR/val/manifest.json" \
        --output-dir outputs/train/cosmos2b-multidepth-smolexpert \
        --patience 20 \
        --min-steps 10000 \
        --max-steps 50000 \
        --seed 0 \
        --overwrite
fi

echo ""
echo "================================================================================"
echo "ALL 3 QUEUED TASKS COMPLETED AT $(date)"
echo "================================================================================"
