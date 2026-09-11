#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="/home/anton/lerobot-video-vam"
cd "$REPO_ROOT"

source scripts/video_vam/cosmos_cuda_env.sh
export PYTHONPATH=".:src"
export PATH="/home/anton/.local/bin:$PATH"
PYTHON="/home/anton/lerobot-video-vam/.venv/bin/python"

LOG_FILE="$REPO_ROOT/outputs/logs/day_queue_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$REPO_ROOT/outputs/logs" "$REPO_ROOT/outputs/train/cosmos2b-multidepth-smolexpert"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "LAUNCHING DAY QUEUE AT $(date)"
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
        echo ">>> TASK COMPLETED: $task_name at $(date)"
    else
        echo ">>> TASK FAILED (status $status): $task_name at $(date)"
    fi
}

# ------------------------------------------------------------------------------
# 1. Train SmolExpert on Cosmos 2B Multi-Depth (Layers 16+20, 4096 channels)
# ------------------------------------------------------------------------------
run_task "Train SmolExpert on Cosmos 2B Layers 16+20" \
    "$PYTHON" scripts/video_vam/train_smolexpert.py \
    --backbone cosmos \
    --protocol protocol1 \
    --train-manifest outputs/features/cosmos2b_multidepth/train/manifest.json \
    --val-manifest outputs/features/cosmos2b_multidepth/val/manifest.json \
    --output-dir outputs/train/cosmos2b-multidepth-smolexpert \
    --patience 20 \
    --min-steps 10000 \
    --max-steps 50000 \
    --seed 0 \
    --overwrite

echo ""
echo "================================================================================"
echo "DAY QUEUE COMPLETED AT $(date)"
echo "================================================================================"
