#!/usr/bin/env bash
# ==============================================================================
# ROBUST STANDALONE NIGHTLY QUEUE (RUNS AUTONOMOUSLY ON ABAKUS IN TMUX)
#
# Tasks in order:
# 1. Render Full-Episode Video Comparison for Cosmos 3 Edge on V2 (Eps 32, 35)
# 2. Run Cosmos 7B V2 Held-Out Evaluation (weights are already downloaded)
# 3. Cosmos 2B Multi-Depth Policy Training on extracted layers (4,8,12,16,18,20)
# 4. Cosmos 3 Edge Video-LoRA extended training (5,000 -> 15,000 steps)
# ==============================================================================

set -Eeuo pipefail
REPO_ROOT="/home/anton/lerobot-video-vam"
cd "$REPO_ROOT"

source scripts/video_vam/cosmos_cuda_env.sh
export PYTHONPATH=".:src"
export PATH="/home/anton/.local/bin:$PATH"
PYTHON="/home/anton/lerobot-video-vam/.venv/bin/python"

V2_DATASET="/home/anton/.cache/huggingface/lerobot/hub/datasets--Orellius--cube_out_of_box_v2/snapshots/5d0325cc1412f4774223a0beb528958108814962"
LOG_FILE="$REPO_ROOT/outputs/logs/robust_nightly_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$REPO_ROOT/outputs/logs" "$REPO_ROOT/outputs/evaluation/v2_c3_rollouts"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "LAUNCHING ROBUST NIGHTLY QUEUE AT $(date)"
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
# 1. Cosmos 3 Edge Full-Episode Rollout Comparison on V2 (Episodes 32 and 35)
# ------------------------------------------------------------------------------
run_task "Cosmos 3 Edge V2 Full-Episode Rollouts (Eps 32, 35)" \
    "$PYTHON" src/lerobot/policies/vam/rollout_harness.py \
    --backbone cosmos3_edge \
    --dataset-repo-id Orellius/cube_out_of_box_v2 \
    --dataset-root "$V2_DATASET" \
    --lora-checkpoint outputs/train/v2-cosmos3-edge-video-lora/best_lora.safetensors \
    --episodes 32 35 \
    --output-dir outputs/evaluation/v2_c3_rollouts \
    --steps 25

# ------------------------------------------------------------------------------
# 2. Cosmos 7B V2 Held-Out Evaluation
# ------------------------------------------------------------------------------
run_task "Cosmos 7B V2 Held-Out Evaluation" \
    "$PYTHON" scripts/video_vam/eval_v2_cosmos7b_only.py

# ------------------------------------------------------------------------------
# 3. Cosmos 2B Multi-Depth SmolExpert Training (on extracted layers 4,8,12,16,18,20)
# ------------------------------------------------------------------------------
C2B_DIR="outputs/features/cosmos2b_multidepth"
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

# ------------------------------------------------------------------------------
# 4. Cosmos 3 Edge Video-LoRA Extended Training on V2 (to 15,000 steps)
# ------------------------------------------------------------------------------
TRAIN_EPS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
           40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 \
           70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89)
VAL_EPS=(32 33 34 35 36 37 38 39)

run_task "Cosmos 3 Edge Extended Training on V2 (15,000 steps)" \
    "$PYTHON" scripts/video_vam/train_cosmos3_edge_video_lora.py \
    --protocol scale100 \
    --dataset-repo-id Orellius/cube_out_of_box_v2 \
    --dataset-root "$V2_DATASET" \
    --train-episodes "${TRAIN_EPS[@]}" \
    --val-episodes "${VAL_EPS[@]}" \
    --pathway dual \
    --clip-frames 17 \
    --max-steps 15000 \
    --val-every 500 \
    --patience 15 \
    --output-dir outputs/train/v2-cosmos3-edge-video-lora-15k

echo ""
echo "================================================================================"
echo "ROBUST NIGHTLY QUEUE COMPLETED AT $(date)"
echo "================================================================================"
