#!/usr/bin/env bash
# ==============================================================================
# MASTER OVERNIGHT V2 (SCALE-100) VIDEO-ACTION MODEL BENCHMARK PIPELINE
#
# Dataset: Orellius/cube_out_of_box_v2 (100 episodes)
# - Train Split (82 eps): 0..31, 40..89 (3,065 samples with stride 3)
# - Eval-1 Historical Benchmark (8 eps): 32..39 (88 anchors with stride 20)
# - Eval-2 New Distribution Benchmark (10 eps): 90..99 (51 anchors with stride 20)
# ==============================================================================

set -Eeuo pipefail
REPO_ROOT="/home/anton/lerobot-video-vam"
cd "$REPO_ROOT"

source scripts/video_vam/cosmos_cuda_env.sh
export PYTHONPATH=".:src"
export PATH="/home/anton/.local/bin:$PATH"
PYTHON="/home/anton/lerobot-video-vam/.venv/bin/python"

V2_DATASET="/home/anton/.cache/huggingface/lerobot/hub/datasets--Orellius--cube_out_of_box_v2/snapshots/5d0325cc1412f4774223a0beb528958108814962"
COSMOS3_DIR="/home/anton/.cache/video-vam/cosmos3-edge"

mkdir -p outputs/logs outputs/train outputs/features outputs/evaluation

LOG_FILE="outputs/logs/v2_overnight_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "LAUNCHING V2 (SCALE-100) OVERNIGHT PIPELINE AT $(date)"
echo "Log file: $LOG_FILE"
echo "================================================================================"

TRAIN_EPS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
           40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 \
           70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89)
VAL_EPS=(32 33 34 35 36 37 38 39)
EVAL2_EPS=(90 91 92 93 94 95 96 97 98 99)

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

# ==============================================================================
# PHASE 1: COSMOS 3 EDGE (VIDEO-LORA + DUAL-EVAL SMOLEXPERT ON SCALE-100)
# ==============================================================================
C3_LORA_DIR="outputs/train/v2-cosmos3-edge-video-lora"
C3_FEAT_DIR="outputs/features/v2-cosmos3-edge-lora"
C3_POLICY_DIR="outputs/train/v2-cosmos3-edge-lora-smolexpert"

if [ ! -f "$C3_LORA_DIR/best_lora.safetensors" ]; then
    run_task "Train Cosmos 3 Edge Video-LoRA on V2 (82 episodes, 5000 steps)" \
        "$PYTHON" scripts/video_vam/train_cosmos3_edge_video_lora.py \
        --protocol scale100 \
        --dataset-repo-id Orellius/cube_out_of_box_v2 \
        --dataset-root "$V2_DATASET" \
        --train-episodes "${TRAIN_EPS[@]}" \
        --val-episodes "${VAL_EPS[@]}" \
        --pathway dual \
        --clip-frames 17 \
        --max-steps 5000 \
        --val-every 500 \
        --patience 10 \
        --output-dir "$C3_LORA_DIR"
fi

if [ -f "$C3_LORA_DIR/best_lora.safetensors" ]; then
    # Full-length autoregressive rollouts on 2 validation episodes
    run_task "Render Cosmos 3 Edge V2 Full-Episode Rollouts (Eps 32, 35)" \
        "$PYTHON" src/lerobot/policies/vam/rollout_harness.py \
        --backbone cosmos3_edge \
        --dataset-root "$V2_DATASET" \
        --lora-checkpoint "$C3_LORA_DIR/best_lora.safetensors" \
        --episodes 32 35 \
        --output-dir outputs/evaluation

    # Extract Train, Val, and Eval2 features on V2
    if [ ! -f "$C3_FEAT_DIR/train/manifest.json" ]; then
        run_task "Extract Cosmos 3 Edge V2 Train Features (82 eps, stride 3)" \
            "$PYTHON" scripts/video_vam/extract_cosmos3_edge_pure_vision.py \
            --dataset-repo-id Orellius/cube_out_of_box_v2 \
            --dataset-root "$V2_DATASET" \
            --episodes "${TRAIN_EPS[@]}" \
            --stride 3 \
            --lora-checkpoint "$C3_LORA_DIR/best_lora.safetensors" \
            --output-dir "$C3_FEAT_DIR/train"
    fi

    if [ ! -f "$C3_FEAT_DIR/val/manifest.json" ]; then
        run_task "Extract Cosmos 3 Edge V2 Val Features (8 eps, stride 20)" \
            "$PYTHON" scripts/video_vam/extract_cosmos3_edge_pure_vision.py \
            --dataset-repo-id Orellius/cube_out_of_box_v2 \
            --dataset-root "$V2_DATASET" \
            --episodes "${VAL_EPS[@]}" \
            --stride 20 \
            --lora-checkpoint "$C3_LORA_DIR/best_lora.safetensors" \
            --output-dir "$C3_FEAT_DIR/val"
    fi

    if [ ! -f "$C3_FEAT_DIR/eval2/manifest.json" ]; then
        run_task "Extract Cosmos 3 Edge V2 Eval2 Features (10 eps, stride 20)" \
            "$PYTHON" scripts/video_vam/extract_cosmos3_edge_pure_vision.py \
            --dataset-repo-id Orellius/cube_out_of_box_v2 \
            --dataset-root "$V2_DATASET" \
            --episodes "${EVAL2_EPS[@]}" \
            --stride 20 \
            --lora-checkpoint "$C3_LORA_DIR/best_lora.safetensors" \
            --output-dir "$C3_FEAT_DIR/eval2"
    fi

    if [ -f "$C3_FEAT_DIR/train/manifest.json" ] && [ -f "$C3_FEAT_DIR/val/manifest.json" ] && [ -f "$C3_FEAT_DIR/eval2/manifest.json" ]; then
        run_task "Train SmolExpert on Cosmos 3 Edge LoRA (Scale-100 Dual-Eval)" \
            "$PYTHON" scripts/video_vam/train_smolexpert.py \
            --backbone cosmos3-edge \
            --protocol scale100 \
            --train-manifest "$C3_FEAT_DIR/train/manifest.json" \
            --val-manifest "$C3_FEAT_DIR/val/manifest.json" \
            --eval2-manifest "$C3_FEAT_DIR/eval2/manifest.json" \
            --output-dir "$C3_POLICY_DIR" \
            --batch-size 8 \
            --lr 1e-4 \
            --warmup-steps 1000 \
            --min-steps 15000 \
            --max-steps 50000 \
            --val-every 500 \
            --patience 20 \
            --seed 0 \
            --overwrite
    fi
fi

# ==============================================================================
# PHASE 2: FLUX.2 KLEIN & COSMOS 14B VIDEO-LORA PIPELINES (SCALE-100)
# ==============================================================================
run_task "Run FLUX.2 klein & Cosmos 14B Scale-100 Pipeline" \
    bash scripts/video_vam/run_scale100_videolora_dual_eval_queue.sh

# ==============================================================================
# PHASE 3: COMPILE FINAL SCALE-100 BENCHMARK MATRIX
# ==============================================================================
run_task "Update Scale-100 Benchmark Leaderboard" \
    "$PYTHON" scripts/video_vam/update_scale100_leaderboard.py

echo ""
echo "================================================================================"
echo "V2 OVERNIGHT BENCHMARK PIPELINE COMPLETED SUCCESSFULLY AT $(date)"
echo "================================================================================"
