#!/usr/bin/env bash
# ==============================================================================
# Expanded 100-Episode Dual-Evaluation Queue Runner (Phase 5 Scaling)
#
# Scales robot demonstration data from 40 to 100 episodes using Orellius/cube_out_of_box_v2:
#   - Train: Episodes 0-31 + 50 new episodes (40-89) = 82 episodes (stride 3, 3,065 samples)
#   - Eval-Set 1 (Historical Benchmark): Episodes 32-39 (strict Protocol 1.0, stride 20, 88 anchors)
#   - Eval-Set 2 (New Benchmark): Episodes 90-99 (stride 20, 51 anchors)
#
# No video-LoRA training! Strictly Base feature extraction + SmolExpert training on Base features.
#
# Sequence:
#   Stage A: Cosmos 14B Base (Layers 18 & 30, FP8 Block Streaming)
#     - Extract Base features on 100 episodes
#     - Train SmolExpert to true convergence (min_steps=20000, max_steps=60000, patience=25)
#     - Dual-evaluate on Eval-Set 1 and Eval-Set 2
#   Stage B: FLUX.2 [klein] Base (Multi-Reference Junction Tap)
#     - Extract Base features on 100 episodes
#     - Train SmolExpert to true convergence (min_steps=15000, max_steps=50000, patience=25)
#     - Dual-evaluate on Eval-Set 1 and Eval-Set 2
# ==============================================================================

set -Eeuo pipefail

REPO="/home/anton/lerobot-video-vam"
CACHE="/home/anton/.cache/video-vam"
LOG_DIR="$REPO/logs"
RUNS_DIR="$CACHE/runs"
MASTER_LOG="$RUNS_DIR/scale100_expanded_pipeline.log"

mkdir -p "$LOG_DIR" "$RUNS_DIR"
exec > >(tee -a "$MASTER_LOG") 2>&1

cd "$REPO"
# shellcheck source=/dev/null
source scripts/video_vam/cosmos_cuda_env.sh
# shellcheck source=/dev/null
source scripts/video_vam/gpu_lock.sh

export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH=".:src:${PYTHONPATH:-}"
DEVICE="${DEVICE:-cuda:0}"

timestamp() {
    date --iso-8601=seconds
}

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] STARTING EXPANDED 100-EPISODE DUAL-EVALUATION PIPELINE\n" "$(timestamp)"
printf "[%s] Master Log: %s\n" "$(timestamp)" "$MASTER_LOG"
printf "[%s] Device: %s\n" "$(timestamp)" "$DEVICE"
printf "[%s] Dataset: Orellius/cube_out_of_box_v2 (100 episodes)\n" "$(timestamp)"
printf "[%s] Train Split: Episodes 0-31 + 40-89 (82 eps, stride 3, 3065 samples)\n" "$(timestamp)"
printf "[%s] Eval-Set 1 (Historical): Episodes 32-39 (8 eps, stride 20, 88 anchors)\n" "$(timestamp)"
printf "[%s] Eval-Set 2 (New): Episodes 90-99 (10 eps, stride 20, 51 anchors)\n" "$(timestamp)"
printf "[%s] ====================================================================\n" "$(timestamp)"

acquire_gpu_lock scale100-eval-queue

# ==============================================================================
# STAGE A: Cosmos 14B Base (Layers 18 & 30, FP8 Block Streaming)
# ==============================================================================
COSMOS14B_CACHE="$CACHE/cosmos14b-scale100-cache"
STAGE_A_RUN_DIR="$RUNS_DIR/cosmos14b-scale100-smolexpert"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE A: Cosmos 14B Base (100 Episodes) ===\n" "$(timestamp)"
printf "[%s] Cache Directory: %s\n" "$(timestamp)" "$COSMOS14B_CACHE"
printf "[%s] Run Directory:   %s\n" "$(timestamp)" "$STAGE_A_RUN_DIR"
printf "[%s] Hyperparameters: min_steps=20000, max_steps=60000, patience=25\n" "$(timestamp)"

mkdir -p "$COSMOS14B_CACHE/train" "$COSMOS14B_CACHE/val"

# Seed with existing hardlinks from protocol 1.0 cache if available to save redundant extraction
if [[ -d "$CACHE/cosmos14b-protocol1-cache/train" ]]; then
    printf "[%s] Pre-seeding Cosmos 14B train cache with historical episodes 0-31...\n" "$(timestamp)"
    cp -al "$CACHE/cosmos14b-protocol1-cache/train/"*.safetensors "$COSMOS14B_CACHE/train/" 2>/dev/null || true
fi
if [[ -d "$CACHE/cosmos14b-protocol1-cache/val" ]]; then
    printf "[%s] Pre-seeding Cosmos 14B val cache with historical episodes 32-39...\n" "$(timestamp)"
    cp -al "$CACHE/cosmos14b-protocol1-cache/val/"*.safetensors "$COSMOS14B_CACHE/val/" 2>/dev/null || true
fi

# Extract Base Features on 100 episodes
if [[ ! -f "$COSMOS14B_CACHE/train/manifest.json" || ! -f "$COSMOS14B_CACHE/eval2/manifest.json" ]]; then
    printf "[%s] Extracting Cosmos 14B Base features (FP8 block streaming)...\n" "$(timestamp)"
    uv run python scripts/video_vam/build_vam_feature_cache.py \
        --backbone cosmos14b \
        --output-dir "$COSMOS14B_CACHE" \
        --dataset-repo-id Orellius/cube_out_of_box_v2 \
        --train-episodes "0-31, 40-89" \
        --val-episodes "32-39" \
        --eval2-episodes "90-99" \
        --train-stride 3 \
        --val-stride 20 \
        --eval2-stride 20 \
        --batch-size 16 \
        --device "$DEVICE" \
        --dtype bfloat16 \
        --resume
fi

# Train SmolExpert to true convergence
if [[ ! -f "$STAGE_A_RUN_DIR/final_dual_eval_metrics.json" ]]; then
    printf "[%s] Launching SmolExpert Training on Cosmos 14B Base Features...\n" "$(timestamp)"
    uv run python scripts/video_vam/train_smolexpert.py \
        --train-manifest "$COSMOS14B_CACHE/train/manifest.json" \
        --val-manifest "$COSMOS14B_CACHE/val/manifest.json" \
        --eval2-manifest "$COSMOS14B_CACHE/eval2/manifest.json" \
        --output-dir "$STAGE_A_RUN_DIR" \
        --backbone cosmos14b \
        --batch-size 8 \
        --lr 1e-4 \
        --lr-scheduler cosine \
        --warmup-steps 1000 \
        --min-steps 20000 \
        --max-steps 60000 \
        --val-every 500 \
        --patience 25 \
        --device "$DEVICE" \
        --no-wandb
    printf "[%s] STAGE A COMPLETED.\n" "$(timestamp)"
    uv run python scripts/video_vam/update_scale100_leaderboard.py
else
    printf "[%s] STAGE A already completed: %s/final_dual_eval_metrics.json found.\n" "$(timestamp)" "$STAGE_A_RUN_DIR"
fi

# ==============================================================================
# STAGE B: FLUX.2 [klein] Base (Multi-Reference Junction Tap)
# ==============================================================================
FLUX2_CACHE="$CACHE/flux2-klein-scale100-cache"
STAGE_B_RUN_DIR="$RUNS_DIR/flux2-klein-scale100-smolexpert"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE B: FLUX.2 [klein] Base (100 Episodes) ===\n" "$(timestamp)"
printf "[%s] Cache Directory: %s\n" "$(timestamp)" "$FLUX2_CACHE"
printf "[%s] Run Directory:   %s\n" "$(timestamp)" "$STAGE_B_RUN_DIR"
printf "[%s] Hyperparameters: min_steps=15000, max_steps=50000, patience=25\n" "$(timestamp)"

mkdir -p "$FLUX2_CACHE/train" "$FLUX2_CACHE/val"

# Seed with existing hardlinks from protocol 1.0 cache if available to save redundant extraction
if [[ -d "$CACHE/flux2-klein-protocol1-cache/train" ]]; then
    printf "[%s] Pre-seeding FLUX.2 klein train cache with historical episodes 0-31...\n" "$(timestamp)"
    cp -al "$CACHE/flux2-klein-protocol1-cache/train/"*.safetensors "$FLUX2_CACHE/train/" 2>/dev/null || true
fi
if [[ -d "$CACHE/flux2-klein-protocol1-cache/val" ]]; then
    printf "[%s] Pre-seeding FLUX.2 klein val cache with historical episodes 32-39...\n" "$(timestamp)"
    cp -al "$CACHE/flux2-klein-protocol1-cache/val/"*.safetensors "$FLUX2_CACHE/val/" 2>/dev/null || true
fi

# Extract Base Features on 100 episodes
if [[ ! -f "$FLUX2_CACHE/train/manifest.json" || ! -f "$FLUX2_CACHE/eval2/manifest.json" ]]; then
    printf "[%s] Extracting FLUX.2 klein Base features (Junction Tap)...\n" "$(timestamp)"
    uv run python scripts/video_vam/build_vam_feature_cache.py \
        --backbone flux2_klein \
        --output-dir "$FLUX2_CACHE" \
        --dataset-repo-id Orellius/cube_out_of_box_v2 \
        --train-episodes "0-31, 40-89" \
        --val-episodes "32-39" \
        --eval2-episodes "90-99" \
        --train-stride 3 \
        --val-stride 20 \
        --eval2-stride 20 \
        --batch-size 8 \
        --device "$DEVICE" \
        --dtype bfloat16 \
        --resume
fi

# Train SmolExpert to true convergence
if [[ ! -f "$STAGE_B_RUN_DIR/final_dual_eval_metrics.json" ]]; then
    printf "[%s] Launching SmolExpert Training on FLUX.2 klein Base Features...\n" "$(timestamp)"
    uv run python scripts/video_vam/train_smolexpert.py \
        --train-manifest "$FLUX2_CACHE/train/manifest.json" \
        --val-manifest "$FLUX2_CACHE/val/manifest.json" \
        --eval2-manifest "$FLUX2_CACHE/eval2/manifest.json" \
        --output-dir "$STAGE_B_RUN_DIR" \
        --backbone flux2_klein \
        --batch-size 8 \
        --lr 1e-4 \
        --lr-scheduler cosine \
        --warmup-steps 1000 \
        --min-steps 15000 \
        --max-steps 50000 \
        --val-every 500 \
        --patience 25 \
        --device "$DEVICE" \
        --no-wandb
    printf "[%s] STAGE B COMPLETED.\n" "$(timestamp)"
    uv run python scripts/video_vam/update_scale100_leaderboard.py
else
    printf "[%s] STAGE B already completed: %s/final_dual_eval_metrics.json found.\n" "$(timestamp)" "$STAGE_B_RUN_DIR"
fi

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] ALL 100-EPISODE EXPANDED BENCHMARKS FINISHED SUCCESSFULLY!\n" "$(timestamp)"
printf "[%s] Master Log: %s\n" "$(timestamp)" "$MASTER_LOG"
printf "[%s] ====================================================================\n" "$(timestamp)"

uv run python scripts/video_vam/update_scale100_leaderboard.py
