#!/usr/bin/env bash
# ==============================================================================
# Protocol 1.0 Re-Benchmarking Queue Runner (Phase 4 Architectural Refactoring)
#
# Re-benchmarks video foundation model representations under strict Protocol 1.0:
#   1. Disjoint episode splits: Train (0-31), Val (32-39; 88 fixed anchors).
#   2. True VAE encodings & BaseVAMExtractor contract.
#   3. Zero data leakage (strictly blocks frame-level random splits).
#   4. Exact action masking: action_is_pad excluded from normalizer and loss/RMSE.
#   5. Unified SmolExpert training with standardized hyperparameters.
#
# Sequence:
#   Stage 1: Cosmos 7B SmolExpert Training (64 tokens, 8192-dim)
#   Stage 2: FLUX.2 [klein] SmolExpert Training (256 tokens, 6144-dim)
#   Stage 3: Cosmos 14B Feature Cache Extraction (True VAE, FP8, 10240-dim)
#   Stage 4: Cosmos 14B SmolExpert Training (64 tokens, 10240-dim)
# ==============================================================================

set -Eeuo pipefail

REPO="/home/anton/lerobot-video-vam"
CACHE="/home/anton/.cache/video-vam"
LOG_DIR="$REPO/logs"
RUNS_DIR="$CACHE/runs"
MASTER_LOG="$RUNS_DIR/rebenchmark_queue_protocol1_0.log"

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

# Standardized Hyperparameters
MAX_STEPS="${MAX_STEPS:-50000}"
MIN_STEPS="${MIN_STEPS:-20000}"
VAL_EVERY="${VAL_EVERY:-500}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
SCHEDULER="${SCHEDULER:-cosine}"

timestamp() {
    date --iso-8601=seconds
}

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] STARTING PROTOCOL 1.0 RE-BENCHMARKING QUEUE\n" "$(timestamp)"
printf "[%s] Master Log: %s\n" "$(timestamp)" "$MASTER_LOG"
printf "[%s] Device: %s | Batch Size: %s | LR: %s | Max Steps: %s | Min Steps: %s | Patience: %s\n" \
    "$(timestamp)" "$DEVICE" "$BATCH_SIZE" "$LR" "$MAX_STEPS" "$MIN_STEPS" "$PATIENCE"
printf "[%s] ====================================================================\n" "$(timestamp)"

acquire_gpu_lock rebenchmark-protocol1-queue

# ==============================================================================
# STAGE 1: Cosmos 7B SmolExpert Training (Protocol 1.0)
# ==============================================================================
COSMOS7B_CACHE="$CACHE/cosmos7b-protocol1-cache"
STAGE1_RUN_DIR="$RUNS_DIR/cosmos7b-protocol1-smolexpert"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 1: Cosmos 7B Protocol 1.0 SmolExpert Training ===\n" "$(timestamp)"
printf "[%s] Cache: %s\n" "$(timestamp)" "$COSMOS7B_CACHE"
printf "[%s] Run Dir: %s\n" "$(timestamp)" "$STAGE1_RUN_DIR"

if [[ ! -f "$COSMOS7B_CACHE/train/manifest.json" ]]; then
    printf "[%s] Extracting Cosmos 7B Protocol 1.0 Feature Cache...\n" "$(timestamp)"
    uv run python scripts/video_vam/build_vam_feature_cache.py \
        --backbone cosmos7b \
        --output-dir "$COSMOS7B_CACHE" \
        --train-episodes 0-31 \
        --val-episodes 32-39 \
        --train-stride 3 \
        --val-stride 20 \
        --device "$DEVICE" \
        --dtype bfloat16
fi

if [[ ! -f "$STAGE1_RUN_DIR/best.safetensors" ]]; then
    printf "[%s] Launching SmolExpert Training on Cosmos 7B Features...\n" "$(timestamp)"
    uv run python scripts/video_vam/train_smolexpert.py \
        --train-manifest "$COSMOS7B_CACHE/train/manifest.json" \
        --val-manifest "$COSMOS7B_CACHE/val/manifest.json" \
        --output-dir "$STAGE1_RUN_DIR" \
        --backbone cosmos7b \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --lr-scheduler "$SCHEDULER" \
        --warmup-steps "$WARMUP_STEPS" \
        --min-steps "$MIN_STEPS" \
        --max-steps "$MAX_STEPS" \
        --val-every "$VAL_EVERY" \
        --patience "$PATIENCE" \
        --device "$DEVICE" \
        --no-wandb
    printf "[%s] STAGE 1 COMPLETED. Best metrics:\n" "$(timestamp)"
    cat "$STAGE1_RUN_DIR/best_metrics.json" || true
else
    printf "[%s] STAGE 1 already completed: %s/best.safetensors found.\n" "$(timestamp)" "$STAGE1_RUN_DIR"
fi

# ==============================================================================
# STAGE 2: FLUX.2 [klein] SmolExpert Training (Protocol 1.0)
# ==============================================================================
FLUX2_CACHE="$CACHE/flux2-klein-protocol1-cache"
STAGE2_RUN_DIR="$RUNS_DIR/flux2-klein-protocol1-smolexpert"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 2: FLUX.2 [klein] Protocol 1.0 SmolExpert Training ===\n" "$(timestamp)"
printf "[%s] Cache: %s\n" "$(timestamp)" "$FLUX2_CACHE"
printf "[%s] Run Dir: %s\n" "$(timestamp)" "$STAGE2_RUN_DIR"

if [[ ! -f "$FLUX2_CACHE/train/manifest.json" ]]; then
    printf "[%s] Extracting FLUX.2 klein Protocol 1.0 Feature Cache...\n" "$(timestamp)"
    uv run python scripts/video_vam/build_vam_feature_cache.py \
        --backbone flux2_klein \
        --output-dir "$FLUX2_CACHE" \
        --train-episodes 0-31 \
        --val-episodes 32-39 \
        --train-stride 3 \
        --val-stride 20 \
        --device "$DEVICE" \
        --dtype bfloat16
fi

if [[ ! -f "$STAGE2_RUN_DIR/best.safetensors" ]]; then
    printf "[%s] Launching SmolExpert Training on FLUX.2 klein Features...\n" "$(timestamp)"
    uv run python scripts/video_vam/train_smolexpert.py \
        --train-manifest "$FLUX2_CACHE/train/manifest.json" \
        --val-manifest "$FLUX2_CACHE/val/manifest.json" \
        --output-dir "$STAGE2_RUN_DIR" \
        --backbone flux2_klein \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --lr-scheduler "$SCHEDULER" \
        --warmup-steps "$WARMUP_STEPS" \
        --min-steps "$MIN_STEPS" \
        --max-steps "$MAX_STEPS" \
        --val-every "$VAL_EVERY" \
        --patience "$PATIENCE" \
        --device "$DEVICE" \
        --no-wandb
    printf "[%s] STAGE 2 COMPLETED. Best metrics:\n" "$(timestamp)"
    cat "$STAGE2_RUN_DIR/best_metrics.json" || true
else
    printf "[%s] STAGE 2 already completed: %s/best.safetensors found.\n" "$(timestamp)" "$STAGE2_RUN_DIR"
fi

# ==============================================================================
# STAGE 3 & 4: Cosmos 14B Feature Extraction & SmolExpert Training (Protocol 1.0)
# ==============================================================================
COSMOS14B_CACHE="$CACHE/cosmos14b-protocol1-cache"
STAGE4_RUN_DIR="$RUNS_DIR/cosmos14b-protocol1-smolexpert"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 3: Cosmos 14B Protocol 1.0 Feature Extraction ===\n" "$(timestamp)"
printf "[%s] Cache: %s\n" "$(timestamp)" "$COSMOS14B_CACHE"

if [[ ! -f "$COSMOS14B_CACHE/val/manifest.json" || ! -f "$COSMOS14B_CACHE/train/manifest.json" ]]; then
    printf "[%s] Extracting Cosmos 14B Protocol 1.0 Feature Cache (Batched FP8)...\n" "$(timestamp)"
    uv run python scripts/video_vam/build_vam_feature_cache.py \
        --backbone cosmos14b \
        --output-dir "$COSMOS14B_CACHE" \
        --train-episodes 0-31 \
        --val-episodes 32-39 \
        --train-stride 3 \
        --val-stride 20 \
        --batch-size 16 \
        --device "$DEVICE" \
        --dtype bfloat16 \
        --resume
fi

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 4: Cosmos 14B Protocol 1.0 SmolExpert Training ===\n" "$(timestamp)"
printf "[%s] Run Dir: %s\n" "$(timestamp)" "$STAGE4_RUN_DIR"

if [[ ! -f "$STAGE4_RUN_DIR/best.safetensors" ]]; then
    printf "[%s] Launching SmolExpert Training on Cosmos 14B Features...\n" "$(timestamp)"
    uv run python scripts/video_vam/train_smolexpert.py \
        --train-manifest "$COSMOS14B_CACHE/train/manifest.json" \
        --val-manifest "$COSMOS14B_CACHE/val/manifest.json" \
        --output-dir "$STAGE4_RUN_DIR" \
        --backbone cosmos14b \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --lr-scheduler "$SCHEDULER" \
        --warmup-steps "$WARMUP_STEPS" \
        --min-steps "$MIN_STEPS" \
        --max-steps "$MAX_STEPS" \
        --val-every "$VAL_EVERY" \
        --patience "$PATIENCE" \
        --device "$DEVICE" \
        --no-wandb
    printf "[%s] STAGE 4 COMPLETED. Best metrics:\n" "$(timestamp)"
    cat "$STAGE4_RUN_DIR/best_metrics.json" || true
else
    printf "[%s] STAGE 4 already completed: %s/best.safetensors found.\n" "$(timestamp)" "$STAGE4_RUN_DIR"
fi

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] ALL RE-BENCHMARKING STAGES FINISHED SUCCESSFULLY!\n" "$(timestamp)"
printf "[%s] Master log: %s\n" "$(timestamp)" "$MASTER_LOG"
printf "[%s] ====================================================================\n" "$(timestamp)"
