#!/usr/bin/env bash
# Automated Cosmos-1.0-Diffusion-14B Feature Extraction & SmolExpert Training Pipeline
#
# STAGE 1: Feature extraction (tapped layers 18 & 30) with FP8 block streaming on cuda:0 (batch_size=32)
# STAGE 2: SmolExpert downstream action policy training until convergence
# STAGE 3: Final evaluation and metrics recording

set -Eeuo pipefail

REPO="/home/anton/lerobot-video-vam"
CACHE="/home/anton/.cache/video-vam"
LOG_DIR="$REPO/logs"
RUNS_DIR="$CACHE/runs"
LOG_FILE="${LOG_FILE:-$LOG_DIR/cosmos14b_pipeline_$(date +%Y%m%d).log}"

mkdir -p "$LOG_DIR" "$RUNS_DIR"
exec >> "$LOG_FILE" 2>&1

cd "$REPO"
source scripts/video_vam/cosmos_cuda_env.sh
source scripts/video_vam/gpu_lock.sh

export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH=".:src:${PYTHONPATH:-}"
DEVICE="${DEVICE:-cuda:0}"
ARM_NAME="${GPU_LOCK_ARM:-cosmos14b-pipeline}"

timestamp() {
    date --iso-8601=seconds
}

printf "[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] STARTING COSMOS 14B FEATURE EXTRACTION & DOWNSTREAM PIPELINE\n" "$(timestamp)"
printf "[%s] Log: %s\n" "$(timestamp)" "$LOG_FILE"
printf "[%s] Device: %s\n" "$(timestamp)" "$DEVICE"
printf "[%s] ====================================================================\n" "$(timestamp)"

# Acquire GPU lock
printf "[%s] Requesting GPU lock under arm=%s...\n" "$(timestamp)" "$ARM_NAME"
acquire_gpu_lock "$ARM_NAME"

# =============================================================================
# STAGE 1: Extract features from Cosmos 14B with FP8 streaming
# =============================================================================
CACHE_DIR="$CACHE/cosmos14b-features"
CHECKPOINT_DIR="$CACHE/cosmos-14b"
DATASET_ROOT="$CACHE/cube-out-of-box-dataset"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 1: Cosmos 14B Feature Extraction (Layers 18 & 30) ===\n" "$(timestamp)"
printf "[%s] Checkpoint: %s\n" "$(timestamp)" "$CHECKPOINT_DIR"
printf "[%s] Output cache: %s\n" "$(timestamp)" "$CACHE_DIR"
printf "[%s] ====================================================================\n" "$(timestamp)"

uv run python -m scripts.video_vam.extract_cosmos14b_features \
    --checkpoint-path "$CHECKPOINT_DIR" \
    --output-dir "$CACHE_DIR" \
    --dataset-root "$DATASET_ROOT" \
    --hidden-layers "18,30" \
    --pool-spatial 2 \
    --concat-layers \
    --block-streaming \
    --fp8-linear \
    --stride 3 \
    --batch-size 32 \
    --device "$DEVICE" \
    --dtype "bfloat16"

printf "[%s] Stage 1 complete! Cosmos 14B features saved to %s\n" "$(timestamp)" "$CACHE_DIR"

# =============================================================================
# STAGE 2: Train SmolExpert on Cosmos 14B features until convergence
# =============================================================================
SMOLEXPERT_DIR="$RUNS_DIR/cosmos14b-smolexpert"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 2: Train SmolExpert on Cosmos 14B Features ===\n" "$(timestamp)"
printf "[%s] Cache: %s\n" "$(timestamp)" "$CACHE_DIR"
printf "[%s] Output dir: %s\n" "$(timestamp)" "$SMOLEXPERT_DIR"
printf "[%s] ====================================================================\n" "$(timestamp)"

uv run python -m scripts.video_vam.train_smolexpert_on_cosmos14b \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$SMOLEXPERT_DIR" \
    --batch-size 16 \
    --lr 1e-4 \
    --max-steps 25000 \
    --eval-interval 250 \
    --patience 8 \
    --device "$DEVICE"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] COSMOS 14B PIPELINE COMPLETED SUCCESSFULLY!\n" "$(timestamp)"
printf "[%s] Evaluation Metrics:\n" "$(timestamp)"
if [[ -f "$SMOLEXPERT_DIR/eval_metrics.json" ]]; then
    cat "$SMOLEXPERT_DIR/eval_metrics.json"
fi
printf "\n[%s] ====================================================================\n" "$(timestamp)"
