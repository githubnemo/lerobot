#!/usr/bin/env bash
# Extended FLUX.2 [klein] SmolExpert Training (>= 30 Minutes Guaranteed)
#
# Configured with:
# - min_steps=22000 (ensures >= 30 minutes of real GPU training at ~11.8 steps/sec)
# - max_steps=50000 (allows up to ~70 minutes if validation loss keeps improving)
# - lr=3e-5 (reduced from 1e-4 for stable, non-divergent fine convergence)
# - patience=80 (20,000 steps of plateau tolerance, evaluated every 250 steps)

set -Eeuo pipefail

REPO="/home/anton/lerobot-video-vam"
CACHE="/home/anton/.cache/video-vam"
LOG_DIR="$REPO/logs"
RUNS_DIR="$CACHE/runs"
LOG_FILE="${LOG_FILE:-$LOG_DIR/flux2_smolexpert_extended_$(date +%Y%m%d).log}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUNS_DIR/flux2-klein-extended-smolexpert}"
CACHE_DIR="$CACHE/flux2-klein-raw-features"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
exec >> "$LOG_FILE" 2>&1

cd "$REPO"
# shellcheck source=/dev/null
source scripts/video_vam/cosmos_cuda_env.sh
# shellcheck source=/dev/null
source scripts/video_vam/gpu_lock.sh

export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH=".:src:${PYTHONPATH:-}"
DEVICE="${DEVICE:-cuda:0}"
ARM_NAME="${GPU_LOCK_ARM:-flux2-smolexpert-extended}"

MIN_STEPS="${MIN_STEPS:-22000}"
MAX_STEPS="${MAX_STEPS:-50000}"
LR="${LR:-3e-5}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
PATIENCE="${PATIENCE:-80}"
BATCH_SIZE="${BATCH_SIZE:-16}"

timestamp() {
    date --iso-8601=seconds
}

printf "[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] STARTING EXTENDED FLUX.2 [klein] SMOLEXPERT TRAINING RUN\n" "$(timestamp)"
printf "[%s] Log: %s\n" "$(timestamp)" "$LOG_FILE"
printf "[%s] Output: %s\n" "$(timestamp)" "$OUTPUT_DIR"
printf "[%s] Cache: %s\n" "$(timestamp)" "$CACHE_DIR"
printf "[%s] Hyperparameters: min_steps=%s, max_steps=%s, lr=%s, eval_interval=%s, patience=%s, batch_size=%s\n" \
    "$(timestamp)" "$MIN_STEPS" "$MAX_STEPS" "$LR" "$EVAL_INTERVAL" "$PATIENCE" "$BATCH_SIZE"
printf "[%s] Estimated Duration: at least 30-32 minutes (up to ~70 minutes if converging)\n" "$(timestamp)"
printf "[%s] ====================================================================\n" "$(timestamp)"

printf "[%s] Requesting GPU lock under arm=%s...\n" "$(timestamp)" "$ARM_NAME"
acquire_gpu_lock "$ARM_NAME"

printf "\n[%s] === Launching SmolExpert on FLUX.2 [klein] Raw Features ===\n" "$(timestamp)"

uv run python -m scripts.video_vam.train_smolexpert_on_flux2_klein \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --lr "$LR" \
    --min-steps "$MIN_STEPS" \
    --max-steps "$MAX_STEPS" \
    --eval-interval "$EVAL_INTERVAL" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE"

if [[ -f "$OUTPUT_DIR/eval_metrics.json" ]]; then
    printf "\n[%s] Extended Training Evaluation Metrics:\n" "$(timestamp)"
    cat "$OUTPUT_DIR/eval_metrics.json"
    printf "\n"
fi

printf "\n[%s] EXTENDED FLUX.2 [klein] SMOLEXPERT TRAINING COMPLETE!\n" "$(timestamp)"
