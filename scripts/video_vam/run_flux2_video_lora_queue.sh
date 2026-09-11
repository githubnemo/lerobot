#!/usr/bin/env bash
# Automated FLUX.2 [klein] Video-LoRA Fine-Tuning & Downstream Policy Queue Runner
#
# STAGE 1: Fine-tune FLUX.2 [klein] Video-LoRA for 5000 steps on robot video dataset (cosine decay / warmup)
# STAGE 2: Multi-reference feature extraction with the new LoRA weights
# STAGE 3: SmolExpert downstream action policy training on adapted FLUX.2 features until convergence
# STAGE 4: Evaluation and metrics summary

set -Eeuo pipefail

REPO="/home/anton/lerobot-video-vam"
CACHE="/home/anton/.cache/video-vam"
LOG_DIR="$REPO/logs"
RUNS_DIR="$CACHE/runs"
LOG_FILE="${LOG_FILE:-$LOG_DIR/flux2_video_lora_queue_$(date +%Y%m%d).log}"
CACHE_LOG="$RUNS_DIR/flux2-video-lora-queue.log"

mkdir -p "$LOG_DIR" "$RUNS_DIR"
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
DRY_RUN_FLAG="${DRY_RUN_FLAG:-}"
LORA_MAX_STEPS="${LORA_MAX_STEPS:-5000}"
LORA_WARMUP_STEPS="${LORA_WARMUP_STEPS:-250}"
SMOLEXPERT_MAX_STEPS="${SMOLEXPERT_MAX_STEPS:-50000}"
SMOLEXPERT_EVAL_INTERVAL="${SMOLEXPERT_EVAL_INTERVAL:-250}"
SMOLEXPERT_PATIENCE="${SMOLEXPERT_PATIENCE:-10}"
STRIDE="${STRIDE:-3}"
DATASET_ROOT="/home/anton/.cache/video-vam/cube-out-of-box-dataset"
ARM_NAME="${GPU_LOCK_ARM:-flux2-video-lora-queue}"

timestamp() {
    date --iso-8601=seconds
}

printf "[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] STARTING FLUX.2 [klein] VIDEO-LORA QUEUE RUNNER\n" "$(timestamp)"
printf "[%s] Log: %s\n" "$(timestamp)" "$LOG_FILE"
printf "[%s] Device: %s, Stride: %s\n" "$(timestamp)" "$DEVICE" "$STRIDE"
printf "[%s] Video-LoRA Settings: max_steps=%s, warmup_steps=%s\n" "$(timestamp)" "$LORA_MAX_STEPS" "$LORA_WARMUP_STEPS"
printf "[%s] SmolExpert Settings: max_steps=%s, eval_interval=%s, patience=%s\n" "$(timestamp)" "$SMOLEXPERT_MAX_STEPS" "$SMOLEXPERT_EVAL_INTERVAL" "$SMOLEXPERT_PATIENCE"
printf "[%s] Dataset: %s\n" "$(timestamp)" "$DATASET_ROOT"
printf "[%s] ====================================================================\n" "$(timestamp)"

# Acquire GPU lock
printf "[%s] Requesting GPU lock under arm=%s...\n" "$(timestamp)" "$ARM_NAME"
acquire_gpu_lock "$ARM_NAME"

# =============================================================================
# STAGE 1: Fine-tune FLUX.2 [klein] Video-LoRA on robot demonstration dataset
# =============================================================================
STAGE1_LORA_DIR="$RUNS_DIR/flux2-klein-videolora"
FLUX2_LORA_FILE="$STAGE1_LORA_DIR/flux2_klein_lora.safetensors"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 1: FLUX.2 [klein] Video-LoRA Fine-Tuning (%s steps) ===\n" "$(timestamp)" "$LORA_MAX_STEPS"
printf "[%s] Output Dir: %s\n" "$(timestamp)" "$STAGE1_LORA_DIR"
printf "[%s] ====================================================================\n" "$(timestamp)"

uv run python -m scripts.video_vam.train_flux2_klein_lora \
    --output-dir "$STAGE1_LORA_DIR" \
    --dataset-root "$DATASET_ROOT" \
    --lora-rank 16 \
    --lora-alpha 16.0 \
    --device "$DEVICE" \
    --dtype "bfloat16" \
    --max-steps "$LORA_MAX_STEPS" \
    --warmup-steps "$LORA_WARMUP_STEPS" \
    $DRY_RUN_FLAG

printf "[%s] Stage 1 complete! FLUX.2 Video-LoRA saved to %s\n" "$(timestamp)" "$FLUX2_LORA_FILE"

# =============================================================================
# STAGE 2: Feature extraction using newly fine-tuned Video-LoRA weights
# =============================================================================
FLUX2_ADAPTED_CACHE="$CACHE/flux2-klein-adapted-features"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 2: Feature Extraction with FLUX.2 Video-LoRA Weights ===\n" "$(timestamp)"
printf "[%s] LoRA weights: %s\n" "$(timestamp)" "$FLUX2_LORA_FILE"
printf "[%s] Output cache: %s\n" "$(timestamp)" "$FLUX2_ADAPTED_CACHE"
printf "[%s] ====================================================================\n" "$(timestamp)"

uv run python -m scripts.video_vam.extract_flux2_klein_features \
    --lora-path "$FLUX2_LORA_FILE" \
    --output-dir "$FLUX2_ADAPTED_CACHE" \
    --dataset-root "$DATASET_ROOT" \
    --stride "$STRIDE" \
    --tap-location "junction" \
    --history-frames 3 \
    --use-goal-frame \
    --device "$DEVICE" \
    --dtype "bfloat16" \
    $DRY_RUN_FLAG

printf "[%s] Stage 2 complete! Adapted features extracted to %s\n" "$(timestamp)" "$FLUX2_ADAPTED_CACHE"

# =============================================================================
# STAGE 3: Train SmolExpert on newly adapted FLUX.2 [klein] features
# =============================================================================
STAGE3_POLICY_DIR="$RUNS_DIR/flux2-klein-adapted-smolexpert"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 3: Train SmolExpert on Adapted FLUX.2 Features ===\n" "$(timestamp)"
printf "[%s] Cache: %s\n" "$(timestamp)" "$FLUX2_ADAPTED_CACHE"
printf "[%s] Output directory: %s\n" "$(timestamp)" "$STAGE3_POLICY_DIR"
printf "[%s] Max steps: %s, Eval interval: %s, Patience: %s\n" "$(timestamp)" "$SMOLEXPERT_MAX_STEPS" "$SMOLEXPERT_EVAL_INTERVAL" "$SMOLEXPERT_PATIENCE"
printf "[%s] ====================================================================\n" "$(timestamp)"

uv run python -m scripts.video_vam.train_smolexpert_on_flux2_klein \
    --cache-dir "$FLUX2_ADAPTED_CACHE" \
    --output-dir "$STAGE3_POLICY_DIR" \
    --device "$DEVICE" \
    --max-steps "$SMOLEXPERT_MAX_STEPS" \
    --eval-interval "$SMOLEXPERT_EVAL_INTERVAL" \
    --patience "$SMOLEXPERT_PATIENCE" \
    --batch-size 16 \
    $DRY_RUN_FLAG

# =============================================================================
# STAGE 4: Metrics Summary
# =============================================================================
printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] FLUX.2 [klein] ADAPTED VIDEO-LORA RUN SUMMARY\n" "$(timestamp)"
printf "[%s] ====================================================================\n" "$(timestamp)"

if [[ -f "$STAGE3_POLICY_DIR/eval_metrics.json" ]]; then
    printf "\n[%s] FLUX.2 [klein] Adapted Video-LoRA Evaluation Metrics:\n" "$(timestamp)"
    cat "$STAGE3_POLICY_DIR/eval_metrics.json"
    printf "\n"
fi

printf "\n[%s] FLUX.2 [klein] VIDEO-LORA PIPELINE COMPLETED SUCCESSFULLY!\n" "$(timestamp)"
