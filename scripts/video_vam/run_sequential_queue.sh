#!/usr/bin/env bash
# Automated Sequential Queue Runner - REAL Full GPU Convergence Pipeline
#
# STAGE 1: SmolExpert (MoE) on Cosmos 7B Base features until real convergence (max_steps=50000, patience=10)
# STAGE 2: SmolExpert (MoE) on FLUX.2 [klein] Base features until real convergence (max_steps=50000, patience=10)
# STAGE 3: Fine-tune Cosmos 7B Video-LoRA for 5000 steps on full dataset (cosine decay / warmup)
# STAGE 4: Feature extraction with the new LoRA weights
# STAGE 5: SmolExpert on newly adapted Cosmos 7B features until real convergence (max_steps=50000, patience=10)

set -Eeuo pipefail

REPO="/home/anton/lerobot-video-vam"
CACHE="/home/anton/.cache/video-vam"
LOG_DIR="$REPO/logs"
RUNS_DIR="$CACHE/runs"
LOG_FILE="${LOG_FILE:-$LOG_DIR/convergence_queue_20260905.log}"
CACHE_LOG="$RUNS_DIR/convergence-queue-20260905.log"

mkdir -p "$LOG_DIR" "$RUNS_DIR"
exec > >(tee -a "$LOG_FILE" "$CACHE_LOG") 2>&1

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
SMOLEXPERT_MAX_STEPS="${SMOLEXPERT_MAX_STEPS:-50000}"
SMOLEXPERT_EVAL_INTERVAL="${SMOLEXPERT_EVAL_INTERVAL:-250}"
SMOLEXPERT_PATIENCE="${SMOLEXPERT_PATIENCE:-10}"
LORA_MAX_STEPS="${LORA_MAX_STEPS:-5000}"
LORA_WARMUP_STEPS="${LORA_WARMUP_STEPS:-250}"
STRIDE="${STRIDE:-3}"
DATASET_ROOT="/home/anton/.cache/video-vam/cube-out-of-box-dataset"
ARM_NAME="${GPU_LOCK_ARM:-convergence-queue-20260905}"

timestamp() {
    date --iso-8601=seconds
}

printf "[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] STARTING 5-STAGE REAL GPU CONVERGENCE QUEUE RUNNER\n" "$(timestamp)"
printf "[%s] Log: %s\n" "$(timestamp)" "$LOG_FILE"
printf "[%s] Device: %s, Stride: %s\n" "$(timestamp)" "$DEVICE" "$STRIDE"
printf "[%s] SmolExpert Settings: max_steps=%s, eval_interval=%s, patience=%s\n" "$(timestamp)" "$SMOLEXPERT_MAX_STEPS" "$SMOLEXPERT_EVAL_INTERVAL" "$SMOLEXPERT_PATIENCE"
printf "[%s] Cosmos 7B Video-LoRA Settings: max_steps=%s, warmup_steps=%s\n" "$(timestamp)" "$LORA_MAX_STEPS" "$LORA_WARMUP_STEPS"
printf "[%s] Dataset: %s\n" "$(timestamp)" "$DATASET_ROOT"
printf "[%s] ====================================================================\n" "$(timestamp)"

# Acquire GPU lock for the entire sequential pipeline
printf "[%s] Requesting GPU lock under arm=%s...\n" "$(timestamp)" "$ARM_NAME"
acquire_gpu_lock "$ARM_NAME"

COSMOS7B_RAW_CACHE="$CACHE/cosmos7b-raw-features"
FLUX2_RAW_CACHE="$CACHE/flux2-klein-raw-features"

# Fallback feature extraction if not already cached
if [[ ! -f "$COSMOS7B_RAW_CACHE/manifest.json" ]]; then
    printf "\n[%s] Cosmos 7B Base feature cache not found at %s. Running extraction...\n" "$(timestamp)" "$COSMOS7B_RAW_CACHE"
    uv run python -m scripts.video_vam.extract_cosmos7b_features \
        --output-dir "$COSMOS7B_RAW_CACHE" \
        --dataset-root "$DATASET_ROOT" \
        --stride "$STRIDE" \
        --hidden-layers "14,20" \
        --pool-spatial 2 \
        --concat-layers \
        --device "$DEVICE" \
        --dtype "bfloat16" \
        $DRY_RUN_FLAG
fi

if [[ ! -f "$FLUX2_RAW_CACHE/manifest.json" ]]; then
    printf "\n[%s] FLUX.2 [klein] Base feature cache not found at %s. Running extraction...\n" "$(timestamp)" "$FLUX2_RAW_CACHE"
    uv run python -m scripts.video_vam.extract_flux2_klein_features \
        --output-dir "$FLUX2_RAW_CACHE" \
        --dataset-root "$DATASET_ROOT" \
        --stride "$STRIDE" \
        --tap-location "junction" \
        --history-frames 3 \
        --use-goal-frame \
        --device "$DEVICE" \
        --dtype "bfloat16" \
        $DRY_RUN_FLAG
fi

# =============================================================================
# STAGE 1: SmolExpert on Cosmos 7B Base features (already extracted)
# Train until real convergence (max_steps=50000, patience=10)
# Record downstream RMSE, H1, First-5
# =============================================================================
STAGE1_DIR="$RUNS_DIR/cosmos7b-raw-smolexpert"
printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 1: Train SmolExpert on Cosmos 7B Base Features ===\n" "$(timestamp)"
printf "[%s] Cache: %s\n" "$(timestamp)" "$COSMOS7B_RAW_CACHE"
printf "[%s] Output directory: %s\n" "$(timestamp)" "$STAGE1_DIR"
printf "[%s] Max steps: %s, Eval interval: %s, Patience: %s\n" "$(timestamp)" "$SMOLEXPERT_MAX_STEPS" "$SMOLEXPERT_EVAL_INTERVAL" "$SMOLEXPERT_PATIENCE"
printf "[%s] ====================================================================\n" "$(timestamp)"

uv run python -m scripts.video_vam.train_smolexpert_on_cosmos7b \
    --cache-dir "$COSMOS7B_RAW_CACHE" \
    --output-dir "$STAGE1_DIR" \
    --device "$DEVICE" \
    --max-steps "$SMOLEXPERT_MAX_STEPS" \
    --eval-interval "$SMOLEXPERT_EVAL_INTERVAL" \
    --patience "$SMOLEXPERT_PATIENCE" \
    --batch-size 16 \
    $DRY_RUN_FLAG

if [[ -f "$STAGE1_DIR/eval_metrics.json" ]]; then
    printf "[%s] Stage 1 Evaluation Metrics (Cosmos 7B Base):\n" "$(timestamp)"
    cat "$STAGE1_DIR/eval_metrics.json"
    printf "\n"
fi
printf "[%s] Stage 1 completed successfully!\n" "$(timestamp)"

# =============================================================================
# STAGE 2: SmolExpert on FLUX.2 [klein] features (already extracted)
# Train until real convergence (max_steps=50000, patience=10)
# Record metrics
# =============================================================================
STAGE2_DIR="$RUNS_DIR/flux2-klein-raw-smolexpert"
printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 2: Train SmolExpert on FLUX.2 [klein] Base Features ===\n" "$(timestamp)"
printf "[%s] Cache: %s\n" "$(timestamp)" "$FLUX2_RAW_CACHE"
printf "[%s] Output directory: %s\n" "$(timestamp)" "$STAGE2_DIR"
printf "[%s] Max steps: %s, Eval interval: %s, Patience: %s\n" "$(timestamp)" "$SMOLEXPERT_MAX_STEPS" "$SMOLEXPERT_EVAL_INTERVAL" "$SMOLEXPERT_PATIENCE"
printf "[%s] ====================================================================\n" "$(timestamp)"

uv run python -m scripts.video_vam.train_smolexpert_on_flux2_klein \
    --cache-dir "$FLUX2_RAW_CACHE" \
    --output-dir "$STAGE2_DIR" \
    --device "$DEVICE" \
    --max-steps "$SMOLEXPERT_MAX_STEPS" \
    --eval-interval "$SMOLEXPERT_EVAL_INTERVAL" \
    --patience "$SMOLEXPERT_PATIENCE" \
    --batch-size 16 \
    $DRY_RUN_FLAG

if [[ -f "$STAGE2_DIR/eval_metrics.json" ]]; then
    printf "[%s] Stage 2 Evaluation Metrics (FLUX.2 [klein] Base):\n" "$(timestamp)"
    cat "$STAGE2_DIR/eval_metrics.json"
    printf "\n"
fi
printf "[%s] Stage 2 completed successfully!\n" "$(timestamp)"

# =============================================================================
# STAGE 3: Cosmos 7B Video-LoRA fine-tuning for 5000 steps on full dataset
# (with linear warmup and cosine decay scheduler)
# =============================================================================
STAGE3_LORA_DIR="$RUNS_DIR/cosmos7b-videolora"
COSMOS7B_LORA_FILE="$STAGE3_LORA_DIR/cosmos7b_video_lora.safetensors"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 3: Cosmos 7B Video-LoRA Fine-Tuning (%s steps, cosine decay) ===\n" "$(timestamp)" "$LORA_MAX_STEPS"
printf "[%s] LoRA Output Dir: %s\n" "$(timestamp)" "$STAGE3_LORA_DIR"
printf "[%s] Max steps: %s, Warmup steps: %s\n" "$(timestamp)" "$LORA_MAX_STEPS" "$LORA_WARMUP_STEPS"
printf "[%s] ====================================================================\n" "$(timestamp)"

uv run python -m scripts.video_vam.train_cosmos7b_video_lora \
    --output-dir "$STAGE3_LORA_DIR" \
    --dataset-root "$DATASET_ROOT" \
    --lora-rank 16 \
    --lora-alpha 16.0 \
    --target-blocks "14,20" \
    --device "$DEVICE" \
    --dtype "bfloat16" \
    --max-steps "$LORA_MAX_STEPS" \
    --warmup-steps "$LORA_WARMUP_STEPS" \
    $DRY_RUN_FLAG

printf "[%s] Stage 3 complete! Video-LoRA fine-tuning saved to %s\n" "$(timestamp)" "$COSMOS7B_LORA_FILE"

# =============================================================================
# STAGE 4: Feature extraction with the new LoRA weights
# =============================================================================
COSMOS7B_ADAPTED_CACHE="$CACHE/cosmos7b-adapted-features"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 4: Feature Extraction with New Video-LoRA Weights ===\n" "$(timestamp)"
printf "[%s] LoRA weights: %s\n" "$(timestamp)" "$COSMOS7B_LORA_FILE"
printf "[%s] Output cache: %s\n" "$(timestamp)" "$COSMOS7B_ADAPTED_CACHE"
printf "[%s] ====================================================================\n" "$(timestamp)"

uv run python -m scripts.video_vam.extract_cosmos7b_features \
    --lora-path "$COSMOS7B_LORA_FILE" \
    --output-dir "$COSMOS7B_ADAPTED_CACHE" \
    --dataset-root "$DATASET_ROOT" \
    --stride "$STRIDE" \
    --hidden-layers "14,20" \
    --pool-spatial 2 \
    --concat-layers \
    --device "$DEVICE" \
    --dtype "bfloat16" \
    $DRY_RUN_FLAG

printf "[%s] Stage 4 complete! Adapted features extracted to %s\n" "$(timestamp)" "$COSMOS7B_ADAPTED_CACHE"

# =============================================================================
# STAGE 5: SmolExpert on newly adapted Cosmos 7B features until convergence
# =============================================================================
STAGE5_POLICY_DIR="$RUNS_DIR/cosmos7b-adapted-smolexpert"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 5: Train SmolExpert on Adapted Cosmos 7B Features ===\n" "$(timestamp)"
printf "[%s] Cache: %s\n" "$(timestamp)" "$COSMOS7B_ADAPTED_CACHE"
printf "[%s] Output directory: %s\n" "$(timestamp)" "$STAGE5_POLICY_DIR"
printf "[%s] Max steps: %s, Eval interval: %s, Patience: %s\n" "$(timestamp)" "$SMOLEXPERT_MAX_STEPS" "$SMOLEXPERT_EVAL_INTERVAL" "$SMOLEXPERT_PATIENCE"
printf "[%s] ====================================================================\n" "$(timestamp)"

uv run python -m scripts.video_vam.train_smolexpert_on_cosmos7b \
    --cache-dir "$COSMOS7B_ADAPTED_CACHE" \
    --output-dir "$STAGE5_POLICY_DIR" \
    --device "$DEVICE" \
    --max-steps "$SMOLEXPERT_MAX_STEPS" \
    --eval-interval "$SMOLEXPERT_EVAL_INTERVAL" \
    --patience "$SMOLEXPERT_PATIENCE" \
    --batch-size 16 \
    $DRY_RUN_FLAG

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] FINAL PERFORMANCE SUMMARY ACROSS ALL MODELS\n" "$(timestamp)"
printf "[%s] ====================================================================\n" "$(timestamp)"

if [[ -f "$STAGE1_DIR/eval_metrics.json" ]]; then
    printf "\n[%s] Cosmos 7B Base (Stage 1):\n" "$(timestamp)"
    cat "$STAGE1_DIR/eval_metrics.json"
fi

if [[ -f "$STAGE2_DIR/eval_metrics.json" ]]; then
    printf "\n[%s] FLUX.2 [klein] Base (Stage 2):\n" "$(timestamp)"
    cat "$STAGE2_DIR/eval_metrics.json"
fi

if [[ -f "$STAGE5_POLICY_DIR/eval_metrics.json" ]]; then
    printf "\n[%s] Cosmos 7B Adapted Video-LoRA (Stage 5):\n" "$(timestamp)"
    cat "$STAGE5_POLICY_DIR/eval_metrics.json"
fi

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] ALL 5 STAGES OF THE REAL GPU TRAINING QUEUE COMPLETED SUCCESSFULLY!\n" "$(timestamp)"
printf "[%s] ====================================================================\n" "$(timestamp)"
