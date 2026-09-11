#!/usr/bin/env bash
# Automated Cosmos 14B End-to-End True Convergence & Quantized Video-LoRA Pipeline
#
# STAGE 1: Cosmos 14B Base SmolExpert Training to TRUE Convergence (patience=60, eval_interval=500, max_steps=60000, lr=3e-5)
# STAGE 2: Cosmos 14B Quantized Video-LoRA (QLoRA / FP8 on single RTX 4090, targeting attention across 36 blocks)
# STAGE 3: Feature Extraction on Adapted 14B LoRA into dedicated cache (layers 18 & 30, 10,240-dim representation)
# STAGE 4: SmolExpert Training on 14B LoRA Features to Full Convergence
# STAGE 5: Evaluation & Leaderboard / Research Diary Update

set -Eeuo pipefail

REPO="/home/anton/lerobot-video-vam"
CACHE="/home/anton/.cache/video-vam"
LOG_DIR="$REPO/logs"
RUNS_DIR="$CACHE/runs"
LOG_FILE="${LOG_FILE:-$LOG_DIR/cosmos14b_pipeline_$(date +%Y%m%d).log}"
CACHE_LOG="$RUNS_DIR/cosmos14b-pipeline-$(date +%Y%m%d).log"

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
ARM_NAME="${GPU_LOCK_ARM:-cosmos14b-pipeline}"

# Pipeline Hyperparameters
BASE_CACHE="$CACHE/cosmos14b-features"
ADAPTED_CACHE="$CACHE/cosmos14b-adapted-features"
CHECKPOINT_DIR="$CACHE/cosmos-14b"
DATASET_ROOT="$CACHE/cube-out-of-box-dataset"

# Stage 1 & 4: SmolExpert settings
SMOL_MAX_STEPS="${SMOL_MAX_STEPS:-60000}"
SMOL_MIN_STEPS="${SMOL_MIN_STEPS:-20000}"
SMOL_EVAL_INTERVAL="${SMOL_EVAL_INTERVAL:-500}"
SMOL_PATIENCE="${SMOL_PATIENCE:-60}"
SMOL_LR="${SMOL_LR:-3e-5}"
SMOL_BATCH_SIZE="${SMOL_BATCH_SIZE:-16}"

# Stage 2: Video-LoRA settings
LORA_MAX_STEPS="${LORA_MAX_STEPS:-2000}"
LORA_WARMUP_STEPS="${LORA_WARMUP_STEPS:-200}"
LORA_LR="${LORA_LR:-1e-4}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"

timestamp() {
    date --iso-8601=seconds
}

printf "[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] STARTING COSMOS 14B TRUE CONVERGENCE & QUANTIZED VIDEO-LORA PIPELINE\n" "$(timestamp)"
printf "[%s] Log: %s\n" "$(timestamp)" "$LOG_FILE"
printf "[%s] Device: %s\n" "$(timestamp)" "$DEVICE"
printf "[%s] Stage 1 Base SmolExpert: max_steps=%s, min_steps=%s, eval_interval=%s, patience=%s, lr=%s\n" \
    "$(timestamp)" "$SMOL_MAX_STEPS" "$SMOL_MIN_STEPS" "$SMOL_EVAL_INTERVAL" "$SMOL_PATIENCE" "$SMOL_LR"
printf "[%s] Stage 2 Video-LoRA: rank=%s, alpha=%s, max_steps=%s, lr=%s\n" \
    "$(timestamp)" "$LORA_RANK" "$LORA_ALPHA" "$LORA_MAX_STEPS" "$LORA_LR"
printf "[%s] ====================================================================\n" "$(timestamp)"

# Acquire GPU lock for the entire sequential pipeline
printf "[%s] Requesting GPU lock under arm=%s...\n" "$(timestamp)" "$ARM_NAME"
acquire_gpu_lock "$ARM_NAME"

# =============================================================================
# STAGE 1: Cosmos 14B Base SmolExpert Training to TRUE Convergence
# =============================================================================
STAGE1_DIR="$RUNS_DIR/cosmos14b-base-converged-smolexpert"
printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 1: Cosmos 14B Base SmolExpert Training to TRUE Convergence ===\n" "$(timestamp)"
printf "[%s] Features Cache: %s\n" "$(timestamp)" "$BASE_CACHE"
printf "[%s] Output Directory: %s\n" "$(timestamp)" "$STAGE1_DIR"
printf "[%s] ====================================================================\n" "$(timestamp)"

uv run python -m scripts.video_vam.train_smolexpert_on_cosmos14b \
    --cache-dir "$BASE_CACHE" \
    --output-dir "$STAGE1_DIR" \
    --lr "$SMOL_LR" \
    --lr-scheduler cosine \
    --warmup-steps 1000 \
    --min-steps "$SMOL_MIN_STEPS" \
    --max-steps "$SMOL_MAX_STEPS" \
    --eval-interval "$SMOL_EVAL_INTERVAL" \
    --patience "$SMOL_PATIENCE" \
    --batch-size "$SMOL_BATCH_SIZE" \
    --device "$DEVICE"

if [[ -f "$STAGE1_DIR/eval_metrics.json" ]]; then
    printf "[%s] Stage 1 Evaluation Metrics (Cosmos 14B Base Converged):\n" "$(timestamp)"
    cat "$STAGE1_DIR/eval_metrics.json"
    printf "\n"
fi
printf "[%s] Stage 1 completed successfully!\n" "$(timestamp)"

# =============================================================================
# STAGE 2: Cosmos 14B Quantized Video-LoRA (QLoRA / FP8)
# =============================================================================
STAGE2_DIR="$RUNS_DIR/cosmos14b-videolora"
LORA_WEIGHTS="$STAGE2_DIR/cosmos14b_video_lora.safetensors"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 2: Cosmos 14B Quantized Video-LoRA Training (FP8 QLoRA) ===\n" "$(timestamp)"
printf "[%s] Checkpoint: %s\n" "$(timestamp)" "$CHECKPOINT_DIR"
printf "[%s] Output Directory: %s\n" "$(timestamp)" "$STAGE2_DIR"
printf "[%s] Attention Projections: to_q, to_k, to_v, to_out.0 across all 36 blocks\n" "$(timestamp)"
printf "[%s] ====================================================================\n" "$(timestamp)"

uv run python -m scripts.video_vam.train_cosmos14b_video_lora \
    --checkpoint-path "$CHECKPOINT_DIR" \
    --output-dir "$STAGE2_DIR" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --target-blocks "all" \
    --target-modules "to_q,to_k,to_v,to_out.0" \
    --lr "$LORA_LR" \
    --max-steps "$LORA_MAX_STEPS" \
    --warmup-steps "$LORA_WARMUP_STEPS" \
    --batch-size 1 \
    --dataset-root "$DATASET_ROOT" \
    --device "$DEVICE" \
    --dtype "bfloat16"

printf "[%s] Stage 2 complete! Video-LoRA weights saved to %s\n" "$(timestamp)" "$LORA_WEIGHTS"

# =============================================================================
# STAGE 3: Feature Extraction on Adapted 14B LoRA into Dedicated Cache
# =============================================================================
printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 3: Extracting Adapted 14B Features (Layers 18 & 30) ===\n" "$(timestamp)"
printf "[%s] LoRA weights: %s\n" "$(timestamp)" "$LORA_WEIGHTS"
printf "[%s] Output cache: %s\n" "$(timestamp)" "$ADAPTED_CACHE"
printf "[%s] ====================================================================\n" "$(timestamp)"

uv run python -m scripts.video_vam.extract_cosmos14b_features \
    --checkpoint-path "$CHECKPOINT_DIR" \
    --lora-path "$LORA_WEIGHTS" \
    --output-dir "$ADAPTED_CACHE" \
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

printf "[%s] Stage 3 complete! Adapted features saved to %s\n" "$(timestamp)" "$ADAPTED_CACHE"

# =============================================================================
# STAGE 4: SmolExpert Training on 14B LoRA Features to Full Convergence
# =============================================================================
STAGE4_DIR="$RUNS_DIR/cosmos14b-adapted-smolexpert"
printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 4: Train SmolExpert on Adapted 14B LoRA Features ===\n" "$(timestamp)"
printf "[%s] Features Cache: %s\n" "$(timestamp)" "$ADAPTED_CACHE"
printf "[%s] Output Directory: %s\n" "$(timestamp)" "$STAGE4_DIR"
printf "[%s] ====================================================================\n" "$(timestamp)"

uv run python -m scripts.video_vam.train_smolexpert_on_cosmos14b \
    --cache-dir "$ADAPTED_CACHE" \
    --output-dir "$STAGE4_DIR" \
    --lr "$SMOL_LR" \
    --lr-scheduler cosine \
    --warmup-steps 1000 \
    --min-steps "$SMOL_MIN_STEPS" \
    --max-steps "$SMOL_MAX_STEPS" \
    --eval-interval "$SMOL_EVAL_INTERVAL" \
    --patience "$SMOL_PATIENCE" \
    --batch-size "$SMOL_BATCH_SIZE" \
    --device "$DEVICE"

if [[ -f "$STAGE4_DIR/eval_metrics.json" ]]; then
    printf "[%s] Stage 4 Evaluation Metrics (Cosmos 14B Video-LoRA Adapted):\n" "$(timestamp)"
    cat "$STAGE4_DIR/eval_metrics.json"
    printf "\n"
fi
printf "[%s] Stage 4 completed successfully!\n" "$(timestamp)"

# =============================================================================
# STAGE 5: Update Leaderboard and Research Diary
# =============================================================================
printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 5: Updating Leaderboard & Research Diary ===\n" "$(timestamp)"
printf "[%s] ====================================================================\n" "$(timestamp)"

uv run python -m scripts.video_vam.update_cosmos14b_leaderboard \
    --base-metrics "$STAGE1_DIR/eval_metrics.json" \
    --lora-metrics "$STAGE4_DIR/eval_metrics.json"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] FULL COSMOS 14B PIPELINE COMPLETED SUCCESSFULLY!\n" "$(timestamp)"
printf "[%s] ====================================================================\n" "$(timestamp)"
