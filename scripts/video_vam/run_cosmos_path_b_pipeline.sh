#!/usr/bin/env bash
# Complete Cosmos T=16 -> T=2 Path B Pipeline (Reconstructed 16-Frame pool2 Features):
# Stage 1: Build Distilled Path B Feature Caches (T=2 + Linear Readout Head + pool2 -> 4,800 tokens)
# Stage 2: Train SmolExpert Action Policy and evaluate RMSE against 13.06° gold standard

set -Eeuo pipefail

REPO="/home/anton/lerobot-video-vam"
PYTHON="$REPO/.venv/bin/python"
CACHE="/home/anton/.cache/video-vam"
DATASET_ROOT="$CACHE/cube-out-of-box-dataset"
CHECKPOINT="$CACHE/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
TOKENIZER="$CACHE/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
PROMPT="$CACHE/prompt-embeddings/cube-out-of-box-t5-11b.safetensors"
OLD_LORA="$CACHE/runs/cosmos2b-video-lora-20260828/paused-step6000/best_lora.safetensors"
SPLIT="$CACHE/splits/rehearsal-stride20.json"
TRAIN_EPISODES=({0..31})
VAL_EPISODES=({32..39})

DISTILL_RUN_DIR="$CACHE/runs/cosmos-t2-distillation-t16linear-20260901"
BEST_LORA="$DISTILL_RUN_DIR/best_lora.safetensors"
BEST_HEAD="$DISTILL_RUN_DIR/best_head.safetensors"

MASTER_LOG="$CACHE/runs/distillation-pathb-pipeline-20260901.log"
mkdir -p "$CACHE/runs"
exec > >(tee -a "$MASTER_LOG") 2>&1

cd "$REPO"
# shellcheck source=/dev/null
source scripts/video_vam/cosmos_cuda_env.sh
# shellcheck source=/dev/null
source scripts/video_vam/gpu_lock.sh
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

timestamp() {
    date --iso-8601=seconds
}

printf "[%s] STARTING COSMOS T=2 PATH B (LINEAR 16-FRAME POOL2) PIPELINE\n" "$(timestamp)"
acquire_gpu_lock cosmos-distillation-pathb-pipeline

# -----------------------------------------------------------------------------
# STAGE 1: Build Distilled Path B Feature Caches (4,800 tokens per sample)
# -----------------------------------------------------------------------------
DISTILLED_TRAIN_CACHE="$CACHE/cosmos-t2-distilled-pathb-train0-31-stride3-pool2"
DISTILLED_VAL_CACHE="$CACHE/cosmos-t2-distilled-pathb-val32-39-stride20-pool2"

build_pathb_cache() {
    local episodes="$1"
    local stride="$2"
    local output="$3"
    local mode=--overwrite
    if [[ -d "$output" ]] && compgen -G "$output/*.safetensors" > /dev/null; then
        mode=--resume
    fi

    printf "[%s] Building Path B cache episodes=%s stride=%s out=%s mode=%s\n" \
        "$(timestamp)" "$episodes" "$stride" "$output" "$mode"
    "$PYTHON" -m scripts.video_vam.build_cosmos_feature_cache \
        --root "$DATASET_ROOT" \
        --episodes $episodes \
        --stride "$stride" \
        --prompt "$PROMPT" \
        --checkpoint "$CHECKPOINT" \
        --lora-weights "$BEST_LORA" \
        --merge-lora-weights "$OLD_LORA" \
        --lora-blocks 0-19 \
        --readout-head-weights "$BEST_HEAD" \
        --tokenizer "$TOKENIZER" \
        --output-dir "$output" \
        --sigma 80 \
        --seed 0 \
        --state-t 2 \
        --vae-input-mode observed_prefix \
        --context-transform pool2 \
        "$mode"
}

printf "\n[%s] === STAGE 1: Building Distilled Path B Feature Caches (4,800 tokens) ===\n" "$(timestamp)"
build_pathb_cache "${TRAIN_EPISODES[*]}" 3 "$DISTILLED_TRAIN_CACHE"
build_pathb_cache "${VAL_EPISODES[*]}" 20 "$DISTILLED_VAL_CACHE"

# -----------------------------------------------------------------------------
# STAGE 2: Train SmolExpert Action Policy on Path B Features
# -----------------------------------------------------------------------------
POLICY_RUN_DIR="$CACHE/runs/cosmos-t2-distilled-pathb-smolexpert-20260901"
mkdir -p "$POLICY_RUN_DIR"

printf "\n[%s] === STAGE 2: Training SmolExpert on Path B Features -> %s ===\n" "$(timestamp)" "$POLICY_RUN_DIR"
"$PYTHON" -m scripts.video_vam.train_smolexpert_on_cosmos \
    --manifest "$DISTILLED_TRAIN_CACHE/manifest.json" \
    --val-manifest "$DISTILLED_VAL_CACHE/manifest.json" \
    --split "$SPLIT" \
    --train-episodes "${TRAIN_EPISODES[@]}" \
    --output-dir "$POLICY_RUN_DIR/smolexpert" \
    --expert-checkpoint lerobot/smolvla_base \
    --batch-size 8 \
    --max-steps 500000 \
    --max-hours 2 \
    --val-every 1000 \
    --patience 10 \
    --lr 1e-4 \
    --warmup-steps 1000 \
    --num-steps 10 \
    --context-transform auto \
    --seed 0 \
    --wandb-project video-vam-world2action \
    --run-name "cosmos-t2-distilled-pathb-smolexpert-20260901"

date --iso-8601=seconds > "$POLICY_RUN_DIR/COMPLETE"

printf "\n[%s] ALL PATH B PIPELINE STAGES COMPLETED SUCCESSFULLY!\n" "$(timestamp)"
