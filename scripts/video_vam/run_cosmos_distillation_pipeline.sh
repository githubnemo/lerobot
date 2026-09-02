#!/usr/bin/env bash
# Complete Cosmos T=16 -> T=2 Direct Representation Distillation Pipeline:
# Stage 1: Verify Teacher T=16 Targets
# Stage 2: Train Student T=2 LoRA blocks 0-19 via Direct Manifold Distillation
# Stage 3: Build Distilled Student Feature Caches (T=2 unpooled 2,400 tokens)
# Stage 4: Train SmolExpert Action Policy and evaluate RMSE

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

MASTER_LOG="$CACHE/runs/distillation-direct-pipeline-20260902.log"
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

printf "[%s] STARTING COSMOS T=16 -> T=2 DIRECT MANIFOLD DISTILLATION PIPELINE\n" "$(timestamp)"
acquire_gpu_lock cosmos-direct-distillation-pipeline

build_cache() {
    local episodes="$1"
    local stride="$2"
    local state_t="$3"
    local transform="$4"
    local output="$5"
    local lora_weights="$6"
    local merge_lora="${7:-}"
    local lora_blocks="${8:-}"
    local mode=--overwrite
    if [[ -d "$output" ]] && compgen -G "$output/*.safetensors" > /dev/null; then
        mode=--resume
    fi

    local -a extra_args=()
    if [[ -n "$merge_lora" ]]; then
        extra_args+=(--merge-lora-weights "$merge_lora")
    fi
    if [[ -n "$lora_blocks" ]]; then
        extra_args+=(--lora-blocks "$lora_blocks")
    fi

    printf "[%s] Building cache episodes=%s stride=%s state_t=%s transform=%s out=%s mode=%s\n" \
        "$(timestamp)" "$episodes" "$stride" "$state_t" "$transform" "$output" "$mode"
    "$PYTHON" -m scripts.video_vam.build_cosmos_feature_cache \
        --root "$DATASET_ROOT" \
        --episodes $episodes \
        --stride "$stride" \
        --prompt "$PROMPT" \
        --checkpoint "$CHECKPOINT" \
        --lora-weights "$lora_weights" \
        --tokenizer "$TOKENIZER" \
        --output-dir "$output" \
        --sigma 80 \
        --seed 0 \
        --state-t "$state_t" \
        --vae-input-mode observed_prefix \
        --context-transform "$transform" \
        "${extra_args[@]}" \
        "$mode"
}

# -----------------------------------------------------------------------------
# STAGE 1: Verify Teacher T=16 Targets
# -----------------------------------------------------------------------------
TEACHER_TRAIN_CACHE="$CACHE/cosmos-t16-teacher-train0-31-stride3-unpooled"
TEACHER_VAL_CACHE="$CACHE/cosmos-t16-teacher-val32-39-stride20-unpooled"

printf "\n[%s] === STAGE 1: Verifying Teacher T=16 Unpooled Targets ===\n" "$(timestamp)"
build_cache "${TRAIN_EPISODES[*]}" 3 16 none "$TEACHER_TRAIN_CACHE" "$OLD_LORA"
build_cache "${VAL_EPISODES[*]}" 20 16 none "$TEACHER_VAL_CACHE" "$OLD_LORA"

# -----------------------------------------------------------------------------
# STAGE 2: Train Student T=2 LoRA blocks 0-19 via Direct Manifold Distillation
# -----------------------------------------------------------------------------
DISTILL_RUN_DIR="$CACHE/runs/cosmos-t2-direct-distillation-aligned"
mkdir -p "$DISTILL_RUN_DIR"

printf "\n[%s] === STAGE 2: Training Student T=2 LoRA via Direct Distillation -> %s ===\n" "$(timestamp)" "$DISTILL_RUN_DIR"
if [[ -f "$DISTILL_RUN_DIR/COMPLETE" ]]; then
    printf "[%s] Distillation training already completed, skipping.\n" "$(timestamp)"
else
    "$PYTHON" -m scripts.video_vam.train_cosmos_t2_distillation \
        --train-manifest "$TEACHER_TRAIN_CACHE/manifest.json" \
        --val-manifest "$TEACHER_VAL_CACHE/manifest.json" \
        --dataset-root "$DATASET_ROOT" \
        --backbone-checkpoint "$CHECKPOINT" \
        --tokenizer "$TOKENIZER" \
        --prompt "$PROMPT" \
        --merge-lora-weights "$OLD_LORA" \
        --output-dir "$DISTILL_RUN_DIR" \
        --batch-size 4 \
        --lora-rank 16 \
        --lora-alpha 16.0 \
        --lora-lr 1e-4 \
        --cosine-weight 1.0 \
        --max-steps 15000 \
        --val-every 500 \
        --patience 10 \
        --wandb-project video-vam-world2action \
        --run-name "cosmos-t2-direct-distillation-aligned"
    date --iso-8601=seconds > "$DISTILL_RUN_DIR/COMPLETE"
fi

# Ensure best_lora.json sidecar exists for the cache builder
if [[ -f "$DISTILL_RUN_DIR/best.json" && ! -f "$DISTILL_RUN_DIR/best_lora.json" ]]; then
    ln -sfn "$DISTILL_RUN_DIR/best.json" "$DISTILL_RUN_DIR/best_lora.json"
fi

# -----------------------------------------------------------------------------
# STAGE 3: Build Distilled Student Feature Caches (T=2 unpooled 2,400 tokens)
# -----------------------------------------------------------------------------
DISTILLED_TRAIN_CACHE="$CACHE/cosmos-t2-direct-distilled-train0-31-stride3-unpooled"
DISTILLED_VAL_CACHE="$CACHE/cosmos-t2-direct-distilled-val32-39-stride20-unpooled"

printf "\n[%s] === STAGE 3: Building Distilled Student Feature Caches (2,400 tokens) ===\n" "$(timestamp)"
build_cache "${TRAIN_EPISODES[*]}" 3 2 none "$DISTILLED_TRAIN_CACHE" "$DISTILL_RUN_DIR/best_lora.safetensors" "$OLD_LORA" "0-19"
build_cache "${VAL_EPISODES[*]}" 20 2 none "$DISTILLED_VAL_CACHE" "$DISTILL_RUN_DIR/best_lora.safetensors" "$OLD_LORA" "0-19"

# -----------------------------------------------------------------------------
# STAGE 4: Train SmolExpert Action Policy on Direct Distilled Features
# -----------------------------------------------------------------------------
POLICY_RUN_DIR="$CACHE/runs/cosmos-t2-direct-distilled-smolexpert-20260902"
mkdir -p "$POLICY_RUN_DIR"

printf "\n[%s] === STAGE 4: Training SmolExpert on Distilled Features (2,400 tokens) -> %s ===\n" "$(timestamp)" "$POLICY_RUN_DIR"
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
    --context-transform none \
    --seed 0 \
    --wandb-project video-vam-world2action \
    --run-name "cosmos-t2-direct-distilled-smolexpert-20260902"

date --iso-8601=seconds > "$POLICY_RUN_DIR/COMPLETE"

printf "\n[%s] ALL DIRECT DISTILLATION PIPELINE STAGES COMPLETED SUCCESSFULLY!\n" "$(timestamp)"
