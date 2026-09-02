#!/usr/bin/env bash
# Exploration Master Queue:
# Step 1 (Option A): T=2 with Self-Attention Scale 0.60 -> Feature Cache -> SmolExpert
# Step 2 (Option B): T=4 with 2 Clean + 2 Noise Slots -> Feature Cache -> SmolExpert
# Step 3 (Option C1): T=2 Unmodulated Baseline with LR = 5e-5 -> SmolExpert
# Step 4 (Option C2): T=2 Unmodulated Baseline with LR = 2e-4 -> SmolExpert

set -Eeuo pipefail

REPO="/home/anton/lerobot-video-vam"
PYTHON="$REPO/.venv/bin/python"
CACHE="/home/anton/.cache/video-vam"
DATASET_ROOT="$CACHE/cube-out-of-box-dataset"
CHECKPOINT="$CACHE/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
TOKENIZER="$CACHE/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
PROMPT="$CACHE/prompt-embeddings/cube-out-of-box-t5-11b.safetensors"
LORA="$CACHE/runs/cosmos2b-video-lora-20260828/paused-step6000/best_lora.safetensors"
SPLIT="$CACHE/splits/rehearsal-stride20.json"
TRAIN_EPISODES=({0..31})
VAL_EPISODES=({32..39})

QUEUE_LOG="$CACHE/runs/exploration-queue-20260830.log"
mkdir -p "$CACHE/runs"
exec > >(tee -a "$QUEUE_LOG") 2>&1

cd "$REPO"
source scripts/video_vam/cosmos_cuda_env.sh
source scripts/video_vam/gpu_lock.sh
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

timestamp() {
    date --iso-8601=seconds
}

printf "[%s] STARTING EXPLORATION QUEUE (A + B + C)
" "$(timestamp)"
acquire_gpu_lock cosmos-exploration-queue

build_cache() {
    local episodes="$1"
    local stride="$2"
    local state_t="$3"
    local scale="$4"
    local output="$5"
    local mode=--overwrite
    if [[ -d "$output" ]] && compgen -G "$output/*.safetensors" > /dev/null; then
        mode=--resume
    fi
    printf "[%s] Building cache episodes=%s stride=%s state_t=%s scale=%s out=%s mode=%s
"         "$(timestamp)" "$episodes" "$stride" "$state_t" "$scale" "$output" "$mode"
    "$PYTHON" -m scripts.video_vam.build_cosmos_feature_cache         --root "$DATASET_ROOT"         --episodes $episodes         --stride "$stride"         --prompt "$PROMPT"         --checkpoint "$CHECKPOINT"         --lora-weights "$LORA"         --tokenizer "$TOKENIZER"         --output-dir "$output"         --sigma 80         --seed 0         --state-t "$state_t"         --self-attn-scale "$scale"         --vae-input-mode observed_prefix         --context-transform none         "$mode"
}

# =============================================================================
# STEP 1 (OPTION A): T=2 with Self-Attention Scale 0.60
# =============================================================================
CACHE_T2_S06_TRAIN="$CACHE/cosmos-videolora-train0-31-stride3-statet2-scale060"
CACHE_T2_S06_VAL="$CACHE/cosmos-videolora-val32-39-stride20-statet2-scale060"
RUN_T2_S06="$CACHE/runs/cosmos-t2-scale060-smolexpert-20260830"

if [[ -f "$RUN_T2_S06/COMPLETE" ]]; then
    printf "
[%s] === STEP 1 (T=2, scale=0.60) ALREADY COMPLETED -> SKIPPING ===
" "$(timestamp)"
else
    printf "
[%s] === STEP 1: OPTION A (T=2, Self-Attn Scale 0.60) ===
" "$(timestamp)"
    build_cache "${TRAIN_EPISODES[*]}" 3 2 0.6 "$CACHE_T2_S06_TRAIN"
    build_cache "${VAL_EPISODES[*]}" 20 2 0.6 "$CACHE_T2_S06_VAL"
    mkdir -p "$RUN_T2_S06"
    printf "[%s] Training SmolExpert on T=2 (scale=0.60) -> %s
" "$(timestamp)" "$RUN_T2_S06"
    "$PYTHON" -m scripts.video_vam.train_smolexpert_on_cosmos         --manifest "$CACHE_T2_S06_TRAIN/manifest.json"         --val-manifest "$CACHE_T2_S06_VAL/manifest.json"         --split "$SPLIT"         --train-episodes "${TRAIN_EPISODES[@]}"         --output-dir "$RUN_T2_S06/smolexpert"         --expert-checkpoint lerobot/smolvla_base         --batch-size 8         --max-steps 500000         --max-hours 2         --val-every 1000         --patience 10         --lr 1e-4         --warmup-steps 1000         --num-steps 10         --context-transform none         --seed 0         --wandb-project video-vam-world2action         --run-name "cosmos-t2-scale060-smolexpert-20260830"
    date --iso-8601=seconds > "$RUN_T2_S06/COMPLETE"
fi

# =============================================================================
# STEP 2 (OPTION B): T=4 with 2 Clean + 2 Noise Query Slots
# =============================================================================
CACHE_T4_TRAIN="$CACHE/cosmos-videolora-train0-31-stride3-statet4-unpooled"
CACHE_T4_VAL="$CACHE/cosmos-videolora-val32-39-stride20-statet4-unpooled"
RUN_T4="$CACHE/runs/cosmos-t4-noise-smolexpert-20260830"

if [[ -f "$RUN_T4/COMPLETE" ]]; then
    printf "
[%s] === STEP 2 (T=4 noise slots) ALREADY COMPLETED -> SKIPPING ===
" "$(timestamp)"
else
    printf "
[%s] === STEP 2: OPTION B (T=4 with 2 Clean + 2 Noise Slots) ===
" "$(timestamp)"
    build_cache "${TRAIN_EPISODES[*]}" 3 4 1.0 "$CACHE_T4_TRAIN"
    build_cache "${VAL_EPISODES[*]}" 20 4 1.0 "$CACHE_T4_VAL"
    mkdir -p "$RUN_T4"
    printf "[%s] Training SmolExpert on T=4 -> %s
" "$(timestamp)" "$RUN_T4"
    "$PYTHON" -m scripts.video_vam.train_smolexpert_on_cosmos         --manifest "$CACHE_T4_TRAIN/manifest.json"         --val-manifest "$CACHE_T4_VAL/manifest.json"         --split "$SPLIT"         --train-episodes "${TRAIN_EPISODES[@]}"         --output-dir "$RUN_T4/smolexpert"         --expert-checkpoint lerobot/smolvla_base         --batch-size 8         --max-steps 500000         --max-hours 2         --val-every 1000         --patience 10         --lr 1e-4         --warmup-steps 1000         --num-steps 10         --context-transform none         --seed 0         --wandb-project video-vam-world2action         --run-name "cosmos-t4-noise-smolexpert-20260830"
    date --iso-8601=seconds > "$RUN_T4/COMPLETE"
fi

# =============================================================================
# STEP 3 (OPTION C1): T=2 Unmodulated Baseline with LR = 5e-5
# =============================================================================
CACHE_T2_STD_TRAIN="$CACHE/cosmos-videolora-train0-31-stride3-statet2-unpooled"
CACHE_T2_STD_VAL="$CACHE/cosmos-videolora-val32-39-stride20-statet2-unpooled"
RUN_T2_LR5E5="$CACHE/runs/cosmos-t2-lr5e5-smolexpert-20260830"

if [[ -f "$RUN_T2_LR5E5/COMPLETE" ]]; then
    printf "
[%s] === STEP 3 (T=2, LR=5e-5) ALREADY COMPLETED -> SKIPPING ===
" "$(timestamp)"
else
    printf "
[%s] === STEP 3: OPTION C1 (T=2 Unmodulated with LR = 5e-5) ===
" "$(timestamp)"
    mkdir -p "$RUN_T2_LR5E5"
    "$PYTHON" -m scripts.video_vam.train_smolexpert_on_cosmos         --manifest "$CACHE_T2_STD_TRAIN/manifest.json"         --val-manifest "$CACHE_T2_STD_VAL/manifest.json"         --split "$SPLIT"         --train-episodes "${TRAIN_EPISODES[@]}"         --output-dir "$RUN_T2_LR5E5/smolexpert"         --expert-checkpoint lerobot/smolvla_base         --batch-size 8         --max-steps 500000         --max-hours 2         --val-every 1000         --patience 10         --lr 5e-5         --warmup-steps 1000         --num-steps 10         --context-transform none         --seed 0         --wandb-project video-vam-world2action         --run-name "cosmos-t2-lr5e5-smolexpert-20260830"
    date --iso-8601=seconds > "$RUN_T2_LR5E5/COMPLETE"
fi

# =============================================================================
# STEP 4 (OPTION C2): T=2 Unmodulated Baseline with LR = 2e-4
# =============================================================================
RUN_T2_LR2E4="$CACHE/runs/cosmos-t2-lr2e4-smolexpert-20260830"

if [[ -f "$RUN_T2_LR2E4/COMPLETE" ]]; then
    printf "
[%s] === STEP 4 (T=2, LR=2e-4) ALREADY COMPLETED -> SKIPPING ===
" "$(timestamp)"
else
    printf "
[%s] === STEP 4: OPTION C2 (T=2 Unmodulated with LR = 2e-4) ===
" "$(timestamp)"
    mkdir -p "$RUN_T2_LR2E4"
    "$PYTHON" -m scripts.video_vam.train_smolexpert_on_cosmos         --manifest "$CACHE_T2_STD_TRAIN/manifest.json"         --val-manifest "$CACHE_T2_STD_VAL/manifest.json"         --split "$SPLIT"         --train-episodes "${TRAIN_EPISODES[@]}"         --output-dir "$RUN_T2_LR2E4/smolexpert"         --expert-checkpoint lerobot/smolvla_base         --batch-size 8         --max-steps 500000         --max-hours 2         --val-every 1000         --patience 10         --lr 2e-4         --warmup-steps 1000         --num-steps 10         --context-transform none         --seed 0         --wandb-project video-vam-world2action         --run-name "cosmos-t2-lr2e4-smolexpert-20260830"
    date --iso-8601=seconds > "$RUN_T2_LR2E4/COMPLETE"
fi

printf "
[%s] ALL EXPLORATION QUEUE RUNS COMPLETED SUCCESSFULLY!
" "$(timestamp)"
