#!/usr/bin/env bash
# Observed-only Cosmos (state_t=2) unpooled cache, then SmolExpert to plateau.
# Order: train cache -> val cache -> expert. Holds the GPU lock for the whole queue.
set -Eeuo pipefail

REPO=/home/anton/lerobot-video-vam
PYTHON="$REPO/.venv/bin/python"
CACHE=/home/anton/.cache/video-vam
RUN_ROOT="$CACHE/runs/cosmos2b-statet2-unpooled-smolexpert-20260828"
TRAIN_CACHE="$CACHE/cosmos-videolora-train0-31-stride3-statet2-unpooled"
VAL_CACHE="$CACHE/cosmos-videolora-val32-39-stride20-statet2-unpooled"
LORA_RUN="$CACHE/runs/cosmos2b-video-lora-20260828"
BEST_LORA="$LORA_RUN/paused-step6000/best_lora.safetensors"
SPLIT="$CACHE/splits/rehearsal-stride20.json"
DATASET_ROOT="$CACHE/cube-out-of-box-dataset"
CHECKPOINT="$CACHE/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
TOKENIZER="$CACHE/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
PROMPT="$CACHE/prompt-embeddings/cube-out-of-box-t5-11b.safetensors"

cd "$REPO"
# shellcheck source=/dev/null
source scripts/video_vam/cosmos_cuda_env.sh
# shellcheck source=/dev/null
source scripts/video_vam/gpu_lock.sh
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/queue.log") 2>&1
stage=initializing
trap 'status=$?; set +e; printf "[%s] FAILED stage=%s status=%s\n" "$(date --iso-8601=seconds)" "$stage" "$status"; printf "%s\n" "$stage" > "$RUN_ROOT/FAILED"; exit "$status"' ERR

if [[ -f "$RUN_ROOT/COMPLETE" ]]; then
    printf "[%s] queue already complete; exiting\n" "$(date --iso-8601=seconds)"
    exit 0
fi
rm -f -- "$RUN_ROOT/FAILED"

TRAIN_EPISODES=({0..31})
VAL_EPISODES=({32..39})

if [[ ! -f "$BEST_LORA" || ! -f "$CHECKPOINT" || ! -f "$SPLIT" ]]; then
    printf 'missing LoRA, Cosmos checkpoint, or split file\n' >&2
    exit 1
fi

stage=gpu-lock
acquire_gpu_lock cosmos-statet2-unpooled

build_cache() {
    local episodes="$1"
    local stride="$2"
    local output="$3"
    local mode=--overwrite
    if [[ -f "$output/manifest.json" ]]; then
        mode=--resume
    fi
    printf "[%s] cache episodes=%s stride=%s out=%s mode=%s\n" \
        "$(date --iso-8601=seconds)" "$episodes" "$stride" "$output" "$mode"
    "$PYTHON" -m scripts.video_vam.build_cosmos_feature_cache \
        --root "$DATASET_ROOT" \
        --episodes $episodes \
        --stride "$stride" \
        --prompt "$PROMPT" \
        --checkpoint "$CHECKPOINT" \
        --lora-weights "$BEST_LORA" \
        --tokenizer "$TOKENIZER" \
        --output-dir "$output" \
        --sigma 80 \
        --seed 0 \
        --state-t 2 \
        --vae-input-mode observed_prefix \
        --context-transform none \
        "$mode"
}

stage=train-cache
build_cache "${TRAIN_EPISODES[*]}" 3 "$TRAIN_CACHE"
stage=val-cache
build_cache "${VAL_EPISODES[*]}" 20 "$VAL_CACHE"

stage=smolexpert
"$PYTHON" -m scripts.video_vam.train_smolexpert_on_cosmos \
    --manifest "$TRAIN_CACHE/manifest.json" \
    --val-manifest "$VAL_CACHE/manifest.json" \
    --split "$SPLIT" \
    --train-episodes "${TRAIN_EPISODES[@]}" \
    --output-dir "$RUN_ROOT/smolexpert" \
    --expert-checkpoint lerobot/smolvla_base \
    --batch-size 8 \
    --max-steps 500000 \
    --max-hours 12 \
    --val-every 1000 \
    --patience 10 \
    --lr 1e-4 \
    --warmup-steps 1000 \
    --num-steps 10 \
    --context-transform none \
    --seed 0 \
    --wandb-project video-vam-world2action \
    --run-name cosmos2b-videolora-statet2-unpooled-smolexpert-20260828

date --iso-8601=seconds > "$RUN_ROOT/COMPLETE"
printf "[%s] COMPLETE\n" "$(date --iso-8601=seconds)"
