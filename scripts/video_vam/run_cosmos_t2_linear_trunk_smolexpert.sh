#!/usr/bin/env bash
# Merge the original Cosmos video LoRA into the 2B trunk, apply the T=2 linear-readout
# LoRA on blocks 0-19, rebuild state_t=2 caches, then train SmolExpert for actions.
# Order: train cache -> val cache -> expert. Holds the GPU lock for the whole queue.
set -Eeuo pipefail

REPO=/home/anton/lerobot-video-vam
PYTHON="$REPO/.venv/bin/python"
CACHE=/home/anton/.cache/video-vam
DATE_TAG="$(date +%Y%m%d)"
RUN_ROOT="${RUN_ROOT:-$CACHE/runs/cosmos-t2-linear-trunk-smolexpert-$DATE_TAG}"
TRAIN_CACHE="$CACHE/cosmos-t2-linear-trunk-train0-31-stride3-statet2-unpooled"
VAL_CACHE="$CACHE/cosmos-t2-linear-trunk-val32-39-stride20-statet2-unpooled"
OLD_LORA="$CACHE/runs/cosmos2b-video-lora-20260828/paused-step6000/best_lora.safetensors"
NEW_LORA="$CACHE/runs/cosmos-t2-linear-20260829/best_lora.safetensors"
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

# Cache builder looks for best_lora.json next to best_lora.safetensors.
if [[ ! -f "${NEW_LORA%.safetensors}.json" ]]; then
    ln -sfn "$(dirname "$NEW_LORA")/best.json" "${NEW_LORA%.safetensors}.json"
fi

for required in "$OLD_LORA" "$NEW_LORA" "${NEW_LORA%.safetensors}.json" "$CHECKPOINT" "$TOKENIZER" "$PROMPT" "$SPLIT"; do
    if [[ ! -e "$required" ]]; then
        printf 'missing required artifact: %s\n' "$required" >&2
        exit 1
    fi
done

stage=gpu-lock
acquire_gpu_lock cosmos-t2-linear-trunk-smolexpert

build_cache() {
    local episodes="$1"
    local stride="$2"
    local output="$3"
    local mode=--overwrite
    if [[ -f "$output/manifest.json" ]]; then
        mode=--resume
    fi
    printf "[%s] cache episodes=%s stride=%s out=%s mode=%s merge=%s lora=%s blocks=0-19\n" \
        "$(date --iso-8601=seconds)" "$episodes" "$stride" "$output" "$mode" "$OLD_LORA" "$NEW_LORA"
    "$PYTHON" -m scripts.video_vam.build_cosmos_feature_cache \
        --root "$DATASET_ROOT" \
        --episodes $episodes \
        --stride "$stride" \
        --prompt "$PROMPT" \
        --checkpoint "$CHECKPOINT" \
        --merge-lora-weights "$OLD_LORA" \
        --lora-weights "$NEW_LORA" \
        --lora-blocks 0-19 \
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
    --run-name "cosmos-t2-linear-trunk-smolexpert-$DATE_TAG"

date --iso-8601=seconds > "$RUN_ROOT/COMPLETE"
printf "[%s] COMPLETE\n" "$(date --iso-8601=seconds)"
