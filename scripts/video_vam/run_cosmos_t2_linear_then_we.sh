#!/usr/bin/env bash
# Train the cheap linear readout first, then run the T=2 WorldExpert chain.
set -Eeuo pipefail

REPO=/home/anton/lerobot-video-vam
PYTHON="$REPO/.venv/bin/python"
CACHE=/home/anton/.cache/video-vam
DATE_TAG="$(date +%Y%m%d)"
CHAIN_ROOT="${CHAIN_ROOT:-$CACHE/runs/cosmos-t2-linear-then-we-$DATE_TAG}"
LINEAR_RUN="${LINEAR_RUN:-$CACHE/runs/cosmos-t2-linear-$DATE_TAG}"
TRAIN_MANIFEST="$CACHE/cosmos-videolora-train0-31-stride3-statet2-unpooled/manifest.json"
VAL_MANIFEST="$CACHE/cosmos-videolora-val32-39-stride20-statet2-unpooled/manifest.json"
OLD_LORA="$CACHE/runs/cosmos2b-video-lora-20260828/paused-step6000/best_lora.safetensors"
CHECKPOINT="$CACHE/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
TOKENIZER="$CACHE/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
PROMPT="$CACHE/prompt-embeddings/cube-out-of-box-t5-11b.safetensors"
DATASET_ROOT="$CACHE/cube-out-of-box-dataset"

cd "$REPO"
# shellcheck source=/dev/null
source scripts/video_vam/cosmos_cuda_env.sh
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$CHAIN_ROOT"
exec > >(tee -a "$CHAIN_ROOT/queue.log") 2>&1
stage=initializing
trap 'status=$?; set +e; printf "[%s] FAILED stage=%s status=%s\n" "$(date --iso-8601=seconds)" "$stage" "$status"; printf "%s\n" "$stage" > "$CHAIN_ROOT/FAILED"; exit "$status"' ERR

if [[ -f "$CHAIN_ROOT/DONE" ]]; then
    printf "[%s] queue already complete; exiting\n" "$(date --iso-8601=seconds)"
    exit 0
fi
rm -f -- "$CHAIN_ROOT/FAILED"

for required in "$TRAIN_MANIFEST" "$VAL_MANIFEST" "$OLD_LORA" "$CHECKPOINT" "$TOKENIZER" "$PROMPT"; do
    if [[ ! -e "$required" ]]; then
        printf 'missing required artifact: %s\n' "$required" >&2
        exit 1
    fi
done

stage=linear-readout
(
    set -Eeuo pipefail
    # shellcheck source=/dev/null
    source scripts/video_vam/gpu_lock.sh
    acquire_gpu_lock cosmos-t2-linear

    "$PYTHON" -m scripts.video_vam.train_cosmos_t2_world_expert \
        --train-manifest "$TRAIN_MANIFEST" \
        --val-manifest "$VAL_MANIFEST" \
        --dataset-root "$DATASET_ROOT" \
        --backbone-checkpoint "$CHECKPOINT" \
        --tokenizer "$TOKENIZER" \
        --prompt "$PROMPT" \
        --merge-lora-weights "$OLD_LORA" \
        --output-dir "$LINEAR_RUN" \
        --head linear \
        --max-hours 6 \
        --wandb-project video-vam-world2action \
        --run-name "cosmos-t2-linear-$DATE_TAG"

    "$PYTHON" -m scripts.video_vam.preview_cosmos_t2_we_video_prediction \
        --run-dir "$LINEAR_RUN" \
        --episode 0 \
        --frame-index 4 \
        --steps 1 \
        --seed 0 \
        --output-dir "$LINEAR_RUN/previews" \
        --overwrite

    "$PYTHON" -m scripts.video_vam.preview_cosmos_t2_we_video_prediction \
        --run-dir "$LINEAR_RUN" \
        --episode 19 \
        --frame-index 2203 \
        --steps 1 \
        --seed 0 \
        --output-dir "$LINEAR_RUN/previews" \
        --overwrite
)

stage=world-expert
exec bash "$REPO/scripts/video_vam/run_cosmos_t2_we.sh"
