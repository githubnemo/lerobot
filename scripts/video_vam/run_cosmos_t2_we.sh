#!/usr/bin/env bash
# Train the state_t=2 WorldExpert and preview both pinned anchors.
set -Eeuo pipefail

REPO=/home/anton/lerobot-video-vam
PYTHON="$REPO/.venv/bin/python"
CACHE=/home/anton/.cache/video-vam
RUN_ROOT="${RUN_ROOT:-$CACHE/runs/cosmos-t2-we-$(date +%Y%m%d)}"
TRAIN_RUN="$RUN_ROOT/training"
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
# shellcheck source=/dev/null
source scripts/video_vam/gpu_lock.sh
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/queue.log") 2>&1
stage=initializing
trap 'status=$?; set +e; printf "[%s] FAILED stage=%s status=%s\n" "$(date --iso-8601=seconds)" "$stage" "$status"; printf "%s\n" "$stage" > "$RUN_ROOT/FAILED"; exit "$status"' ERR

if [[ -f "$RUN_ROOT/DONE" ]]; then
    printf "[%s] queue already complete; exiting\n" "$(date --iso-8601=seconds)"
    exit 0
fi
rm -f -- "$RUN_ROOT/FAILED"

for required in "$TRAIN_MANIFEST" "$VAL_MANIFEST" "$OLD_LORA" "$CHECKPOINT" "$TOKENIZER" "$PROMPT"; do
    if [[ ! -e "$required" ]]; then
        printf 'missing required artifact: %s\n' "$required" >&2
        exit 1
    fi
done

stage=gpu-lock
acquire_gpu_lock cosmos-t2-we

stage=train
"$PYTHON" -m scripts.video_vam.train_cosmos_t2_world_expert \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --dataset-root "$DATASET_ROOT" \
    --backbone-checkpoint "$CHECKPOINT" \
    --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" \
    --merge-lora-weights "$OLD_LORA" \
    --output-dir "$TRAIN_RUN" \
    --max-hours 12 \
    --wandb-project video-vam-world2action \
    --run-name "cosmos-t2-we-$(date +%Y%m%d)"

stage=preview-anchor-episode-0
"$PYTHON" -m scripts.video_vam.preview_cosmos_t2_we_video_prediction \
    --run-dir "$TRAIN_RUN" \
    --episode 0 \
    --frame-index 4 \
    --steps 35 \
    --seed 0 \
    --output-dir "$RUN_ROOT/previews" \
    --overwrite

stage=preview-anchor-episode-19
"$PYTHON" -m scripts.video_vam.preview_cosmos_t2_we_video_prediction \
    --run-dir "$TRAIN_RUN" \
    --episode 19 \
    --frame-index 2203 \
    --steps 35 \
    --seed 0 \
    --output-dir "$RUN_ROOT/previews" \
    --overwrite

# Optional cache rebuild after training, using the new adapter plus the merged old LoRA:
# "$PYTHON" -m scripts.video_vam.build_cosmos_feature_cache \
#     --root "$DATASET_ROOT" --episodes {0..31} --stride 3 \
#     --checkpoint "$CHECKPOINT" --tokenizer "$TOKENIZER" --prompt "$PROMPT" \
#     --merge-lora-weights "$OLD_LORA" --lora-weights "$TRAIN_RUN/best_lora.safetensors" \
#     --lora-blocks 0-19 --state-t 2 --sigma 80 --vae-input-mode observed_prefix \
#     --context-transform none \
#     --output-dir "$CACHE/cosmos-t2-we-train0-31-stride3-statet2-unpooled" --overwrite
# "$PYTHON" -m scripts.video_vam.build_cosmos_feature_cache \
#     --root "$DATASET_ROOT" --episodes {32..39} --stride 20 \
#     --checkpoint "$CHECKPOINT" --tokenizer "$TOKENIZER" --prompt "$PROMPT" \
#     --merge-lora-weights "$OLD_LORA" --lora-weights "$TRAIN_RUN/best_lora.safetensors" \
#     --lora-blocks 0-19 --state-t 2 --sigma 80 --vae-input-mode observed_prefix \
#     --context-transform none \
#     --output-dir "$CACHE/cosmos-t2-we-val32-39-stride20-statet2-unpooled" --overwrite

date --iso-8601=seconds > "$RUN_ROOT/DONE"
printf "[%s] DONE\n" "$(date --iso-8601=seconds)"
