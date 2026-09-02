#!/usr/bin/env bash
# Smoke-first LTX-2.5 observed-only (state_t=2) unpooled SmolExpert.
set -Eeuo pipefail

if [[ "${1:-}" == "--smoke-only" ]]; then
    SMOKE_ONLY=1
    shift
else
    SMOKE_ONLY=0
fi
if (( $# != 0 )); then
    printf 'usage: %s [--smoke-only]\n' "$0" >&2
    exit 64
fi

REPO=/home/anton/lerobot-video-vam
PYTHON="$REPO/.venv/bin/python"
CACHE=/home/anton/.cache/video-vam
RUN_ROOT="$CACHE/runs/ltx25-statet2-unpooled-smolexpert-20260829"
TRAIN_CACHE="$CACHE/ltx25-train0-31-stride3-statet2-unpooled"
VAL_CACHE="$CACHE/ltx25-val32-39-stride20-statet2-unpooled"
TRAIN_EPISODE_ARGS=(--train-episodes {0..31})
VAL_EPISODE_ARGS=(--val-episodes {32..39})
if (( SMOKE_ONLY == 1 )); then
    TRAIN_CACHE="$CACHE/ltx25-smoke-ep0-statet2-unpooled"
    VAL_CACHE="$CACHE/ltx25-smoke-val32-39-statet2-unpooled"
    TRAIN_EPISODE_ARGS=(--train-episodes 0)
    VAL_EPISODE_ARGS=(--val-episodes {32..39})
fi
SPLIT="$CACHE/splits/rehearsal-stride20.json"

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
    printf '[%s] queue already complete; exiting\n' "$(date --iso-8601=seconds)"
    exit 0
fi
rm -f -- "$RUN_ROOT/FAILED"

stage=gpu-lock
acquire_gpu_lock ltx-statet2-unpooled

stage=cache
"$PYTHON" -m scripts.video_vam.build_ltx_feature_cache \
    --train-output-dir "$TRAIN_CACHE" \
    --val-output-dir "$VAL_CACHE" \
    --train-stride 3 \
    --val-stride 20 \
    --state-t 2 \
    --context-transform none \
    --min-free-gib 20 \
    --seed 0 \
    "${TRAIN_EPISODE_ARGS[@]}" \
    "${VAL_EPISODE_ARGS[@]}" \
    --overwrite
touch "$RUN_ROOT/CACHE_COMPLETE"

stage=smoke
"$PYTHON" -m scripts.video_vam.train_smolexpert_on_cosmos \
    --manifest "$TRAIN_CACHE/manifest.json" \
    --val-manifest "$VAL_CACHE/manifest.json" \
    --split "$SPLIT" \
    --train-episodes 0 \
    --output-dir "$RUN_ROOT/smoke" \
    --context-transform none \
    --batch-size 2 \
    --max-steps 2 \
    --val-every 2 \
    --patience 1 \
    --min-delta 0 \
    --warmup-steps 1 \
    --num-steps 2 \
    --no-wandb \
    --no-save-checkpoints \
    --overwrite
touch "$RUN_ROOT/SMOKE_OK"
if (( SMOKE_ONLY == 1 )); then
    printf '[%s] smoke-only requested; stopping before overnight train\n' "$(date --iso-8601=seconds)"
    touch "$RUN_ROOT/SMOKE_ONLY_COMPLETE"
    exit 0
fi

stage=full-train
"$PYTHON" -m scripts.video_vam.train_smolexpert_on_cosmos \
    --manifest "$TRAIN_CACHE/manifest.json" \
    --val-manifest "$VAL_CACHE/manifest.json" \
    --split "$SPLIT" \
    --output-dir "$RUN_ROOT/full" \
    --context-transform none \
    --batch-size 8 \
    --max-steps 500000 \
    --max-hours 12 \
    --val-every 1000 \
    --patience 10 \
    --min-delta 0.02 \
    --lr 1e-4 \
    --weight-decay 1e-10 \
    --grad-clip 10 \
    --warmup-steps 1000 \
    --num-steps 10 \
    --seed 0 \
    --wandb-project video-vam-world2action \
    --run-name ltx25-statet2-unpooled-smolexpert-20260829 \
    --overwrite

touch "$RUN_ROOT/COMPLETE"
printf '[%s] COMPLETE\n' "$(date --iso-8601=seconds)"
