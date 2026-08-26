#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/anton/lerobot-video-vam
CACHE=/home/anton/.cache/video-vam
PYTHON="$REPO/.venv/bin/python"
RUN_ROOT="$CACHE/runs/vam-rollout-validation-20260826"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/queue.log") 2>&1
stage=initializing
trap 'status=$?; set +e; printf "[%s] FAILED stage=%s status=%s\n" "$(date --iso-8601=seconds)" "$stage" "$status"; printf "%s\n" "$stage" > "$RUN_ROOT/FAILED"; exit "$status"' ERR

rm -f -- "$RUN_ROOT/FAILED"
cd "$REPO"
source scripts/video_vam/cosmos_cuda_env.sh
source scripts/video_vam/gpu_lock.sh

stage=acquire-gpu-lock
acquire_gpu_lock wrapper-validation-20260826

for backend in cosmos ltx; do
    stage="validate-$backend"
    printf "[%s] starting %s\n" "$(date --iso-8601=seconds)" "$stage"
    "$PYTHON" -m scripts.video_vam.validate_video_vam_rollout --backend "$backend"
    printf "[%s] %s ok\n" "$(date --iso-8601=seconds)" "$stage"
done

touch "$RUN_ROOT/COMPLETE"
printf "[%s] VALIDATION_QUEUE_COMPLETE\n" "$(date --iso-8601=seconds)"
