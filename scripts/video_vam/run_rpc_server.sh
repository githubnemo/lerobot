#!/usr/bin/env bash
# Universal Model Server Wrapper for SmolVLA and Video-VAM on abakus
# Concurrent inference is allowed: intentionally do not acquire the training GPU lock.
set -euo pipefail

REPO=/home/anton/lerobot-video-vam
cd "$REPO"
source scripts/video_vam/cosmos_cuda_env.sh

DEFAULT_SMOLVLA_CHECKPOINT="$REPO/outputs/train/cube_out_of_box_il_smolvla_train_only_stats_0_31_20260826_1hr/checkpoints/029200/pretrained_model"
DEFAULT_PORT=8766

if [[ $# -gt 0 ]]; then
    printf "[%s] Launching rpc_server.py with custom arguments: %s\n" "$(date --iso-8601=seconds)" "$*" >&2
    exec "$VAM_VENV/bin/python" scripts/video_vam/rpc_server.py "$@"
else
    printf "[%s] Launching default SmolVLA model server on port %s\n" "$(date --iso-8601=seconds)" "$DEFAULT_PORT"
    exec "$VAM_VENV/bin/python" scripts/video_vam/rpc_server.py \
        --policy smolvla \
        --checkpoint "$DEFAULT_SMOLVLA_CHECKPOINT" \
        --port "$DEFAULT_PORT" \
        --device cuda
fi
