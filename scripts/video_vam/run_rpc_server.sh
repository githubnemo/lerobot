#!/usr/bin/env bash
set -euo pipefail

cd -- /home/anton/lerobot-video-vam
source scripts/video_vam/gpu_lock.sh
source scripts/video_vam/cosmos_cuda_env.sh

printf '%s\n' 'WARNING: joint limits below are placeholders, not this arm'
printf '%s\n' 'WARNING: use the arm-specific calibrated limits before hardware motion'
acquire_gpu_lock video-vam-rpc
# torch.compile is on by default; pass --no-compile for eager DiT.
exec "$VAM_VENV/bin/python" scripts/video_vam/rpc_server.py \
  --policy video_vam \
  --checkpoint /home/anton/.cache/video-vam/runs/cosmos2b-videolora-smolexpert-20260828/smolexpert \
  --joint-limits-min -180 -180 -180 -180 -180 0 \
  --joint-limits-max 180 180 180 180 180 100
