#!/usr/bin/env bash
set -u
export HF_HUB_DISABLE_XET=1
LOG=/home/anton/lerobot-video-vam/smolvla-model-prefetch-retry.log
exec > >(tee -a "$LOG") 2>&1
while true; do
  echo "PREFETCH_ATTEMPT $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if /home/anton/lerobot-video-vam/.venv/bin/python -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download("lerobot/smolvla_base", "model.safetensors"))'; then
    echo "PREFETCH_SUCCESS $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 0
  fi
  echo "PREFETCH_RETRY $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 15
done
