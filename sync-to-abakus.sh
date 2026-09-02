#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Syncing local code -> abakus:/home/anton/lerobot-video-vam/..."

rsync -avz \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.safetensors' \
    --exclude '*.pt' \
    --exclude '*.pth' \
    --exclude '*.png' \
    --exclude '*.mp4' \
    --exclude '.git' \
    "$SRC/" abakus:/home/anton/lerobot-video-vam/

echo "Sync to abakus completed successfully!"
