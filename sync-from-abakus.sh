#!/usr/bin/env bash
set -euo pipefail

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Syncing code, scripts, and logs from abakus -> $DEST..."

rsync -avz \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.safetensors' \
    --exclude '*.pt' \
    --exclude '*.pth' \
    --exclude '*.png' \
    --exclude '*.mp4' \
    abakus:/home/anton/lerobot-video-vam/ "$DEST/"

echo "Sync completed successfully!"
