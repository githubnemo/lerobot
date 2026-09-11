#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="/home/anton/lerobot-video-vam/outputs/models/cosmos2b/video_backbone"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo "Downloading v2w_pretrained_cosmos.pt (3.9 GB) via curl with resume support..."
curl -L -C - --retry 10 --retry-delay 3 --max-time 1800 \
  -o "v2w_pretrained_cosmos.pt" \
  "https://huggingface.co/jonpai/mimic-video/resolve/main/video_backbone/v2w_pretrained_cosmos.pt"

echo "Downloading tokenizer.pth..."
curl -L -C - --retry 10 --retry-delay 3 \
  -o "tokenizer.pth" \
  "https://huggingface.co/jonpai/mimic-video/resolve/main/video_backbone/tokenizer/tokenizer.pth"

# Legacy symlink setup
LEGACY_DIR="/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone"
mkdir -p "$LEGACY_DIR"
ln -sf "$DEST_DIR/v2w_pretrained_cosmos.pt" "$LEGACY_DIR/v2w_pretrained_cosmos.pt"
ln -sf "$DEST_DIR/tokenizer.pth" "$LEGACY_DIR/tokenizer.pth" 2>/dev/null || true

echo "ALL DOWNLOADS FINISHED SUCCESSFULLY!"
