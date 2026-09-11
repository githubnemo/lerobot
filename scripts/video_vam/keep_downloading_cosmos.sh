#!/usr/bin/env bash
DEST_DIR="/home/anton/lerobot-video-vam/outputs/models/cosmos2b/video_backbone"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

URL="https://huggingface.co/jonpai/mimic-video/resolve/main/video_backbone/v2w_pretrained_cosmos.pt"
TARGET_SIZE=3913017214

while true; do
  CURRENT_SIZE=0
  if [ -f "v2w_pretrained_cosmos.pt" ]; then
    CURRENT_SIZE=$(stat -c%s "v2w_pretrained_cosmos.pt")
  fi
  if [ "$CURRENT_SIZE" -ge "$TARGET_SIZE" ]; then
    echo "[$(date)] Download complete! ($CURRENT_SIZE bytes)"
    break
  fi
  echo "[$(date)] Resuming download from $CURRENT_SIZE / $TARGET_SIZE bytes..."
  curl -L -C - --connect-timeout 10 --max-time 300 -o "v2w_pretrained_cosmos.pt" "$URL" || true
  sleep 2
done

echo "[$(date)] Downloading tokenizer.pth..."
curl -L -C - --connect-timeout 10 -o "tokenizer.pth" "https://huggingface.co/jonpai/mimic-video/resolve/main/video_backbone/tokenizer/tokenizer.pth" || true

# Symlink setup
LEGACY_DIR="/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone"
mkdir -p "$LEGACY_DIR"
ln -sf "$DEST_DIR/v2w_pretrained_cosmos.pt" "$LEGACY_DIR/v2w_pretrained_cosmos.pt"
ln -sf "$DEST_DIR/tokenizer.pth" "$LEGACY_DIR/tokenizer.pth" 2>/dev/null || true

echo "[$(date)] All files downloaded and verified!"
