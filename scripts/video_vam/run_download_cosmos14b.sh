#!/usr/bin/env bash
# Background downloader for Cosmos-1.0-Diffusion-14B weights

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/download_cosmos14b.log"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

echo "[$(date --iso-8601=seconds)] Starting Cosmos 14B download -> logging to $LOG_FILE"

uv run python -u -m scripts.video_vam.download_cosmos14b "$@" >> "$LOG_FILE" 2>&1
