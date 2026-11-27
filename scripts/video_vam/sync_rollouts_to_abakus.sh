#!/usr/bin/env bash
# Sync all rollout debug traces + online EXPO checkpoints from sahni -> abakus (via Mac relay).
# Usage: bash scripts/video_vam/sync_rollouts_to_abakus.sh [SUBDIR_NAME]
set -Eeuo pipefail

SUBDIR="${1:-$(date +%Y-%m-%d)_shabby_rollouts}"
REMOTE_DEST="/home/anton/lerobot-video-vam/outputs/rollout_debug/${SUBDIR}"

echo "[SYNC] Streaming sahni rollout_debug + expo_online -> abakus:${REMOTE_DEST}"
ssh -o ControlMaster=no -o ControlPath=none nemo@sahni \
  'cd ~/code/hf/lerobot/githubnemo/outputs && tar czf - rollout_debug $( [ -d expo_online ] && echo expo_online )' \
| ssh -o ControlMaster=no -o ControlPath=none abakus \
  "mkdir -p '${REMOTE_DEST}' && tar xzf - -C '${REMOTE_DEST}' && du -sh '${REMOTE_DEST}'/*"
echo "[SYNC] Done -> abakus:${REMOTE_DEST}"
