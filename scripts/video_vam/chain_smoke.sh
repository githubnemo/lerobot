#!/usr/bin/env bash
# Wait for the running probe's Python PID (never a tmux PID) to exit, then smoke-test.
set -uo pipefail
WAIT_PID="${1:?python pid to wait for}"
LOG=/home/anton/lerobot-video-vam/outputs/smoke/smoke.log
mkdir -p "$(dirname "$LOG")"
if ! ps -o cmd= -p "$WAIT_PID" | grep -q '^\.venv/bin/python'; then
  echo "refusing to wait on non-python pid $WAIT_PID" | tee -a "$LOG"
  exit 1
fi
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
echo "probe pid $WAIT_PID exited at $(date -Is)" | tee -a "$LOG"
sleep 30
bash /home/anton/lerobot-video-vam/scripts/video_vam/smoke_random_anchor.sh 2>&1 | tee -a "$LOG"
echo "SMOKE_EXIT=$?" | tee -a "$LOG"
touch /home/anton/lerobot-video-vam/outputs/smoke/SMOKE_COMPLETE
