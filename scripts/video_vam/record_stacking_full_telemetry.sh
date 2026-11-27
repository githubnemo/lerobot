#!/usr/bin/env bash
# Record SO-101 stacking demonstrations on sahni at the camera's maximum FPS (30 fps)
# with FULL Feetech STS3215 servo telemetry (pos, vel, load, current, voltage, temp, moving, status).
# Usage:
#   bash scripts/video_vam/record_stacking_full_telemetry.sh [REPO_ID] [NUM_EPISODES] [TASK]
set -Eeuo pipefail

REPO_ID="${1:-Orellius/so101_stack_cubes_full_telemetry}"
NUM_EPISODES="${2:-30}"
TASK="${3:-Stack the cubes on top of each other}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM1}"
FOLLOWER_ID="${FOLLOWER_ID:-shabby}"
LEADER_ID="${LEADER_ID:-shabby_leader}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
RECORD_FPS="${RECORD_FPS:-30}"
EPISODE_TIME_S="${EPISODE_TIME_S:-45}"
RESET_TIME_S="${RESET_TIME_S:-10}"

export HF_HOME="${HF_HOME:-/home/nemo/.cache/pysandbox-lerobot/huggingface}"
if [ -z "${HF_TOKEN:-}" ] && [ -s "$HOME/.cache/orellius_hf_token" ]; then
  export HF_TOKEN="$(cat "$HOME/.cache/orellius_hf_token")"
fi
PUSH_TO_HUB="${PUSH_TO_HUB:-$([ -n "${HF_TOKEN:-}" ] && echo true || echo false)}"
LEROBOT_BIN="${LEROBOT_BIN:-/home/nemo/envs/lerobot/bin/lerobot-record}"
command -v "$LEROBOT_BIN" >/dev/null 2>&1 || LEROBOT_BIN="lerobot-record"

echo "[RECORD] Repo: $REPO_ID | Episodes: $NUM_EPISODES | FPS: ${RECORD_FPS} (camera max)"
echo "[RECORD] Follower: $FOLLOWER_ID ($FOLLOWER_PORT) | Leader: $LEADER_ID ($LEADER_PORT)"
echo "[RECORD] Full STS3215 telemetry: pos + vel + load + current + voltage + temperature + moving + status"

exec "$LEROBOT_BIN" \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.num_read_retries=10 \
  --robot.record_all_telemetry=true \
  --robot.cameras="{front: {type: opencv, index_or_path: ${CAMERA_INDEX}, width: 640, height: 480, fps: ${RECORD_FPS}}}" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --teleop.num_read_retries=10 \
  --dataset.repo_id="$REPO_ID" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.fps="$RECORD_FPS" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s="$RESET_TIME_S" \
  --dataset.single_task="$TASK" \
  --dataset.push_to_hub="$PUSH_TO_HUB" \
  --display_data=false
