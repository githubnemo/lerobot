#!/usr/bin/env bash
# Teleop recording for cube_out_of_box v2 or stack_cubes.
#
# First session on a dataset creates it. Later sessions (other robot, next day)
# auto-resume into the same local folder. ESC finishes the session early;
# num_episodes is "this session", not the dataset total.
#
#   ./robot/record_teleop.sh --task cube_v2 --setup boxed --num-episodes 20
#   # unplug, move to the other arm
#   ./robot/record_teleop.sh --task cube_v2 --setup free --num-episodes 20
#
#   ./robot/record_teleop.sh --task stack_cubes --setup boxed --num-episodes 20
#   ./robot/record_teleop.sh --task stack_cubes --setup free --num-episodes 20
#
# Cube v2 new episodes only live in data/cube_out_of_box_v2. Merge v1 afterwards:
#   ./robot/cube_out_of_box/2.2_merge_v1_v2.sh
#
# Keys: → next episode, ← redo, ESC stop.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=hw.sh
source "$SCRIPT_DIR/hw.sh"
cd "$REPO_ROOT"

HF_USER="${HF_USER:-hubnemo}"
TASK=""
SETUP=""
NUM_EPISODES=20
EPISODE_TIME_S=30
RESET_TIME_S=10
FPS=10
CAMERA_FPS=30
CAMERA_ROTATION="ROTATE_180"
CAMERA_INDEX_OVERRIDE=""
FOLLOWER_PORT_OVERRIDE=""
LEADER_PORT_OVERRIDE=""
PUSH=false
DISPLAY_DATA=true

usage() {
    cat <<EOF
Usage: $0 --task cube_v2|stack_cubes --setup boxed|free [options]

  --task cube_v2|stack_cubes   Which dataset to write
  --setup boxed|free           Which physical arm (see robot/hw.sh)
  --num-episodes N             Episodes this session (default: 20)
  --episode-time SECONDS       Recording length per episode (default: 30)
  --reset-time SECONDS         Pause between episodes (default: 10)
  --fps N                      Dataset / control FPS (default: 10, match cube v1)
  --camera-index N             OpenCV camera index
  --camera-rotation VALUE      ROTATE_180 (default) or NO_ROTATION
  --follower-port PATH         Override serial port
  --leader-port PATH           Override leader serial port
  --push                       Upload to the Hub when the session ends
  --no-display                 Do not open the Rerun viewer

Environment: HF_USER (default hubnemo). Ports/ids: see robot/hw.sh.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --task) shift; TASK="$1" ;;
        --setup) shift; SETUP="$1" ;;
        --num-episodes) shift; NUM_EPISODES="$1" ;;
        --episode-time) shift; EPISODE_TIME_S="$1" ;;
        --reset-time) shift; RESET_TIME_S="$1" ;;
        --fps) shift; FPS="$1" ;;
        --camera-index) shift; CAMERA_INDEX_OVERRIDE="$1" ;;
        --camera-rotation) shift; CAMERA_ROTATION="$1" ;;
        --follower-port) shift; FOLLOWER_PORT_OVERRIDE="$1" ;;
        --leader-port) shift; LEADER_PORT_OVERRIDE="$1" ;;
        --push) PUSH=true ;;
        --no-display) DISPLAY_DATA=false ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
    shift
done

if [ -z "$TASK" ] || [ -z "$SETUP" ]; then
    usage
    exit 2
fi

load_setup "$SETUP"

if [ -n "$CAMERA_INDEX_OVERRIDE" ]; then
    CAMERA_INDEX="$CAMERA_INDEX_OVERRIDE"
fi
if [ -n "$FOLLOWER_PORT_OVERRIDE" ]; then
    FOLLOWER_PORT="$FOLLOWER_PORT_OVERRIDE"
fi
if [ -n "$LEADER_PORT_OVERRIDE" ]; then
    LEADER_PORT="$LEADER_PORT_OVERRIDE"
fi

case "$TASK" in
    cube_v2)
        REPO_ID="${HF_USER}/cube_out_of_box_v2"
        DATASET_ROOT="${CUBE_V2_ROOT:-$REPO_ROOT/data/cube_out_of_box_v2}"
        SINGLE_TASK="take cube out of box"
        ;;
    stack_cubes)
        REPO_ID="${HF_USER}/stack_cubes"
        DATASET_ROOT="${STACK_CUBES_ROOT:-$REPO_ROOT/data/stack_cubes}"
        SINGLE_TASK="stack one cube on top of another"
        ;;
    *)
        echo "Unknown task '$TASK'. Use cube_v2 or stack_cubes." >&2
        usage
        exit 2
        ;;
esac

if [ ! -e "$FOLLOWER_PORT" ]; then
    echo "Follower port not found: $FOLLOWER_PORT" >&2
    echo "Pass --follower-port or set the port in robot/hw.sh." >&2
    exit 2
fi
if [ ! -e "$LEADER_PORT" ]; then
    echo "Leader port not found: $LEADER_PORT" >&2
    echo "Pass --leader-port or set the port in robot/hw.sh." >&2
    exit 2
fi

RESUME=false
if [ -d "$DATASET_ROOT/meta" ]; then
    RESUME=true
fi

CAMERAS="{front: {type: opencv, index_or_path: ${CAMERA_INDEX}, width: 640, height: 480, fps: ${CAMERA_FPS}, rotation: ${CAMERA_ROTATION}}}"

echo "Task:           $TASK  ($SINGLE_TASK)"
echo "Setup:          $SETUP  follower=$FOLLOWER_ID  leader=$LEADER_ID"
echo "Follower port:  $FOLLOWER_PORT"
echo "Leader port:    $LEADER_PORT"
echo "Camera:         index=$CAMERA_INDEX  rotation=$CAMERA_ROTATION  capture=${CAMERA_FPS} fps"
echo "Dataset:        $REPO_ID"
echo "Local root:     $DATASET_ROOT"
echo "This session:   $NUM_EPISODES episodes × ${EPISODE_TIME_S}s @ ${FPS} Hz"
if [ "$RESUME" = true ]; then
    echo "Mode:           RESUME (append to existing dataset)"
else
    echo "Mode:           CREATE (first session for this dataset)"
fi
echo "Keys:           → next   ← redo   ESC stop"
echo

uv run lerobot-record \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id="$FOLLOWER_ID" \
    --robot.use_degrees=true \
    --robot.cameras="$CAMERAS" \
    --teleop.type=so101_leader \
    --teleop.port="$LEADER_PORT" \
    --teleop.id="$LEADER_ID" \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.single_task="$SINGLE_TASK" \
    --dataset.fps="$FPS" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.episode_time_s="$EPISODE_TIME_S" \
    --dataset.reset_time_s="$RESET_TIME_S" \
    --dataset.no_stamp=true \
    --dataset.push_to_hub="$PUSH" \
    --resume="$RESUME" \
    --display_data="$DISPLAY_DATA" \
    --play_sounds=true
