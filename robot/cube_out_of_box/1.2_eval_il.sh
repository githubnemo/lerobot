#!/usr/bin/env bash
# Evaluate a trained policy on hardware using the current lerobot-rollout CLI.
# This script is intentionally blocked unless --allow-robot is supplied.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/home/anton/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export DISPLAY="${DISPLAY:-:0.0}"

TASK_NAME="take cube out of box"
POLICY_TYPE="smolvla"
POLICY_PATH=""
NUM_EPISODES=10
EPISODE_TIME_S=30
RESET_TIME_S=5
FPS=10
DEVICE="cuda"
ROBOT_PORT="/dev/ttyACM1"
ROBOT_ID="shabby"
CAMERA_INDEX=0
EVAL_DATASET_ROOT="${CUBE_EVAL_DATASET_ROOT:-$REPO_ROOT/data/eval_cube_out_of_box_il}"
USE_LEADER=false
LEADER_PORT="/dev/ttyACM0"
N_ACTION_STEPS=10
ALLOW_ROBOT=false
OVERWRITE=false

usage() {
    echo "Usage: $0 --allow-robot [--policy-path PATH] [--policy-type act|smolvla]"
    echo "       [--num-episodes N] [--episode-time SECONDS] [--fps N]"
    echo "       [--device cuda|cpu] [--robot-port PORT] [--leader-port PORT]"
    echo "       [--dataset-root PATH] [--overwrite]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --allow-robot) ALLOW_ROBOT=true ;;
        --policy-path) shift; POLICY_PATH="$1" ;;
        --policy-type) shift; POLICY_TYPE="$1" ;;
        --num-episodes) shift; NUM_EPISODES="$1" ;;
        --episode-time) shift; EPISODE_TIME_S="$1" ;;
        --fps) shift; FPS="$1" ;;
        --device) shift; DEVICE="$1" ;;
        --robot-port) shift; ROBOT_PORT="$1" ;;
        --leader-port) shift; LEADER_PORT="$1"; USE_LEADER=true ;;
        --use-leader) USE_LEADER=true ;;
        --n-action-steps) shift; N_ACTION_STEPS="$1" ;;
        --dataset-root) shift; EVAL_DATASET_ROOT="$1" ;;
        --overwrite) OVERWRITE=true ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
    shift
done

case "$POLICY_TYPE" in
    act|smolvla) ;;
    *) echo "Unsupported policy type: $POLICY_TYPE" >&2; exit 2 ;;
esac

if [ "$ALLOW_ROBOT" != true ]; then
    echo "Refusing physical rollout. This command can move the robot." >&2
    echo "Inspect the policy and hardware first, then re-run with --allow-robot." >&2
    exit 2
fi

if [ -z "$POLICY_PATH" ]; then
    POLICY_PATH="$REPO_ROOT/outputs/train/cube_out_of_box_il_${POLICY_TYPE}_subset_all/checkpoints/last/pretrained_model"
fi
if [ ! -f "$POLICY_PATH/config.json" ]; then
    echo "Policy config not found at $POLICY_PATH/config.json" >&2
    exit 2
fi

if [ -e "$EVAL_DATASET_ROOT" ]; then
    if [ "$OVERWRITE" = true ]; then
        rm -rf "$EVAL_DATASET_ROOT"
    else
        echo "Evaluation output already exists: $EVAL_DATASET_ROOT" >&2
        echo "Choose --dataset-root or pass --overwrite explicitly." >&2
        exit 2
    fi
fi

TELEOP_ARGS=()
if [ "$USE_LEADER" = true ]; then
    TELEOP_ARGS+=("--teleop.type=so101_leader" "--teleop.port=$LEADER_PORT" "--teleop.id=$ROBOT_ID")
fi

echo "PHYSICAL ROLLOUT ENABLED: $POLICY_PATH"
echo "Robot: $ROBOT_ID on $ROBOT_PORT; saved data: $EVAL_DATASET_ROOT"

lerobot-rollout \
    --strategy.type=episodic \
    --policy.path="$POLICY_PATH" \
    --policy.n_action_steps="$N_ACTION_STEPS" \
    --policy.device="$DEVICE" \
    --robot.type=so101_follower \
    --robot.port="$ROBOT_PORT" \
    --robot.id="$ROBOT_ID" \
    --robot.cameras="{front: {type: opencv, index_or_path: $CAMERA_INDEX, width: 640, height: 480, fps: 30, rotation: ROTATE_180}}" \
    "${TELEOP_ARGS[@]}" \
    --dataset.repo_id="hubnemo/eval_cube_out_of_box_il" \
    --dataset.root="$EVAL_DATASET_ROOT" \
    --dataset.single_task="$TASK_NAME" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.fps="$FPS" \
    --dataset.episode_time_s="$EPISODE_TIME_S" \
    --dataset.reset_time_s="$RESET_TIME_S" \
    --dataset.push_to_hub=false \
    --fps="$FPS" \
    --task="$TASK_NAME" \
    --display_data=true \
    --play_sounds=false \
    --return_to_initial_position=true
