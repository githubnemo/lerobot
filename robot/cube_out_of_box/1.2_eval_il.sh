#!/bin/bash
# Step 1.2: Evaluate an Imitation Learning policy on the real robot
# Runs the trained policy and records evaluation episodes

set -e

export HF_HOME="/home/nemo/.cache/pysandbox-lerobot/huggingface"
export DISPLAY=:0.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_NAME="cube_out_of_box"

# Defaults (must match 1.1_train_il.sh)
POLICY_TYPE="smolvla"
POLICY_PATH=""
NUM_EPISODES=10
EPISODE_TIME_S=30
RESET_TIME_S=5
FPS=10
DISPLAY_DATA=true

while [ "$#" -gt 0 ]; do
    case $1 in
        --policy-path)
            shift; POLICY_PATH="$1" ;;
        --policy-type)
            shift; POLICY_TYPE="$1" ;;
        --num-episodes)
            shift; NUM_EPISODES="$1" ;;
        --episode-time)
            shift; EPISODE_TIME_S="$1" ;;
        --fps)
            shift; FPS="$1" ;;
        *)
            echo "Usage: $0 [--policy-path PATH] [--policy-type act|smolvla] [--num-episodes N] [--episode-time S] [--fps N]"
            exit 1 ;;
    esac
    shift
done

# Auto-detect policy path if not specified
if [ -z "$POLICY_PATH" ]; then
    OUTPUT_DIR="outputs/train/${TASK_NAME}_il_${POLICY_TYPE}"
    POLICY_PATH="$OUTPUT_DIR/checkpoints/last/pretrained_model"
fi

# Verify policy exists
if [ ! -f "$POLICY_PATH/config.json" ]; then
    echo "ERROR: No policy found at $POLICY_PATH"
    echo ""
    echo "Either:"
    echo "  1. Run ./1.1_train_il.sh first to train a policy"
    echo "  2. Specify a path: ./1.2_eval_il.sh --policy-path /path/to/checkpoint"
    exit 1
fi

EVAL_DATASET_ROOT="./data/eval_${TASK_NAME}_il"

echo "=============================================="
echo "    CUBE OUT OF BOX - IL EVALUATION"
echo "=============================================="
echo ""
echo "  Policy:       $POLICY_PATH"
echo "  Episodes:     $NUM_EPISODES"
echo "  Episode time: ${EPISODE_TIME_S}s"
echo "  FPS:          $FPS"
echo ""
echo "=== CONTROLS ==="
echo "  The policy runs autonomously."
echo "  Right Arrow:   Save episode & move to next"
echo "  Left Arrow:    Discard episode & re-record"
echo "  Escape:        Stop evaluation"
echo "================"
echo ""

# Clean previous eval dataset if exists
rm -rf "$EVAL_DATASET_ROOT" 2>/dev/null || true

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=shabby \
    --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, rotation: ROTATE_180}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=shabby \
    --dataset.repo_id="hubnemo/eval_${TASK_NAME}_il" \
    --dataset.root="$EVAL_DATASET_ROOT" \
    --dataset.single_task="take cube out of box" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.fps="$FPS" \
    --dataset.episode_time_s="$EPISODE_TIME_S" \
    --dataset.reset_time_s="$RESET_TIME_S" \
    --display_data="$DISPLAY_DATA" \
    --play_sounds=false \
    --policy.path="$POLICY_PATH" \
    --policy.n_action_steps=10

echo ""
echo "=============================================="
echo "    EVALUATION COMPLETE"
echo "=============================================="
echo ""
echo "Evaluation episodes saved to: $EVAL_DATASET_ROOT"
echo "Review the recordings to assess policy quality."
