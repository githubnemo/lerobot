#!/bin/bash
# Step 7: Residual RL fine-tuning with minimal_rl_resfit.py
# Uses frozen SmolVLA base + reward classifier + residual actor.
# Replaces steps 5/6 (full distributed HIL-SERL) with the single-process ResFit script.

set -e

export HF_HOME="/home/nemo/.cache/pysandbox-lerobot/huggingface"
export DISPLAY=:0.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TASK_NAME="cube_out_of_box"
SMOLVLA_PATH="outputs/train/${TASK_NAME}_il_smolvla/checkpoints/last/pretrained_model"
CLASSIFIER_PATH="outputs/reward_classifier/${TASK_NAME}/checkpoints/last/pretrained_model"
DATASET_ROOT="data/${TASK_NAME}_dataset"

# Defaults
RESIDUAL_SCALE=0.2
CHUNK_SIZE=5
NO_WANDB_FLAG=""
NO_PRELOAD_FLAG=""
PURE_BASE_FLAG=""
ROBOT_PORT="/dev/ttyACM1"
TASK_PROMPT="take cube out of box"

while [ "$#" -gt 0 ]; do
    case $1 in
        --smolvla-path)
            shift; SMOLVLA_PATH="$1" ;;
        --classifier-path)
            shift; CLASSIFIER_PATH="$1" ;;
        --dataset-root)
            shift; DATASET_ROOT="$1" ;;
        --residual-scale)
            shift; RESIDUAL_SCALE="$1" ;;
        --chunk-size)
            shift; CHUNK_SIZE="$1" ;;
        --port)
            shift; ROBOT_PORT="$1" ;;
        --task)
            shift; TASK_PROMPT="$1" ;;
        --no-wandb)
            NO_WANDB_FLAG="--no-wandb" ;;
        --no-preload)
            NO_PRELOAD_FLAG="--no-preload" ;;
        --pure-base)
            PURE_BASE_FLAG="--pure-base" ;;
        *)
            echo "Usage: $0 [--smolvla-path PATH|HF_ID] [--classifier-path PATH] [--dataset-root PATH]"
            echo "          [--residual-scale 0.2] [--chunk-size 5] [--port /dev/ttyACM0]"
            echo "          [--no-wandb] [--no-preload] [--pure-base]"
            echo ""
            echo "  --pure-base: run ONLY the SmolVLA base (no residual, no training)."
            echo "               Useful to sanity-check the IL policy through this pipeline."
            exit 1 ;;
    esac
    shift
done

# Only verify local files if paths look like local paths (contain "/" but no HF ID format user/repo)
check_path() {
    local p="$1"; local name="$2"
    # If it's a local-looking path (starts with ./ or /), verify config.json exists
    if [[ "$p" == /* || "$p" == ./* ]]; then
        if [ ! -f "$p/config.json" ]; then
            echo "ERROR: $name checkpoint missing at $p (config.json not found)"
            exit 1
        fi
    fi
    # HF IDs (user/repo or user/repo/... ) will be validated when loading
}
check_path "$SMOLVLA_PATH" "SmolVLA"
check_path "$CLASSIFIER_PATH" "Reward classifier"

if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATASET_ROOT"
    exit 1
fi

echo "=============================================="
echo "    CUBE OUT OF BOX - ResFit RL"
echo "=============================================="
echo "  SmolVLA base:      $SMOLVLA_PATH"
echo "  Reward classifier: $CLASSIFIER_PATH"
echo "  Dataset (demos):   $DATASET_ROOT"
echo "  Residual scale:    ±${RESIDUAL_SCALE} of action range"
echo "  VLA chunk size:    $CHUNK_SIZE (1 VLA forward per $CHUNK_SIZE env steps)"
echo "  Robot port:        $ROBOT_PORT"
if [ -n "$PURE_BASE_FLAG" ]; then
    echo "  Mode:              PURE-BASE (residual disabled, no training)"
fi
echo ""

python robot/minimal_rl_resfit.py \
    --smolvla-path "$SMOLVLA_PATH" \
    --classifier-path "$CLASSIFIER_PATH" \
    --dataset-root "$DATASET_ROOT" \
    --residual-scale "$RESIDUAL_SCALE" \
    --chunk-size "$CHUNK_SIZE" \
    --port "$ROBOT_PORT" \
    --task "$TASK_PROMPT" \
    $NO_WANDB_FLAG $NO_PRELOAD_FLAG $PURE_BASE_FLAG
