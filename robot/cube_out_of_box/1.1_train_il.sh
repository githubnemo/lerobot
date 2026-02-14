#!/bin/bash
# Step 1.1: Train an Imitation Learning policy on the collected demonstrations
# Uses the dataset from step 1 to train ACT (or another IL policy)

set -e

export HF_HOME="/home/nemo/.cache/pysandbox-lerobot/huggingface"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_ROOT="./data/cube_out_of_box_dataset"
DATASET_REPO_ID="hubnemo/cube_out_of_box_dataset"
TASK_NAME="cube_out_of_box"

# Defaults
POLICY_TYPE="smolvla"
NUM_STEPS=5000
BATCH_SIZE=1
RESUME=false
WANDB=false
DEVICE="cuda"
LOG_FREQ=100
EVAL_FREQ=500
SAVE_FREQ=5000

# Pretrained model paths for fine-tuning (use --policy.path instead of --policy.type)
# Set to "" to train from scratch with --policy.type instead
SMOLVLA_PRETRAINED="lerobot/smolvla_base"

while [ "$#" -gt 0 ]; do
    case $1 in
        --policy-type)
            shift; POLICY_TYPE="$1" ;;
        --steps)
            shift; NUM_STEPS="$1" ;;
        --batch-size)
            shift; BATCH_SIZE="$1" ;;
        --resume)
            RESUME=true ;;
        --wandb)
            WANDB=true ;;
        --device)
            shift; DEVICE="$1" ;;
        --from-scratch)
            SMOLVLA_PRETRAINED="" ;;
        *)
            echo "Usage: $0 [--policy-type act|smolvla] [--steps N] [--batch-size N] [--resume] [--wandb] [--device cuda|cpu] [--from-scratch]"
            exit 1 ;;
    esac
    shift
done

OUTPUT_DIR="outputs/train/${TASK_NAME}_il_${POLICY_TYPE}"

# Determine whether to use --policy.path (pretrained) or --policy.type (from scratch)
POLICY_ARG=""
POLICY_LABEL=""
if [ "$POLICY_TYPE" = "smolvla" ] && [ -n "$SMOLVLA_PRETRAINED" ]; then
    POLICY_ARG="--policy.path=$SMOLVLA_PRETRAINED"
    POLICY_LABEL="$POLICY_TYPE (pretrained: $SMOLVLA_PRETRAINED)"
else
    POLICY_ARG="--policy.type=$POLICY_TYPE"
    POLICY_LABEL="$POLICY_TYPE (from scratch)"
fi

echo "=============================================="
echo "    CUBE OUT OF BOX - IMITATION LEARNING"
echo "=============================================="
echo ""
echo "  Policy:       $POLICY_LABEL"
echo "  Dataset:      $DATASET_ROOT"
echo "  Output:       $OUTPUT_DIR"
echo "  Steps:        $NUM_STEPS"
echo "  Batch size:   $BATCH_SIZE"
echo "  Device:       $DEVICE"
echo "  WandB:        $WANDB"
echo "  Resume:       $RESUME"
echo ""

# Verify dataset exists
if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATASET_ROOT"
    echo "Please run ./1_collect_data.sh first to collect demonstrations."
    exit 1
fi

if $RESUME; then
    CHECKPOINT_PATH="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "ERROR: No checkpoint found at $CHECKPOINT_PATH"
        echo "Cannot resume — train from scratch first."
        exit 1
    fi
    echo "Resuming training from last checkpoint..."
    lerobot-train \
        --config_path="$CHECKPOINT_PATH" \
        --resume=true \
        --steps="$NUM_STEPS"
else
    echo "Fine-tuning: $POLICY_LABEL"

    # When fine-tuning from a pretrained model, we must set input_features to null
    # so that features are inferred from our dataset (not the pretrained model's robot).
    # SmolVLA's architecture handles this fine via max_state_dim/max_action_dim padding.
    FEATURES_ARG=""
    if [ "$POLICY_TYPE" = "smolvla" ] && [ -n "$SMOLVLA_PRETRAINED" ]; then
        FEATURES_ARG="--policy.input_features=null"
    fi

    lerobot-train \
        --dataset.repo_id="$DATASET_REPO_ID" \
        --dataset.root="$DATASET_ROOT" \
        $POLICY_ARG \
        $FEATURES_ARG \
        --output_dir="$OUTPUT_DIR" \
        --job_name="${TASK_NAME}_il_${POLICY_TYPE}" \
        --policy.device="$DEVICE" \
        --policy.n_action_steps=10 \
        --policy.push_to_hub=false \
        --steps="$NUM_STEPS" \
        --batch_size="$BATCH_SIZE" \
        --log_freq="$LOG_FREQ" \
        --eval_freq="$EVAL_FREQ" \
        --save_freq="$SAVE_FREQ" \
        --dataset.image_transforms.enable=true \
        --wandb.enable="$WANDB" \
        --wandb.project="${TASK_NAME}_il"
fi

echo ""
echo "=============================================="
echo "    TRAINING COMPLETE"
echo "=============================================="
echo ""
echo "Checkpoints saved to: $OUTPUT_DIR/checkpoints/"
echo ""
echo "To evaluate on the robot, run:"
echo "  ./1.2_eval_il.sh"
echo ""
echo "To evaluate a specific checkpoint:"
echo "  ./1.2_eval_il.sh --policy-path $OUTPUT_DIR/checkpoints/last/pretrained_model"
