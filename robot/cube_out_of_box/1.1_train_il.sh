#!/usr/bin/env bash
# Train a small SmolVLA or ACT imitation-learning baseline on the pinned dataset.
# Future VAM policy integration belongs behind the policy-type extension point.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/home/anton/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

DATASET_REPO_ID="hubnemo/cube_out_of_box_dataset"
DATASET_REVISION="243370c3c08bcbd860133c4a0d658ea7c1d2e77e"
DEFAULT_DATASET_ROOT="$REPO_ROOT/data/cube_out_of_box_dataset"
DATASET_ROOT="${CUBE_DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"

POLICY_TYPE="smolvla"
NUM_STEPS=20000
BATCH_SIZE=16
SUBSET="all"
RESUME=false
WANDB=false
DEVICE="cuda"
IMAGE_AUG=false
USE_AMP=true
OUTPUT_DIR=""
SMOLVLA_PRETRAINED="lerobot/smolvla_base"

usage() {
    echo "Usage: $0 [--policy-type act|smolvla] [--steps N] [--batch-size N]"
    echo "       [--subset 1|3|5|all] [--dataset-root PATH] [--output-dir PATH]"
    echo "       [--resume] [--wandb] [--device cuda|cpu] [--from-scratch] [--no-amp]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --policy-type) shift; POLICY_TYPE="$1" ;;
        --steps) shift; NUM_STEPS="$1" ;;
        --batch-size) shift; BATCH_SIZE="$1" ;;
        --subset) shift; SUBSET="$1" ;;
        --dataset-root) shift; DATASET_ROOT="$1" ;;
        --output-dir) shift; OUTPUT_DIR="$1" ;;
        --resume) RESUME=true ;;
        --wandb) WANDB=true ;;
        --device) shift; DEVICE="$1" ;;
        --from-scratch) SMOLVLA_PRETRAINED="" ;;
        --no-amp) USE_AMP=false ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
    shift
done

case "$POLICY_TYPE" in
    act|smolvla) ;;
    vam)
        echo "The VAM policy is intentionally not implemented yet. Use act or smolvla." >&2
        exit 2
        ;;
    *) echo "Unsupported policy type: $POLICY_TYPE" >&2; usage; exit 2 ;;
esac

case "$SUBSET" in
    1) EPISODES="[0]" ;;
    3) EPISODES="[0,20,39]" ;;
    5) EPISODES="[0,9,20,33,39]" ;;
    all) EPISODES="" ;;
    *) echo "Unsupported subset: $SUBSET" >&2; exit 2 ;;
esac

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$REPO_ROOT/outputs/train/cube_out_of_box_il_${POLICY_TYPE}_subset_${SUBSET}"
fi

if [ "$RESUME" = false ] && [ -e "$OUTPUT_DIR" ]; then
    echo "Refusing to overwrite existing output: $OUTPUT_DIR" >&2
    echo "Choose another --output-dir or pass --resume for an existing checkpoint." >&2
    exit 2
fi

if [ "$RESUME" = true ]; then
    CHECKPOINT_CONFIG="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
    if [ ! -f "$CHECKPOINT_CONFIG" ]; then
        echo "No resumable checkpoint found at $CHECKPOINT_CONFIG" >&2
        exit 2
    fi
    lerobot-train \
        --config_path="$CHECKPOINT_CONFIG" \
        --resume=true \
        --steps="$NUM_STEPS"
    exit 0
fi

POLICY_ARGS=()
if [ "$POLICY_TYPE" = "smolvla" ] && [ -n "$SMOLVLA_PRETRAINED" ]; then
    POLICY_ARGS+=("--policy.path=$SMOLVLA_PRETRAINED" "--policy.input_features=null")
else
    POLICY_ARGS+=("--policy.type=$POLICY_TYPE")
fi

DATASET_ARGS=(
    "--dataset.repo_id=$DATASET_REPO_ID"
    "--dataset.root=$DATASET_ROOT"
    "--dataset.revision=$DATASET_REVISION"
)
if [ -n "$EPISODES" ]; then
    DATASET_ARGS+=("--dataset.episodes=$EPISODES")
fi

echo "Training $POLICY_TYPE on $DATASET_REPO_ID@$DATASET_REVISION"
echo "Dataset root: $DATASET_ROOT"
echo "Episode subset: $SUBSET"
echo "Output: $OUTPUT_DIR"

lerobot-train \
    "${DATASET_ARGS[@]}" \
    "${POLICY_ARGS[@]}" \
    --policy.chunk_size=30 \
    --policy.n_action_steps=10 \
    --policy.device="$DEVICE" \
    --policy.use_amp="$USE_AMP" \
    --policy.push_to_hub=false \
    --output_dir="$OUTPUT_DIR" \
    --job_name="cube_out_of_box_il_${POLICY_TYPE}_subset_${SUBSET}" \
    --steps="$NUM_STEPS" \
    --batch_size="$BATCH_SIZE" \
    --dataset.image_transforms.enable="$IMAGE_AUG" \
    --wandb.enable="$WANDB" \
    --wandb.project="cube_out_of_box_il"
