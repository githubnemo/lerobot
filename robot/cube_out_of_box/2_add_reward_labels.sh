#!/bin/bash
# Step 2: Add reward labels to collected dataset
# This post-processes the dataset to add reward=1.0 for success frames

set -e

export HF_HOME="/home/nemo/.cache/pysandbox-lerobot/huggingface"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=============================================="
echo "    CUBE OUT OF BOX - ADD REWARD LABELS"
echo "=============================================="
echo ""
echo "This script will:"
echo "  1. Load the collected dataset"
echo "  2. Mark last N frames of each episode as success (reward=1.0)"
echo "  3. Mark all other frames as failure (reward=0.0)"
echo "  4. Add 'next.done' column for episode boundaries"
echo ""

# Number of frames at end of episode to mark as success
# More frames = less class imbalance = better training
N_SUCCESS_FRAMES=5

# Dataset paths - must match what was used in collection
REPO_ID="nemo/cube_out_of_box_dataset"
ROOT="./data/cube_out_of_box_dataset"

echo "Dataset: $REPO_ID"
echo "Root: $ROOT"
echo "Success frames per episode: $N_SUCCESS_FRAMES"
echo ""

python "${SCRIPT_DIR}/../helpers/add_reward_labels.py" \
    --repo-id "$REPO_ID" \
    --root "$ROOT" \
    --n-success-frames "$N_SUCCESS_FRAMES"

echo ""
echo "=============================================="
echo "    DONE! Dataset now has reward labels."
echo "=============================================="
echo ""
echo "Next step: Train reward classifier with:"
echo "  ./3_train_reward_classifier.sh"

