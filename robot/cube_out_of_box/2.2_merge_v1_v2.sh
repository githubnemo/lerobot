#!/usr/bin/env bash
# Merge pinned cube v1 with locally recorded v2 episodes.
# Training should use the merged root, not the v2-only collection folder.
#
#   ./robot/cube_out_of_box/2.2_merge_v1_v2.sh
#   ./robot/cube_out_of_box/2.2_merge_v1_v2.sh --push

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

HF_USER="${HF_USER:-hubnemo}"
V1_REPO_ID="hubnemo/cube_out_of_box_dataset"
V1_REVISION="243370c3c08bcbd860133c4a0d658ea7c1d2e77e"
V1_ROOT="${CUBE_DATASET_ROOT:-$REPO_ROOT/data/cube_out_of_box_dataset}"
V2_REPO_ID="${HF_USER}/cube_out_of_box_v2"
V2_ROOT="${CUBE_V2_ROOT:-$REPO_ROOT/data/cube_out_of_box_v2}"
MERGED_REPO_ID="${HF_USER}/cube_out_of_box_v2_full"
MERGED_ROOT="${CUBE_V2_FULL_ROOT:-$REPO_ROOT/data/cube_out_of_box_v2_full}"
PUSH=false

usage() {
    echo "Usage: $0 [--push] [--v2-root PATH] [--merged-root PATH]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --push) PUSH=true ;;
        --v2-root) shift; V2_ROOT="$1" ;;
        --merged-root) shift; MERGED_ROOT="$1" ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
    shift
done

if [ ! -d "$V2_ROOT/meta" ]; then
    echo "No v2 collection at $V2_ROOT" >&2
    echo "Record both setups first: ./robot/cube_out_of_box/2.1_record_v2.sh --setup boxed" >&2
    exit 2
fi

if [ -e "$MERGED_ROOT" ]; then
    echo "Merged output already exists: $MERGED_ROOT" >&2
    echo "Move or delete it, or pass --merged-root." >&2
    exit 2
fi

echo "Ensuring v1 is local at $V1_ROOT (revision $V1_REVISION)..."
uv run python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
LeRobotDataset(
    '$V1_REPO_ID',
    root='$V1_ROOT',
    revision='$V1_REVISION',
    download_videos=True,
)
print('v1 ready')
"

echo "Merging:"
echo "  v1  $V1_REPO_ID @ $V1_REVISION"
echo "      $V1_ROOT"
echo "  v2  $V2_REPO_ID"
echo "      $V2_ROOT"
echo "  ->  $MERGED_REPO_ID"
echo "      $MERGED_ROOT"

MERGE_ARGS=(
    --new_repo_id "$MERGED_REPO_ID"
    --new_root "$MERGED_ROOT"
    --operation.type merge
    --operation.repo_ids "['$V1_REPO_ID', '$V2_REPO_ID']"
    --operation.roots "['$V1_ROOT', '$V2_ROOT']"
    --push_to_hub "$PUSH"
)

uv run lerobot-edit-dataset "${MERGE_ARGS[@]}"

echo "Done. Train on --dataset.repo_id=$MERGED_REPO_ID --dataset.root=$MERGED_ROOT"
