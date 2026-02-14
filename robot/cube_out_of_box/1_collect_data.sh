#!/bin/bash
# Step 1: Collect demonstration data for cube out of box task
# Teleoperate the robot to take cube out of box, press Right Arrow to save episode

set -e

export HF_HOME="/home/nemo/.cache/pysandbox-lerobot/huggingface"
export DISPLAY=:0.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_ROOT="./data/cube_out_of_box_dataset"

echo "=============================================="
echo "    CUBE OUT OF BOX - DATA COLLECTION"
echo "=============================================="

# Check if dataset already exists
if [ -d "$DATASET_ROOT" ]; then
    echo ""
    echo "WARNING: Dataset already exists at $DATASET_ROOT"
    echo ""
    echo "Options:"
    echo "  [r] Resume recording (add more episodes)"
    echo "  [d] Delete and start fresh"
    echo "  [q] Quit"
    echo ""
    read -p "Choose [r/d/q]: " choice
    case "$choice" in
        r|R)
            echo "Resuming recording..."
            # Strip reward labels if they were added (they break resume)
            python -c "
import pyarrow.parquet as pq, json, sys
from pathlib import Path
root = Path('$DATASET_ROOT')
for pf in sorted(root.glob('data/**/*.parquet')):
    t = pq.read_table(pf)
    cols_to_drop = [c for c in ['next.reward', 'next.done'] if c in t.column_names]
    if cols_to_drop:
        for c in cols_to_drop:
            t = t.remove_column(t.column_names.index(c))
        pq.write_table(t, pf)
        print(f'Stripped {cols_to_drop} from {pf}')
info_path = root / 'meta' / 'info.json'
if info_path.exists():
    info = json.loads(info_path.read_text())
    changed = False
    for col in ['next.reward', 'next.done']:
        if col in info.get('features', {}):
            del info['features'][col]
            changed = True
    if changed:
        info_path.write_text(json.dumps(info, indent=2))
        print('Stripped reward/done from info.json')
" 2>/dev/null
            # Clear any HF dataset cache
            rm -rf "$DATASET_ROOT/.cache" 2>/dev/null
            RESUME_FLAG="--resume=true"
            ;;
        d|D)
            echo "Deleting existing dataset..."
            rm -rf "$DATASET_ROOT"
            RESUME_FLAG=""
            ;;
        *)
            echo "Exiting."
            exit 0
            ;;
    esac
else
    RESUME_FLAG=""
fi
echo ""
echo "=== CONTROLS ==="
echo "  Leader arm:    Move to control follower"
echo "  Right Arrow:   Save episode (success) & move to next"
echo "  Left Arrow:    Discard episode & re-record"
echo "  Escape:        Stop recording completely"
echo "================"
echo ""
echo "Goal: Touch the cube with the gripper"
echo "Press Right Arrow when the gripper successfully takes the cube out of the box"
echo ""

# Using standard lerobot-record with leader-follower teleoperation
# Cameras are passed as a YAML/JSON string
python -m lerobot.scripts.lerobot_record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=shabby \
    --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, rotation: ROTATE_180}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=shabby \
    --dataset.repo_id=hubnemo/cube_out_of_box_dataset \
    --dataset.root="$DATASET_ROOT" \
    --dataset.single_task="take cube out of box" \
    --dataset.num_episodes=20 \
    --dataset.fps=10 \
    --dataset.vcodec=h264 \
    --dataset.reset_time_s=5 \
    --display_data=true \
    --play_sounds=false \
    $RESUME_FLAG
