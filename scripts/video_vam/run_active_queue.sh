#!/usr/bin/env bash
# Overnight queue:
# 1. Cosmos 2B Multi-Depth Feature Extraction (Layers 4, 8, 12, 16, 18, 20) & Policy Training
# 2. Cosmos 7B Transformer Shards Download & V2 Held-Out Evaluation

set -Eeuo pipefail
REPO_ROOT="/home/anton/lerobot-video-vam"
cd "$REPO_ROOT"

source scripts/video_vam/cosmos_cuda_env.sh
export PYTHONPATH=".:src"
export PATH="/home/anton/.local/bin:$PATH"
PYTHON="/home/anton/lerobot-video-vam/.venv/bin/python"

mkdir -p outputs/logs outputs/train outputs/features outputs/evaluation

LOG_FILE="outputs/logs/active_queue_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "LAUNCHING ACTIVE QUEUE AT $(date)"
echo "Log file: $LOG_FILE"
echo "================================================================================"

run_task() {
    local task_name="$1"
    shift
    echo ""
    echo "================================================================================"
    echo ">>> STARTING: $task_name at $(date)"
    echo "================================================================================"
    set +e
    "$@"
    local status=$?
    set -e
    if [ $status -eq 0 ]; then
        echo ">>> SUCCEEDED: $task_name at $(date)"
    else
        echo ">>> FAILED (exit code $status): $task_name at $(date)"
    fi
    return 0
}

# ------------------------------------------------------------------------------
# 1. Cosmos 2B Multi-Depth Cache Extraction & Policy Training
# ------------------------------------------------------------------------------
C2B_DIR="outputs/features/cosmos2b_multidepth"
mkdir -p "$C2B_DIR/train" "$C2B_DIR/val"

if [ ! -f "$C2B_DIR/train/manifest.json" ]; then
    run_task "Extract Cosmos 2B Multi-Depth Train Features" \
        "$PYTHON" scripts/video_vam/build_cosmos_feature_cache.py \
        --root /home/anton/.cache/video-vam/cube-out-of-box-dataset \
        --episodes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
        --stride 3 \
        --state-t 2 \
        --sigma 80.0 \
        --hidden-layers "4,8,12,16,18,20" \
        --checkpoint /home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt \
        --tokenizer /home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth \
        --lora-weights outputs/train/cosmos-video-lora-step6000/best_lora.safetensors \
        --output-dir "$C2B_DIR/train"
fi

if [ ! -f "$C2B_DIR/val/manifest.json" ]; then
    run_task "Extract Cosmos 2B Multi-Depth Val Features" \
        "$PYTHON" scripts/video_vam/build_cosmos_feature_cache.py \
        --root /home/anton/.cache/video-vam/cube-out-of-box-dataset \
        --episodes 32 33 34 35 36 37 38 39 \
        --stride 20 \
        --state-t 2 \
        --sigma 80.0 \
        --hidden-layers "4,8,12,16,18,20" \
        --checkpoint /home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt \
        --tokenizer /home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth \
        --lora-weights outputs/train/cosmos-video-lora-step6000/best_lora.safetensors \
        --output-dir "$C2B_DIR/val"
fi

# ------------------------------------------------------------------------------
# 2. Cosmos 7B Shards Download & V2 Held-Out Evaluation
# ------------------------------------------------------------------------------
run_task "Download Cosmos 7B Shards & Evaluate V2" \
    "$PYTHON" -c "
import time, subprocess
from huggingface_hub import snapshot_download

for attempt in range(1, 20):
    try:
        print(f'Download attempt {attempt}/20...', flush=True)
        path = snapshot_download(
            repo_id='Arsh9210/Cosmos-1.0-Diffusion-7B-Video2World',
            allow_patterns=['transformer/*'],
            resume_download=True,
        )
        print('Downloaded successfully to:', path)
        break
    except Exception as e:
        print(f'Attempt {attempt} failed: {e}. Retrying in 20s...', flush=True)
        time.sleep(20)

print('Running Cosmos 7B V2 evaluation...')
subprocess.run(['/home/anton/lerobot-video-vam/.venv/bin/python', 'scripts/video_vam/eval_v2_cosmos7b_only.py'], check=True)
"

echo ""
echo "================================================================================"
echo "QUEUE COMPLETE AT $(date)"
echo "================================================================================"
