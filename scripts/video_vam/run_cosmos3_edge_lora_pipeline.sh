#!/usr/bin/env bash
# Corrected native Cosmos 3 pipeline. An explicit fresh run directory prevents stale reuse.
set -Eeuo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/video_vam/run_cosmos3_edge_lora_pipeline.sh /absolute/fresh-run-dir" >&2
    exit 2
fi
RUN_DIR=$1
if [[ $RUN_DIR != /* || -e "$RUN_DIR" ]]; then
    echo "Run directory must be absolute and must not already exist" >&2
    exit 2
fi
cd /home/anton/lerobot-video-vam
source scripts/video_vam/cosmos_cuda_env.sh
source scripts/video_vam/gpu_lock.sh
acquire_gpu_lock cosmos3-native-v2
mkdir -p "$RUN_DIR"
UV=/home/anton/.local/bin/uv
TRAIN_EPISODES=({0..31})
VAL_EPISODES=({32..39})
"$UV" run --no-sync python -m scripts.video_vam.train_cosmos3_edge_video_lora \
    --train-episodes "${TRAIN_EPISODES[@]}" --val-episodes "${VAL_EPISODES[@]}" \
    --pathway dual --clip-frames 17 --max-steps 10000 --patience 20 \
    --output-dir "$RUN_DIR/video-lora"
"$UV" run --no-sync python -m scripts.video_vam.extract_cosmos3_edge_pure_vision \
    --episodes "${TRAIN_EPISODES[@]}" --stride 3 \
    --lora-checkpoint "$RUN_DIR/video-lora/best_lora.safetensors" --output-dir "$RUN_DIR/train"
"$UV" run --no-sync python -m scripts.video_vam.extract_cosmos3_edge_pure_vision \
    --episodes "${VAL_EPISODES[@]}" --stride 20 \
    --lora-checkpoint "$RUN_DIR/video-lora/best_lora.safetensors" --output-dir "$RUN_DIR/val"
"$UV" run --no-sync python -m scripts.video_vam.train_smolexpert \
    --backbone cosmos3-edge --protocol protocol1 \
    --train-manifest "$RUN_DIR/train/manifest.json" --val-manifest "$RUN_DIR/val/manifest.json" \
    --output-dir "$RUN_DIR/policy" --patience 20 --min-steps 10000 --max-steps 50000 --seed 0
printf "Completed native Cosmos 3 pipeline\n"
