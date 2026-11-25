#!/usr/bin/env bash
# exp009 (Arm B — 2-Stage: Video Prediction LoRA -> Frozen-Backbone Action Expert):
# Loads exp008's video-prediction LoRA (frozen) and trains SmolExpert with identical recipe to exp007.
# Target host: RunPod RTX 5090 (32 GB) | Dataset: Orellius/so101_sort_cubes_no_top@a7c156de
set -Eeuo pipefail

export RUN_TAG="exp009_c3_edge_vidlora_then_frozen_action"
export VAM_CACHE_DIR="${VAM_CACHE_DIR:-$HOME/.cache/video-vam}"
export HF_AUTO_UPLOAD="${HF_AUTO_UPLOAD:-1}"
export HF_UPLOAD_OWNER="${HF_UPLOAD_OWNER:-Orellius}"
VIDEO_LORA="${VIDEO_LORA:-outputs/train/exp008_c3_edge_video_pred_lora/best_lora.safetensors}"

bash scripts/video_vam/run_experiment.sh c3_sort_cubes_portable \
  --online-backbone cosmos3_edge \
  --backbone-checkpoint "${VAM_CACHE_DIR}/cosmos3-edge" \
  --backbone-layer 20 \
  --backbone-lora-weights "${VIDEO_LORA}" \
  --backbone-lora-rank 16 \
  --backbone-lora-alpha 32 \
  --lr 3e-4 \
  --batch-size 32 \
  --grad-accum-steps 2 \
  --warmup-steps 200 \
  --val-every 250 \
  --patience 8 \
  --min-steps 500 \
  --max-steps 3500 \
  --select-metric val_first5 \
  --val-flow-num-t 4 \
  --seed 42 \
  "$@"
