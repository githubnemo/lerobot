#!/usr/bin/env bash
# exp008 (Stage 1 — Video Prediction Pre-Adaptation): Train Cosmos 3 Edge gen_only video-prediction LoRA
# (r=16, alpha=32, 17-frame 10Hz clips -> 2 cond + 3 future latents) on sort-cubes train eps 0..67, val 68..76.
# Saves outputs/train/exp008_c3_edge_video_pred_lora/best_lora.safetensors for Stage 2 action runs (exp009, exp010).
# Target host: RunPod RTX 5090 (32 GB) | Dataset: Orellius/so101_sort_cubes_no_top@a7c156de
set -Eeuo pipefail

export RUN_TAG="exp008_c3_edge_video_pred_lora"
export VAM_CACHE_DIR="${VAM_CACHE_DIR:-$HOME/.cache/video-vam}"

HF_AUTO_UPLOAD=0 bash scripts/video_vam/run_experiment.sh lora_cosmos3_edge_sort_cubes_peft \
  --checkpoint-dir "${VAM_CACHE_DIR}/cosmos3-edge" \
  --cache-dir "${VAM_CACHE_DIR}/cosmos3-edge-sortcubes-latents-cache" \
  --adapter-format legacy \
  --pathway gen_only \
  --rank 16 \
  --alpha 32 \
  --lr 2e-4 \
  --warmup-steps 200 \
  --val-every 250 \
  --patience 8 \
  --max-steps 3000 \
  --seed 42 \
  "$@"
