#!/usr/bin/env bash
# exp015 (Phase 1A — Single-Frame + Delta Arm Actions + Calibration/Dropout + Critical Weighting):
# Cosmos 3 Edge (L20) + Joint Action-LoRA (gen_only r=16/a=32, BS 64 = 16x4 for 24 GB RTX 4090),
# --history-size 1, --delta-actions, --state-offset-aug-deg 5.0, --state-dropout-prob 0.25,
# --critical-weight 4.0 --idle-weight 0.1.
# Target host: abakus RTX 4090 (24 GB) | Dataset: Orellius/so101_sort_cubes_no_top@a7c156de
set -Eeuo pipefail

export RUN_TAG="exp015_c3_edge_hist1_delta_calaug"
export VAM_CACHE_DIR="${VAM_CACHE_DIR:-$HOME/.cache/video-vam}"
export HF_AUTO_UPLOAD="${HF_AUTO_UPLOAD:-1}"
export HF_UPLOAD_OWNER="${HF_UPLOAD_OWNER:-Orellius}"

bash scripts/video_vam/run_experiment.sh c3_sort_cubes_portable \
  --online-backbone cosmos3_edge \
  --backbone-checkpoint "${VAM_CACHE_DIR}/cosmos3-edge" \
  --backbone-layer 20 \
  --history-size 1 \
  --delta-actions \
  --state-offset-aug-deg 5.0 \
  --state-dropout-prob 0.25 \
  --critical-weight 4.0 \
  --idle-weight 0.1 \
  --train-backbone-lora \
  --backbone-lora-pathway gen_only \
  --backbone-lora-rank 16 \
  --backbone-lora-alpha 32 \
  --backbone-grad-checkpointing \
  --backbone-lora-lr 5e-5 \
  --lr 3e-4 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --warmup-steps 200 \
  --val-every 250 \
  --patience 8 \
  --min-steps 500 \
  --max-steps 3500 \
  --select-metric val_first5 \
  --val-flow-num-t 4 \
  --seed 42 \
  "$@"
