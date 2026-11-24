#!/usr/bin/env bash
# exp006: Clean leak-free SmolVLA 450M baseline on Orellius/so101_sort_cubes_no_top@a7c156de
# (strictly train episodes 0..67, held-out 68..76 untouched, BS 32, 50k steps, AMP, auto-upload to HF Hub).
# Target host: RunPod RTX 5090 (32 GB)
set -Eeuo pipefail

export RUN_TAG="exp006_smolvla_sort_cubes_clean_heldout"
export HF_AUTO_UPLOAD="${HF_AUTO_UPLOAD:-1}"
export HF_UPLOAD_OWNER="${HF_UPLOAD_OWNER:-Orellius}"

bash scripts/video_vam/run_experiment.sh smolvla_sort_cubes_100k \
  --dataset.revision=a7c156de0e84a81e19457e6d783e4316e9f901e1 \
  --policy.use_amp=true \
  --steps=50000 \
  --scheduler.num_warmup_steps=2000 \
  --scheduler.num_decay_steps=50000 \
  --save_freq=10000 \
  --seed=42 \
  "$@"
