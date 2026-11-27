#!/usr/bin/env bash
# exp021 (Phase 3A — Cube Stacking 6-D pos-only baseline):
# Cosmos 3 Edge (L20) on Orellius/so101_stack_cubes_full_telemetry@d7fc3c81 (52 train / 8 val),
# --state-feature-groups pos (6-D joint positions only). Pass --history-size 3/5 overrides if Phase 1 favors multi-frame.
# Target host: abakus RTX 4090 (24 GB)
set -Eeuo pipefail

export RUN_TAG="exp021_c3_edge_stack_pos6d"
export VAM_CACHE_DIR="${VAM_CACHE_DIR:-$HOME/.cache/video-vam}"
export HF_AUTO_UPLOAD="${HF_AUTO_UPLOAD:-1}"
export HF_UPLOAD_OWNER="${HF_UPLOAD_OWNER:-Orellius}"

bash scripts/video_vam/run_experiment.sh c3_sort_cubes_portable \
  --dataset-repo-id Orellius/so101_stack_cubes_full_telemetry \
  --dataset-revision d7fc3c81f124886f9a1ec4840891660d60691424 \
  --train-episodes 0:52 \
  --val-episodes 52:60 \
  --backbone-prompt "Stack the cubes on top of each other" \
  --online-backbone cosmos3_edge \
  --backbone-checkpoint "${VAM_CACHE_DIR}/cosmos3-edge" \
  --backbone-layer 20 \
  --state-feature-groups pos \
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
