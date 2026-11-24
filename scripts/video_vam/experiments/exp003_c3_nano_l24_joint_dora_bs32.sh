#!/usr/bin/env bash
# exp003: Cosmos 3 Nano (early-exit L24/36, 4096-dim) + Joint Action-DoRA (gen_only, r=16, alpha=32),
# effective BS 32 (8x4 for DoRA fp32 norm headroom), physics aug, stratified 4-t val_flow_loss, auto-upload.
# Target host: RunPod RTX 5090 (32 GB) | Dataset: Orellius/so101_sort_cubes_no_top@a7c156de
set -Eeuo pipefail

export RUN_TAG="exp003_c3_nano_l24_joint_dora_bs32"
export VAM_CACHE_DIR="${VAM_CACHE_DIR:-$HOME/.cache/video-vam}"
export HF_AUTO_UPLOAD="${HF_AUTO_UPLOAD:-1}"
export HF_UPLOAD_OWNER="${HF_UPLOAD_OWNER:-Orellius}"

bash scripts/video_vam/run_experiment.sh c3_sort_cubes_portable \
  --online-backbone cosmos3_nano \
  --backbone-checkpoint "${VAM_CACHE_DIR}/cosmos3-nano" \
  --backbone-layer 24 \
  --train-backbone-lora \
  --dora \
  --backbone-lora-pathway gen_only \
  --backbone-lora-rank 16 \
  --backbone-lora-alpha 32 \
  --backbone-grad-checkpointing \
  --backbone-lora-lr 5e-5 \
  --lr 2e-4 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --warmup-steps 300 \
  --val-every 250 \
  --patience 8 \
  --min-steps 500 \
  --max-steps 10000 \
  --val-flow-num-t 4 \
  --seed 42 \
  "$@"
