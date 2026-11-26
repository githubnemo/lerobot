#!/usr/bin/env bash
# exp013 (REMOTE 4090 RPC moonshot): nvidia/Cosmos3-Super (60B, L42/64, 5120-dim) + Streaming 4-bit NF4 QLoRA
# (gen_only, r=16, alpha=32), BS 32 (4x8), 3,500-step cosine schedule, --select-metric val_first5.
# Fits on 24 GB RTX 4090 (~13-18 GB VRAM) for both QLoRA training and SSH RPC serving to sahni!
# Target host: abakus RTX 4090 (24 GB) or RunPod RTX 5090 (32 GB) | Dataset: Orellius/so101_sort_cubes_no_top@a7c156de
set -Eeuo pipefail

export RUN_TAG="exp013_c3_super_l42_nf4_qlora"
export VAM_CACHE_DIR="${VAM_CACHE_DIR:-$HOME/.cache/video-vam}"
export HF_AUTO_UPLOAD="${HF_AUTO_UPLOAD:-1}"
export HF_UPLOAD_OWNER="${HF_UPLOAD_OWNER:-Orellius}"

bash scripts/video_vam/run_experiment.sh c3_sort_cubes_portable \
  --online-backbone cosmos3_super \
  --backbone-checkpoint "${VAM_CACHE_DIR}/cosmos3-super" \
  --backbone-layer 42 \
  --backbone-quantize nf4 \
  --train-backbone-lora \
  --backbone-lora-pathway gen_only \
  --backbone-lora-rank 16 \
  --backbone-lora-alpha 32 \
  --backbone-grad-checkpointing \
  --backbone-lora-lr 5e-5 \
  --lr 3e-4 \
  --batch-size 4 \
  --grad-accum-steps 8 \
  --warmup-steps 200 \
  --val-every 250 \
  --patience 8 \
  --min-steps 500 \
  --max-steps 3500 \
  --select-metric val_first5 \
  --val-flow-num-t 4 \
  --seed 42 \
  "$@"
