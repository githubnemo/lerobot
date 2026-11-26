#!/usr/bin/env bash
# exp011 (LOCAL 8 GB RTX 3050 upgrade): nvidia/Cosmos3-Edge-Policy-DROID (L20, robot-manipulation pretrained)
# + Joint Action-LoRA (gen_only, r=16, alpha=32), BS 64 (32x2), 3,500-step cosine schedule, --select-metric val_first5.
# Direct 1:1 A/B against exp007 (vanilla Cosmos3-Edge: Full-30 14.577, H1 4.080, First-5 5.601).
# Target host: RunPod RTX 5090 (32 GB) or abakus 4090 | Dataset: Orellius/so101_sort_cubes_no_top@a7c156de
set -Eeuo pipefail

export RUN_TAG="exp011_c3_edge_policy_droid_joint_lora"
export VAM_CACHE_DIR="${VAM_CACHE_DIR:-$HOME/.cache/video-vam}"
export HF_AUTO_UPLOAD="${HF_AUTO_UPLOAD:-1}"
export HF_UPLOAD_OWNER="${HF_UPLOAD_OWNER:-Orellius}"

bash scripts/video_vam/run_experiment.sh c3_sort_cubes_portable \
  --online-backbone cosmos3_edge \
  --backbone-checkpoint "${VAM_CACHE_DIR}/cosmos3-edge-policy-droid" \
  --backbone-layer 20 \
  --train-backbone-lora \
  --backbone-lora-pathway gen_only \
  --backbone-lora-rank 16 \
  --backbone-lora-alpha 32 \
  --backbone-grad-checkpointing \
  --backbone-lora-lr 5e-5 \
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
