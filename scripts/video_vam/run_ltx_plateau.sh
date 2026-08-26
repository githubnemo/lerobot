#!/usr/bin/env bash
set -euo pipefail

cd /home/anton/lerobot-video-vam
source scripts/video_vam/cosmos_cuda_env.sh

run_dir=/home/anton/.cache/video-vam/runs/ltx25-frozen-pool2-w2a-plateau-20260825
mkdir -p "$run_dir"

.venv/bin/python scripts/video_vam/build_ltx_feature_cache.py \
  --train-output-dir /home/anton/.cache/video-vam/ltx25-train0-31-stride3-pool2 \
  --val-output-dir /home/anton/.cache/video-vam/ltx25-val32-39-stride20-pool2 \
  --seed 0 --min-free-gib 20 --resume \
  2>&1 | tee "$run_dir/cache.log"

.venv/bin/python scripts/video_vam/train_ltx_world2action.py \
  --train-manifest /home/anton/.cache/video-vam/ltx25-train0-31-stride3-pool2/manifest.json \
  --val-manifest /home/anton/.cache/video-vam/ltx25-val32-39-stride20-pool2/manifest.json \
  --split /home/anton/.cache/video-vam/splits/rehearsal-stride20.json \
  --normalizer /home/anton/.cache/video-vam/runs/vam-hour-k8/normalizer.safetensors \
  --output-dir "$run_dir" \
  --max-steps 200000 --max-hours 72 \
  --batch-size 2 --grad-accum-steps 2 \
  --flow-draws 8 --flow-draw-chunk 4 \
  --val-every 300 --patience 20 --min-delta 0 \
  --warmup-steps 200 --lr 1e-4 --weight-decay 0.1 \
  --grad-clip 10 --loss-scale 1 --seed 0 \
  --wandb-project video-vam-world2action \
  --wandb-run-name ltx25-frozen-pool2-w2a-plateau-s0-20260825 \
  --overwrite \
  2>&1 | tee "$run_dir/train.log"

.venv/bin/python scripts/video_vam/evaluate_ltx_action_rmse.py \
  --val-manifest /home/anton/.cache/video-vam/ltx25-val32-39-stride20-pool2/manifest.json \
  --split /home/anton/.cache/video-vam/splits/rehearsal-stride20.json \
  --normalizer /home/anton/.cache/video-vam/runs/vam-hour-k8/normalizer.safetensors \
  --checkpoint "$run_dir/best.safetensors" \
  --output "$run_dir/protocol_eval.json" --overwrite \
  2>&1 | tee "$run_dir/eval.log"

printf 'LTX_PLATEAU_QUEUE_COMPLETE=1\n'
