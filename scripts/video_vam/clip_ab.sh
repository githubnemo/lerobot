#!/usr/bin/env bash
# Compare clipping regimes on the fast proprioception path with K=8 draws.
set -uo pipefail
cd /home/anton/lerobot-video-vam
source scripts/video_vam/cosmos_cuda_env.sh
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

MANIFEST=/home/anton/.cache/video-vam/cosmos-rehearsal-stride20-sigma4/manifest.json
SPLIT=/home/anton/.cache/video-vam/splits/rehearsal-stride20.json
ROOT=/home/anton/.cache/video-vam/cube-out-of-box-dataset
OUT=/home/anton/.cache/video-vam/runs/clip-ab

common=(--manifest "$MANIFEST" --split "$SPLIT" --dataset-root "$ROOT"
  --context-mode state-only --batch-size 1 --grad-accum-steps 2 --flow-draws 8
  --lr 1e-4 --weight-decay 0.1 --warmup-steps 10 --val-every 50 --save-every 50
  --eval-sigma 4.0 --select-metric rmse --device cuda --seed 0 --max-steps 100 --overwrite)

run() {
  echo "=== $1 : loss_scale=$2 grad_clip=$3 ==="
  .venv/bin/python scripts/video_vam/train_cosmos_world2action.py "${common[@]}" \
    --loss-scale "$2" --grad-clip "$3" --output-dir "$OUT/$1"
  echo "EXIT_$1=$?"
}

run baseline-ls10-clip10 10.0 10.0
run ls1-clip10 1.0 10.0
run ls1-clip100 1.0 100.0

touch "$OUT/AB_COMPLETE"
echo "AB_DONE"
