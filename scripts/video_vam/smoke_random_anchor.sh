#!/usr/bin/env bash
# Smoke-test both random-anchor context modes, including a mid-run resume.
set -euo pipefail
cd /home/anton/lerobot-video-vam
source scripts/video_vam/cosmos_cuda_env.sh
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

MANIFEST=/home/anton/.cache/video-vam/cosmos-rehearsal-stride20-sigma4/manifest.json
SPLIT=/home/anton/.cache/video-vam/splits/rehearsal-stride20.json
ROOT=/home/anton/.cache/video-vam/cube-out-of-box-dataset
CKPT=/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt
TOK=/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth
PROMPT=/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors
OUT=/home/anton/lerobot-video-vam/outputs/smoke

common=(--manifest "$MANIFEST" --split "$SPLIT" --dataset-root "$ROOT"
  --batch-size 1 --grad-accum-steps 2 --warmup-steps 2 --lr 1e-4 --weight-decay 0.1
  --grad-clip 10.0 --eval-sigma 4.0 --select-metric flow --device cuda --seed 0)

echo "=== state-only: 6 steps ==="
.venv/bin/python scripts/video_vam/train_cosmos_world2action.py "${common[@]}" \
  --context-mode state-only --output-dir "$OUT/state-only" \
  --max-steps 6 --val-every 3 --save-every 3 --overwrite

echo "=== state-only: resume to 10 steps ==="
.venv/bin/python scripts/video_vam/train_cosmos_world2action.py "${common[@]}" \
  --context-mode state-only --output-dir "$OUT/state-only" \
  --max-steps 10 --val-every 5 --save-every 5 --resume

echo "=== online-random: 4 steps ==="
.venv/bin/python scripts/video_vam/train_cosmos_world2action.py "${common[@]}" \
  --context-mode online-random --output-dir "$OUT/online-random" \
  --backbone-checkpoint "$CKPT" --tokenizer "$TOK" --prompt "$PROMPT" \
  --max-steps 4 --val-every 4 --save-every 4 --overwrite

echo "=== online-random: resume to 6 steps ==="
.venv/bin/python scripts/video_vam/train_cosmos_world2action.py "${common[@]}" \
  --context-mode online-random --output-dir "$OUT/online-random" \
  --backbone-checkpoint "$CKPT" --tokenizer "$TOK" --prompt "$PROMPT" \
  --max-steps 6 --val-every 6 --save-every 6 --resume

echo "SMOKE_DONE"
