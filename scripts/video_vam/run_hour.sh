#!/usr/bin/env bash
# One-hour random-anchor VAM run and one-hour proprioception floor, then score both.
set -uo pipefail
cd /home/anton/lerobot-video-vam
source scripts/video_vam/cosmos_cuda_env.sh
export HF_HUB_OFFLINE=1

MANIFEST=/home/anton/.cache/video-vam/cosmos-rehearsal-stride20-sigma4/manifest.json
SPLIT=/home/anton/.cache/video-vam/splits/rehearsal-stride20.json
ROOT=/home/anton/.cache/video-vam/cube-out-of-box-dataset
CKPT=/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt
TOK=/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth
PROMPT=/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors
SMOLVLA=/home/anton/lerobot-video-vam/outputs/train/cube_out_of_box_il_smolvla_heldout_0_31_20260818_retry4/checkpoints/005000/pretrained_model
RUNS=/home/anton/.cache/video-vam/runs
VAM="$RUNS/vam-hour-k8"
PROPRIO="$RUNS/proprio-hour-k8"

# Fix 2 settings, chosen from the measured A/B: clipping is inactive in the normal
# regime and only catches genuine spikes, and the logged norm is the true loss gradient.
common=(--manifest "$MANIFEST" --split "$SPLIT" --dataset-root "$ROOT"
  --batch-size 1 --grad-accum-steps 2 --flow-draws 8 --lr 1e-4 --weight-decay 0.1
  --loss-scale 1.0 --grad-clip 10.0 --val-every 50 --save-every 50 --eval-sigma 4.0
  --select-metric rmse --device cuda --seed 0 --max-hours 1.0
  --wandb-project video-vam-world2action)

score() {
  .venv/bin/python scripts/video_vam/evaluate_action_rmse.py --manifest "$MANIFEST" \
    --split "$SPLIT" --dataset-root "$ROOT" --vam-checkpoint "$1/best.safetensors" \
    --vam-normalizer "$1/normalizer.safetensors" --smolvla-checkpoint "$SMOLVLA" \
    --device cuda --batch-size 1 --seed 0 ${2:-} --output "$1/action-rmse.json" --overwrite
}

echo "=== [1/4] one-hour random-anchor VAM (lr 1e-4, K=8) ==="
.venv/bin/python scripts/video_vam/train_cosmos_world2action.py "${common[@]}" \
  --context-mode online-random --output-dir "$VAM" \
  --backbone-checkpoint "$CKPT" --tokenizer "$TOK" --prompt "$PROMPT" \
  --max-steps 310 --warmup-steps 30 --overwrite
echo "VAM_EXIT=$?"

echo "=== [2/4] scoring VAM ==="
score "$VAM"
echo "VAM_EVAL_EXIT=$?"

echo "=== [3/4] one-hour proprioception floor (K=8, fixed clipping regime) ==="
.venv/bin/python scripts/video_vam/train_cosmos_world2action.py "${common[@]}" \
  --context-mode state-only --output-dir "$PROPRIO" \
  --max-steps 950 --warmup-steps 95 --overwrite
echo "PROPRIO_EXIT=$?"

echo "=== [4/4] scoring proprioception floor ==="
score "$PROPRIO" --vam-zero-context
echo "PROPRIO_EVAL_EXIT=$?"

touch "$RUNS/HOUR_RUNS_COMPLETE"
echo "ALL_DONE"
