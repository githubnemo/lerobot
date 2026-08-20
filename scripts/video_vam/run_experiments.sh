#!/usr/bin/env bash
# Run the proprioception-only ablation, then the random-anchor VAM run, then score both.
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
STATE_ONLY="$RUNS/proprio-only-randanchor"
VAM="$RUNS/vam-randanchor-online"

common=(--manifest "$MANIFEST" --split "$SPLIT" --dataset-root "$ROOT"
  --batch-size 1 --grad-accum-steps 8 --lr 1e-4 --weight-decay 0.1 --grad-clip 10.0
  --warmup-steps 250 --val-every 50 --save-every 50 --eval-sigma 4.0
  --select-metric flow --device cuda --seed 0 --wandb-project video-vam-world2action)

# Sample GPU clocks so a step-time drift can be attributed rather than guessed at.
telemetry() {
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu \
    --format=csv,noheader -l 30 > "$1" 2>&1 &
  echo $!
}

echo "=== [1/4] proprioception-only ablation ==="
TELEMETRY_PID=$(telemetry "$STATE_ONLY.gpu.csv")
.venv/bin/python scripts/video_vam/train_cosmos_world2action.py "${common[@]}" \
  --context-mode state-only --output-dir "$STATE_ONLY" \
  --max-steps 600 --overwrite
echo "STATE_ONLY_EXIT=$?"
kill "$TELEMETRY_PID" 2>/dev/null || true

echo "=== [2/4] scoring proprioception-only ==="
.venv/bin/python scripts/video_vam/evaluate_action_rmse.py --manifest "$MANIFEST" --split "$SPLIT" \
  --dataset-root "$ROOT" --vam-checkpoint "$STATE_ONLY/best.safetensors" \
  --vam-normalizer "$STATE_ONLY/normalizer.safetensors" --smolvla-checkpoint "$SMOLVLA" \
  --device cuda --batch-size 1 --seed 0 --vam-zero-context \
  --output "$STATE_ONLY/action-rmse.json" --overwrite
echo "STATE_ONLY_EVAL_EXIT=$?"

echo "=== [3/4] random-anchor online VAM ==="
TELEMETRY_PID=$(telemetry "$VAM.gpu.csv")
.venv/bin/python scripts/video_vam/train_cosmos_world2action.py "${common[@]}" \
  --context-mode online-random --output-dir "$VAM" \
  --backbone-checkpoint "$CKPT" --tokenizer "$TOK" --prompt "$PROMPT" \
  --max-steps 750 --max-hours 7.0 --overwrite
echo "VAM_EXIT=$?"
kill "$TELEMETRY_PID" 2>/dev/null || true

echo "=== [4/4] scoring random-anchor VAM ==="
.venv/bin/python scripts/video_vam/evaluate_action_rmse.py --manifest "$MANIFEST" --split "$SPLIT" \
  --dataset-root "$ROOT" --vam-checkpoint "$VAM/best.safetensors" \
  --vam-normalizer "$VAM/normalizer.safetensors" --smolvla-checkpoint "$SMOLVLA" \
  --device cuda --batch-size 1 --seed 0 \
  --output "$VAM/action-rmse.json" --overwrite
echo "VAM_EVAL_EXIT=$?"

touch "$RUNS/EXPERIMENTS_COMPLETE"
echo "ALL_DONE"
