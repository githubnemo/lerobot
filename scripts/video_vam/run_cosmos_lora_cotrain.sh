#!/usr/bin/env bash
# Precompute full-clip latents, then run arm A and arm B with crash-resume retries.
set -uo pipefail

cd /home/anton/lerobot-video-vam
# shellcheck source=scripts/video_vam/cosmos_cuda_env.sh
source scripts/video_vam/cosmos_cuda_env.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MANIFEST=/home/anton/.cache/video-vam/cosmos-rehearsal-stride20-sigma4/manifest.json
SPLIT=/home/anton/.cache/video-vam/splits/rehearsal-stride20.json
ROOT=/home/anton/.cache/video-vam/cube-out-of-box-dataset
CKPT=/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt
TOK=/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth
PROMPT=/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors
RUNS=/home/anton/.cache/video-vam/runs
LATENTS="$RUNS/cosmos-lora-full-61-latents"
PRECOMPUTE_OUTPUT="$RUNS/cosmos-lora-latent-precompute"
ARM_A="$RUNS/cosmos-lora-cotrain-arm-a"
ARM_B="$RUNS/cosmos-lora-cotrain-arm-b"

common=(
  --manifest "$MANIFEST"
  --split "$SPLIT"
  --dataset-root "$ROOT"
  --backbone-checkpoint "$CKPT"
  --tokenizer "$TOK"
  --prompt "$PROMPT"
  --latent-cache-dir "$LATENTS"
  --batch-size 1
  --grad-accum-steps 1
  --flow-draws 8
  --lora-rank 16
  --lora-lr 1e-4
  --decoder-lr 1e-4
  --weight-decay 0.1
  --grad-clip 1.0
  --lambda-video 0.5
  --val-every 100
  --save-every 100
  --patience 20
  --eval-sigma 80
  --context-transform pool2
  --device cuda
  --seed 0
  --wandb-project video-vam-world2action
)

run_with_retry() {
  local arm="$1"
  local output_dir="$2"
  local max_hours="$3"
  local grad_arm="$4"
  shift 4
  local extra=("$@")
  local attempt=0
  while :; do
    attempt=$((attempt + 1))
    local resume_flag=()
    if [[ -f "$output_dir/last.json" && -f "$output_dir/last.state.pt" ]]; then
      resume_flag=(--resume)
    elif compgen -G "$output_dir/*" > /dev/null; then
      resume_flag=(--overwrite)
    fi
    echo "=== ${arm}: attempt ${attempt} (mode ${resume_flag[*]}) ==="
    "$VAM_VENV/bin/python" scripts/video_vam/train_cosmos_lora_cotrain.py \
      "${common[@]}" \
      --output-dir "$output_dir" \
      --max-steps 500000 \
      --max-hours "$max_hours" \
      --action-grad-to-backbone "$grad_arm" \
      "${resume_flag[@]}" \
      "${extra[@]}"
    local status=$?
    if (( status == 0 )); then
      echo "${arm}_EXIT=0"
      return 0
    fi
    echo "${arm}_EXIT=${status}; retrying in 15 seconds" >&2
    sleep 15
  done
}

mkdir -p "$RUNS" "$LATENTS" "$PRECOMPUTE_OUTPUT"
latent_count=$(find "$LATENTS" -type f | wc -l)
if (( latent_count >= 300 )); then
  echo "=== [1/3] skip latent cache; already have ${latent_count} files ==="
else
  echo "=== [1/3] full 61-frame latent cache (resume by existing samples) ==="
  run_with_retry "LATENT_CACHE" "$PRECOMPUTE_OUTPUT" 24 off --precompute-latents-only
fi

echo "=== [2/3] arm A: action gradients to LoRA, 8-hour cap ==="
run_with_retry "ARM_A" "$ARM_A" 8 on

echo "=== [3/3] arm B: action gradients detached, 3-hour cap ==="
run_with_retry "ARM_B" "$ARM_B" 3 off

touch "$RUNS/COSMOS_LORA_COTRAIN_COMPLETE"
echo "ALL_DONE"
