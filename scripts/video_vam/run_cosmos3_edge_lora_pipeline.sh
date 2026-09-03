#!/usr/bin/env bash
# End-to-End Cosmos 3 Edge Video LoRA Adaptation & SmolExpert Policy Evaluation
# Stage 1: Train Cosmos 3 Edge Video LoRA on robot demonstration clips
# Stage 2: Extract adapted pure 600-token Layer-20 vision representations
# Stage 3: Train SmolExpert Action Policy on the adapted representations

set -Eeuo pipefail

REPO="/home/anton/lerobot-video-vam"
PYTHON="$REPO/.venv/bin/python"
CACHE="/home/anton/.cache/video-vam"
SPLIT="$CACHE/splits/rehearsal-stride20.json"
TRAIN_EPISODES=({0..31})
VAL_EPISODES=({32..39})

LORA_RUN_DIR="$CACHE/runs/cosmos3-edge-video-lora-20260903"
BEST_LORA="$LORA_RUN_DIR/best_lora.safetensors"

ADAPTED_TRAIN_CACHE="$CACHE/cosmos3-edge-adapted-train0-31-stride3"
ADAPTED_VAL_CACHE="$CACHE/cosmos3-edge-adapted-val32-39-stride20"

POLICY_RUN_DIR="$CACHE/runs/cosmos3-edge-adapted-smolexpert-20260903"
mkdir -p "$LORA_RUN_DIR" "$POLICY_RUN_DIR"

MASTER_LOG="$CACHE/runs/cosmos3-edge-adapted-pipeline-20260903.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

cd "$REPO"
# shellcheck source=/dev/null
source scripts/video_vam/cosmos_cuda_env.sh
# shellcheck source=/dev/null
source scripts/video_vam/gpu_lock.sh
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH=".:src"

timestamp() {
    date --iso-8601=seconds
}

printf "[%s] STARTING COSMOS 3 EDGE VIDEO LORA & POLICY PIPELINE
" "$(timestamp)"
acquire_gpu_lock cosmos3-edge-lora-pipeline

# -----------------------------------------------------------------------------
# STAGE 1: Train Video LoRA on Robot Demonstration Video Clips
# -----------------------------------------------------------------------------
printf "
[%s] === STAGE 1: Training Cosmos 3 Edge Video LoRA ===
" "$(timestamp)"
if [[ ! -f "$BEST_LORA" ]]; then
    "$PYTHON" "$REPO/scripts/video_vam/train_cosmos3_edge_video_lora.py"         --train-episodes "${TRAIN_EPISODES[@]}"         --val-episodes "${VAL_EPISODES[@]}"         --train-stride 3         --val-stride 20         --clip-frames 17         --rank 16         --alpha 16.0         --lr 1e-4         --max-steps 5000         --val-every 500         --patience 10         --output-dir "$LORA_RUN_DIR"
else
    printf "[%s] Video LoRA already exists: %s
" "$(timestamp)" "$BEST_LORA"
fi

# -----------------------------------------------------------------------------
# STAGE 2: Extract Adapted Pure 600-Token Layer-20 Vision Feature Caches
# -----------------------------------------------------------------------------
printf "
[%s] === STAGE 2: Extracting Adapted Pure 600-Token Feature Caches ===
" "$(timestamp)"
if [[ ! -f "$ADAPTED_TRAIN_CACHE/manifest.json" ]]; then
    "$PYTHON" "$REPO/scripts/video_vam/extract_cosmos3_edge_pure_vision.py"         --episodes "${TRAIN_EPISODES[@]}"         --stride 3         --lora-checkpoint "$BEST_LORA"         --output-dir "$ADAPTED_TRAIN_CACHE"
else
    printf "[%s] Adapted train cache already exists: %s
" "$(timestamp)" "$ADAPTED_TRAIN_CACHE"
fi

if [[ ! -f "$ADAPTED_VAL_CACHE/manifest.json" ]]; then
    "$PYTHON" "$REPO/scripts/video_vam/extract_cosmos3_edge_pure_vision.py"         --episodes "${VAL_EPISODES[@]}"         --stride 20         --lora-checkpoint "$BEST_LORA"         --output-dir "$ADAPTED_VAL_CACHE"
else
    printf "[%s] Adapted val cache already exists: %s
" "$(timestamp)" "$ADAPTED_VAL_CACHE"
fi

# -----------------------------------------------------------------------------
# STAGE 3: Train SmolExpert Action Policy on Adapted 600 Vision Tokens
# -----------------------------------------------------------------------------
printf "
[%s] === STAGE 3: Training SmolExpert on Adapted Vision Tokens -> %s ===
" "$(timestamp)" "$POLICY_RUN_DIR"
"$PYTHON" -m scripts.video_vam.train_smolexpert_on_cosmos     --manifest "$ADAPTED_TRAIN_CACHE/manifest.json"     --val-manifest "$ADAPTED_VAL_CACHE/manifest.json"     --split "$SPLIT"     --train-episodes "${TRAIN_EPISODES[@]}"     --output-dir "$POLICY_RUN_DIR/smolexpert"     --expert-checkpoint lerobot/smolvla_base     --batch-size 8     --max-steps 500000     --max-hours 2     --val-every 1000     --patience 10     --lr 1e-4     --warmup-steps 1000     --num-steps 10     --context-transform auto     --seed 0     --wandb-project video-vam-world2action     --run-name "cosmos3-edge-adapted-smolexpert-20260903"

date --iso-8601=seconds > "$POLICY_RUN_DIR/COMPLETE"
printf "
[%s] COSMOS 3 EDGE ADAPTED PIPELINE COMPLETED SUCCESSFULLY!
" "$(timestamp)"
