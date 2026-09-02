#!/usr/bin/env bash
# End-to-End Cosmos 3 Edge Benchmark Pipeline:
# Stage 1: Extract Cosmos3-Edge Feature Caches (Train & Val)
# Stage 2: Train SmolExpert Action Policy on Cosmos3-Edge Features
# Stage 3: Direct Representation Distillation to Fast T=2 LoRA

set -Eeuo pipefail

REPO="/home/anton/lerobot-video-vam"
PYTHON="$REPO/.venv/bin/python"
CACHE="/home/anton/.cache/video-vam"
DATASET_ROOT="$CACHE/cube-out-of-box-dataset"
SPLIT="$CACHE/splits/rehearsal-stride20.json"
TRAIN_EPISODES=({0..31})
VAL_EPISODES=({32..39})

TRAIN_CACHE="$CACHE/cosmos3-edge-train0-31-stride3"
VAL_CACHE="$CACHE/cosmos3-edge-val32-39-stride20"

RUN_DIR="$CACHE/runs/cosmos3-edge-smolexpert-20260902"
mkdir -p "$RUN_DIR"

MASTER_LOG="$CACHE/runs/cosmos3-edge-pipeline-20260902.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

cd "$REPO"
# shellcheck source=/dev/null
source scripts/video_vam/cosmos_cuda_env.sh
# shellcheck source=/dev/null
source scripts/video_vam/gpu_lock.sh
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

timestamp() {
    date --iso-8601=seconds
}

printf "[%s] STARTING COSMOS 3 EDGE PIPELINE\n" "$(timestamp)"
acquire_gpu_lock cosmos3-edge-pipeline

# -----------------------------------------------------------------------------
# STAGE 1: Extract Cosmos3-Edge Features
# -----------------------------------------------------------------------------
printf "\n[%s] === STAGE 1: Extracting Cosmos3-Edge Feature Caches ===\n" "$(timestamp)"
if [[ ! -f "$TRAIN_CACHE/manifest.json" ]]; then
    "$PYTHON" "$REPO/scripts/video_vam/extract_cosmos3_edge_features.py" \
        --episodes "${TRAIN_EPISODES[@]}" \
        --stride 3 \
        --output-dir "$TRAIN_CACHE"
else
    printf "[%s] Train cache already exists: %s\n" "$(timestamp)" "$TRAIN_CACHE"
fi

if [[ ! -f "$VAL_CACHE/manifest.json" ]]; then
    "$PYTHON" "$REPO/scripts/video_vam/extract_cosmos3_edge_features.py" \
        --episodes "${VAL_EPISODES[@]}" \
        --stride 20 \
        --output-dir "$VAL_CACHE"
else
    printf "[%s] Val cache already exists: %s\n" "$(timestamp)" "$VAL_CACHE"
fi

# -----------------------------------------------------------------------------
# STAGE 2: Train SmolExpert Action Policy on Cosmos3-Edge Features
# -----------------------------------------------------------------------------
printf "\n[%s] === STAGE 2: Training SmolExpert on Cosmos3-Edge Features -> %s ===\n" "$(timestamp)" "$RUN_DIR"
"$PYTHON" -m scripts.video_vam.train_smolexpert_on_cosmos \
    --manifest "$TRAIN_CACHE/manifest.json" \
    --val-manifest "$VAL_CACHE/manifest.json" \
    --split "$SPLIT" \
    --train-episodes "${TRAIN_EPISODES[@]}" \
    --output-dir "$RUN_DIR/smolexpert" \
    --expert-checkpoint lerobot/smolvla_base \
    --batch-size 8 \
    --max-steps 500000 \
    --max-hours 2 \
    --val-every 1000 \
    --patience 10 \
    --lr 1e-4 \
    --warmup-steps 1000 \
    --num-steps 10 \
    --context-transform auto \
    --seed 0 \
    --wandb-project video-vam-world2action \
    --run-name "cosmos3-edge-smolexpert-20260902"

date --iso-8601=seconds > "$RUN_DIR/COMPLETE"
printf "\n[%s] COSMOS 3 EDGE PIPELINE COMPLETED SUCCESSFULLY!\n" "$(timestamp)"
