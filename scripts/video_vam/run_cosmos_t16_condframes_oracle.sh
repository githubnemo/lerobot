#!/usr/bin/env bash
# Train SmolExpert directly on Teacher T=16 first 2 latent frames (cond_frames, 2,400 tokens)
# This establishes the exact Oracle Ceiling for T=2 representation distillation.

set -Eeuo pipefail

REPO="/home/anton/lerobot-video-vam"
PYTHON="$REPO/.venv/bin/python"
CACHE="/home/anton/.cache/video-vam"
DATASET_ROOT="$CACHE/cube-out-of-box-dataset"
SPLIT="$CACHE/splits/rehearsal-stride20.json"
TRAIN_EPISODES=({0..31})
VAL_EPISODES=({32..39})

TEACHER_TRAIN_CACHE="$CACHE/cosmos-t16-teacher-train0-31-stride3-unpooled"
TEACHER_VAL_CACHE="$CACHE/cosmos-t16-teacher-val32-39-stride20-unpooled"

ORACLE_RUN_DIR="$CACHE/runs/cosmos-t16-condframes-oracle-smolexpert-20260902"
mkdir -p "$ORACLE_RUN_DIR"

MASTER_LOG="$CACHE/runs/oracle-condframes-pipeline-20260902.log"
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

printf "[%s] STARTING COSMOS T=16 COND_FRAMES ORACLE BENCHMARK\n" "$(timestamp)"
acquire_gpu_lock cosmos-oracle-condframes

printf "\n[%s] === Training SmolExpert on Teacher T=16 cond_frames (2,400 tokens) -> %s ===\n" "$(timestamp)" "$ORACLE_RUN_DIR"
"$PYTHON" -m scripts.video_vam.train_smolexpert_on_cosmos \
    --manifest "$TEACHER_TRAIN_CACHE/manifest.json" \
    --val-manifest "$TEACHER_VAL_CACHE/manifest.json" \
    --split "$SPLIT" \
    --train-episodes "${TRAIN_EPISODES[@]}" \
    --output-dir "$ORACLE_RUN_DIR/smolexpert" \
    --expert-checkpoint lerobot/smolvla_base \
    --batch-size 8 \
    --max-steps 500000 \
    --max-hours 2 \
    --val-every 1000 \
    --patience 10 \
    --lr 1e-4 \
    --warmup-steps 1000 \
    --num-steps 10 \
    --context-transform cond_frames \
    --seed 0 \
    --wandb-project video-vam-world2action \
    --run-name "cosmos-t16-condframes-oracle-smolexpert-20260902"

date --iso-8601=seconds > "$ORACLE_RUN_DIR/COMPLETE"
printf "\n[%s] ORACLE BENCHMARK COMPLETED SUCCESSFULLY!\n" "$(timestamp)"
