#!/usr/bin/env bash
# Complete Cosmos T=2 Pipeline:
# Stage 1: Wait for World Expert training (runs/cosmos-t2-we-20260829-v3) to complete.
# Stage 2: World Expert Previews & Action Policy Training (SmolExpert).
# Stage 3: Retrain Linear Readout baseline with active LoRA gradients.
# Stage 4: Linear Readout Previews & Action Policy Training (SmolExpert).

set -Eeuo pipefail

REPO="/home/anton/lerobot-video-vam"
PYTHON="$REPO/.venv/bin/python"
CACHE="/home/anton/.cache/video-vam"
DATASET_ROOT="$CACHE/cube-out-of-box-dataset"
CHECKPOINT="$CACHE/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
TOKENIZER="$CACHE/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
PROMPT="$CACHE/prompt-embeddings/cube-out-of-box-t5-11b.safetensors"
OLD_LORA="$CACHE/runs/cosmos2b-video-lora-20260828/paused-step6000/best_lora.safetensors"
TRAIN_MANIFEST="$CACHE/cosmos-videolora-train0-31-stride3-statet2-unpooled/manifest.json"
VAL_MANIFEST="$CACHE/cosmos-videolora-val32-39-stride20-statet2-unpooled/manifest.json"
SPLIT="$CACHE/splits/rehearsal-stride20.json"
TRAIN_EPISODES=({0..31})
VAL_EPISODES=({32..39})

MASTER_LOG="$CACHE/runs/master-pipeline-20260830.log"
mkdir -p "$CACHE/runs"
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

printf "[%s] STARTING MASTER PIPELINE\n" "$(timestamp)"

# -----------------------------------------------------------------------------
# STAGE 1: Wait for currently running World Expert training
# -----------------------------------------------------------------------------
WE_RUN_DIR="$CACHE/runs/cosmos-t2-we-20260829-v3"
WE_TRAIN_DIR="$WE_RUN_DIR/training"

printf "[%s] STAGE 1: Monitoring active World Expert training at %s\n" "$(timestamp)" "$WE_TRAIN_DIR"
while pgrep -f "train_cosmos_t2_world_expert.*cosmos-t2-we-20260829-v3" >/dev/null 2>&1; do
    sleep 30
done

# If the old tmux session is still holding the lock via an idle shell, kill that session.
if tmux has-session -t cosmos-t2-we-resume 2>/dev/null; then
    printf "[%s] Closing old cosmos-t2-we-resume session to release GPU lock\n" "$(timestamp)"
    tmux kill-session -t cosmos-t2-we-resume 2>/dev/null || true
fi
sleep 5

# -----------------------------------------------------------------------------
# STAGE 2: World Expert Previews & SmolExpert Action Policy Training
# -----------------------------------------------------------------------------
printf "[%s] STAGE 2: World Expert Previews & Action Policy Training\n" "$(timestamp)"
acquire_gpu_lock cosmos-t2-we-downstream

# Ensure best_lora.json sidecar exists
if [[ -f "$WE_TRAIN_DIR/best.json" && ! -f "$WE_TRAIN_DIR/best_lora.json" ]]; then
    ln -sfn "$WE_TRAIN_DIR/best.json" "$WE_TRAIN_DIR/best_lora.json"
fi

# Previews
mkdir -p "$WE_RUN_DIR/previews"
"$PYTHON" -m scripts.video_vam.preview_cosmos_t2_we_video_prediction \
    --run-dir "$WE_TRAIN_DIR" \
    --episode 0 \
    --frame-index 4 \
    --steps 35 \
    --seed 0 \
    --output-dir "$WE_RUN_DIR/previews" \
    --overwrite || printf "[%s] Warning: ep0 preview failed\n" "$(timestamp)"

"$PYTHON" -m scripts.video_vam.preview_cosmos_t2_we_video_prediction \
    --run-dir "$WE_TRAIN_DIR" \
    --episode 19 \
    --frame-index 2203 \
    --steps 35 \
    --seed 0 \
    --output-dir "$WE_RUN_DIR/previews" \
    --overwrite || printf "[%s] Warning: ep19 preview failed\n" "$(timestamp)"

# Build World Expert feature caches
WE_TRAIN_CACHE="$CACHE/cosmos-t2-we-train0-31-stride3-statet2-unpooled"
WE_VAL_CACHE="$CACHE/cosmos-t2-we-val32-39-stride20-statet2-unpooled"

build_cache() {
    local episodes="$1"
    local stride="$2"
    local output="$3"
    local lora_weights="$4"
    local mode=--overwrite
    if [[ -d "$output" ]] && compgen -G "$output/*.safetensors" > /dev/null; then
        mode=--resume
    fi
    printf "[%s] Building cache episodes=%s stride=%s out=%s mode=%s merge=%s lora=%s\n" \
        "$(timestamp)" "$episodes" "$stride" "$output" "$mode" "$OLD_LORA" "$lora_weights"
    "$PYTHON" -m scripts.video_vam.build_cosmos_feature_cache \
        --root "$DATASET_ROOT" \
        --episodes $episodes \
        --stride "$stride" \
        --prompt "$PROMPT" \
        --checkpoint "$CHECKPOINT" \
        --merge-lora-weights "$OLD_LORA" \
        --lora-weights "$lora_weights" \
        --lora-blocks 0-19 \
        --tokenizer "$TOKENIZER" \
        --output-dir "$output" \
        --sigma 80 \
        --seed 0 \
        --state-t 2 \
        --vae-input-mode observed_prefix \
        --context-transform none \
        "$mode"
}

build_cache "${TRAIN_EPISODES[*]}" 3 "$WE_TRAIN_CACHE" "$WE_TRAIN_DIR/best_lora.safetensors"
build_cache "${VAL_EPISODES[*]}" 20 "$WE_VAL_CACHE" "$WE_TRAIN_DIR/best_lora.safetensors"

# Train SmolExpert on World Expert features
WE_SMOLEXPERT_RUN="$CACHE/runs/cosmos-t2-we-smolexpert-20260830"
mkdir -p "$WE_SMOLEXPERT_RUN"
printf "[%s] Training SmolExpert on World Expert features -> %s\n" "$(timestamp)" "$WE_SMOLEXPERT_RUN"
"$PYTHON" -m scripts.video_vam.train_smolexpert_on_cosmos \
    --manifest "$WE_TRAIN_CACHE/manifest.json" \
    --val-manifest "$WE_VAL_CACHE/manifest.json" \
    --split "$SPLIT" \
    --train-episodes "${TRAIN_EPISODES[@]}" \
    --output-dir "$WE_SMOLEXPERT_RUN/smolexpert" \
    --expert-checkpoint lerobot/smolvla_base \
    --batch-size 8 \
    --max-steps 500000 \
    --max-hours 12 \
    --val-every 1000 \
    --patience 10 \
    --lr 1e-4 \
    --warmup-steps 1000 \
    --num-steps 10 \
    --context-transform none \
    --seed 0 \
    --wandb-project video-vam-world2action \
    --run-name "cosmos-t2-we-smolexpert-20260830"

date --iso-8601=seconds > "$WE_SMOLEXPERT_RUN/COMPLETE"

# -----------------------------------------------------------------------------
# STAGE 3: Retrain Linear Readout with active LoRA gradients
# -----------------------------------------------------------------------------
LINEAR_RUN_ROOT="$CACHE/runs/cosmos-t2-linear-20260830"
LINEAR_TRAIN_DIR="$LINEAR_RUN_ROOT/training"
mkdir -p "$LINEAR_TRAIN_DIR"

printf "[%s] STAGE 3: Retraining Linear Readout -> %s\n" "$(timestamp)" "$LINEAR_TRAIN_DIR"
"$PYTHON" -m scripts.video_vam.train_cosmos_t2_world_expert \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --dataset-root "$DATASET_ROOT" \
    --backbone-checkpoint "$CHECKPOINT" \
    --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" \
    --merge-lora-weights "$OLD_LORA" \
    --output-dir "$LINEAR_TRAIN_DIR" \
    --head linear \
    --max-hours 6 \
    --overwrite \
    --wandb-project video-vam-world2action \
    --run-name "cosmos-t2-linear-20260830"

# Previews for Linear Readout
mkdir -p "$LINEAR_RUN_ROOT/previews"
"$PYTHON" -m scripts.video_vam.preview_cosmos_t2_we_video_prediction \
    --run-dir "$LINEAR_TRAIN_DIR" \
    --episode 0 \
    --frame-index 4 \
    --steps 1 \
    --seed 0 \
    --output-dir "$LINEAR_RUN_ROOT/previews" \
    --overwrite || printf "[%s] Warning: linear ep0 preview failed\n" "$(timestamp)"

"$PYTHON" -m scripts.video_vam.preview_cosmos_t2_we_video_prediction \
    --run-dir "$LINEAR_TRAIN_DIR" \
    --episode 19 \
    --frame-index 2203 \
    --steps 1 \
    --seed 0 \
    --output-dir "$LINEAR_RUN_ROOT/previews" \
    --overwrite || printf "[%s] Warning: linear ep19 preview failed\n" "$(timestamp)"

# -----------------------------------------------------------------------------
# STAGE 4: Linear LoRA Feature Cache & SmolExpert Action Training
# -----------------------------------------------------------------------------
printf "[%s] STAGE 4: Linear LoRA Feature Cache & SmolExpert Action Training\n" "$(timestamp)"

# Ensure best_lora.json sidecar exists
if [[ -f "$LINEAR_TRAIN_DIR/best.json" && ! -f "$LINEAR_TRAIN_DIR/best_lora.json" ]]; then
    ln -sfn "$LINEAR_TRAIN_DIR/best.json" "$LINEAR_TRAIN_DIR/best_lora.json"
fi

LINEAR_TRAIN_CACHE="$CACHE/cosmos-t2-linear-train0-31-stride3-statet2-unpooled"
LINEAR_VAL_CACHE="$CACHE/cosmos-t2-linear-val32-39-stride20-statet2-unpooled"

build_cache "${TRAIN_EPISODES[*]}" 3 "$LINEAR_TRAIN_CACHE" "$LINEAR_TRAIN_DIR/best_lora.safetensors"
build_cache "${VAL_EPISODES[*]}" 20 "$LINEAR_VAL_CACHE" "$LINEAR_TRAIN_DIR/best_lora.safetensors"

LINEAR_SMOLEXPERT_RUN="$CACHE/runs/cosmos-t2-linear-smolexpert-20260830"
mkdir -p "$LINEAR_SMOLEXPERT_RUN"
printf "[%s] Training SmolExpert on Linear LoRA features -> %s\n" "$(timestamp)" "$LINEAR_SMOLEXPERT_RUN"
"$PYTHON" -m scripts.video_vam.train_smolexpert_on_cosmos \
    --manifest "$LINEAR_TRAIN_CACHE/manifest.json" \
    --val-manifest "$LINEAR_VAL_CACHE/manifest.json" \
    --split "$SPLIT" \
    --train-episodes "${TRAIN_EPISODES[@]}" \
    --output-dir "$LINEAR_SMOLEXPERT_RUN/smolexpert" \
    --expert-checkpoint lerobot/smolvla_base \
    --batch-size 8 \
    --max-steps 500000 \
    --max-hours 12 \
    --val-every 1000 \
    --patience 10 \
    --lr 1e-4 \
    --warmup-steps 1000 \
    --num-steps 10 \
    --context-transform none \
    --seed 0 \
    --wandb-project video-vam-world2action \
    --run-name "cosmos-t2-linear-smolexpert-20260830"

date --iso-8601=seconds > "$LINEAR_SMOLEXPERT_RUN/COMPLETE"

printf "[%s] ALL MASTER PIPELINE STAGES COMPLETED SUCCESSFULLY\n" "$(timestamp)"
