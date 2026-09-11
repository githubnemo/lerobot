#!/usr/bin/env bash
# ==============================================================================
# Robust Follow-Up Execution Queue:
#   1. Wait for currently running `cosmos2b-t16-condframes` training to finish.
#   2. Cosmos 3 Edge: Clean `und_seq` feature extraction (600 tokens) + SmolExpert training.
#   3. Multi-Layer Feature Distillation / Extraction:
#      - Cosmos 2B (Layers 14 + 20) -> SmolExpert
#      - Cosmos 7B (Layers 14 + 20) -> SmolExpert
#      - Cosmos 14B (Proportional Layers 18 + 30) -> SmolExpert
#
# Robustness Guarantee:
#   - Each stage is guarded by `run_stage`: failures log and do NOT crash the queue.
#   - Each stage checks its marker file and skips if already completed.
# ==============================================================================

set -uo pipefail

REPO_ROOT="/home/anton/lerobot-video-vam"
CACHE_ROOT="/home/anton/.cache/video-vam"
DATASET_ROOT="/home/anton/.cache/video-vam/cube-out-of-box-dataset"
LOGS_DIR="$REPO_ROOT/logs"
OUTPUTS_TRAIN="$REPO_ROOT/outputs/train"
OUTPUTS_FEAT="$REPO_ROOT/outputs/features"
OUTPUTS_EVAL="$REPO_ROOT/outputs/evaluation"

mkdir -p "$LOGS_DIR" "$OUTPUTS_TRAIN" "$OUTPUTS_FEAT" "$OUTPUTS_EVAL"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOGS_DIR/followup_queue_${TIMESTAMP}.log"
LATEST_LOG="$LOGS_DIR/followup_queue_latest.log"
ln -sf "$MASTER_LOG" "$LATEST_LOG"

exec > >(tee -a "$MASTER_LOG") 2>&1

cd "$REPO_ROOT"
export PATH="/home/anton/.local/bin:$PATH"
source scripts/video_vam/cosmos_cuda_env.sh
source scripts/video_vam/gpu_lock.sh

export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=".:src:${PYTHONPATH:-}"

log_time() {
    date --iso-8601=seconds
}

printf "\n[%s] ====================================================================\n" "$(log_time)"
printf "[%s] STARTING COSMOS 3 EDGE + MULTI-LAYER QUEUE (Protocol 1.0/1.1)\n" "$(log_time)"
printf "[%s] Master Log: %s\n" "$(log_time)" "$MASTER_LOG"
printf "[%s] ====================================================================\n" "$(log_time)"

# ------------------------------------------------------------------------------
# STEP 0: Wait for any currently running Cosmos 2B T=16 training to finish
# ------------------------------------------------------------------------------
printf "[%s] Checking if previous GPU training is active...\n" "$(log_time)"
while pgrep -f "cosmos2b-t16-condframes" >/dev/null 2>&1 || pgrep -f "train_smolexpert.*cosmos2b-t16" >/dev/null 2>&1; do
    printf "[%s] Waiting for running cosmos2b-t16-condframes to complete... (sleep 30s)\n" "$(log_time)"
    sleep 30
done
printf "[%s] GPU is clear for next queue tasks!\n" "$(log_time)"

declare -a STAGE_NAMES=()
declare -a STAGE_STATUSES=()
declare -a STAGE_DURATIONS=()

run_stage() {
    local stage_id="$1"
    local stage_name="$2"
    local marker_file="$3"
    shift 3
    local cmd=("$@")

    STAGE_NAMES+=("$stage_name")

    printf "\n[%s] --------------------------------------------------------------------\n" "$(log_time)"
    printf "[%s] [STAGE %s] %s\n" "$(log_time)" "$stage_id" "$stage_name"
    printf "[%s] --------------------------------------------------------------------\n" "$(log_time)"

    if [[ -n "$marker_file" && -f "$marker_file" ]]; then
        printf "[%s] [SKIP] Stage %s marker already exists: %s\n" "$(log_time)" "$stage_id" "$marker_file"
        STAGE_STATUSES+=("SKIPPED_EXISTING")
        STAGE_DURATIONS+=(0)
        return 0
    fi

    local start_ts
    start_ts=$(date +%s)
    printf "[%s] Executing: %s\n" "$(log_time)" "${cmd[*]}"

    if "${cmd[@]}"; then
        local elapsed=$(( $(date +%s) - start_ts ))
        printf "[%s] [SUCCESS] Stage %s completed in %ds\n" "$(log_time)" "$stage_id" "$elapsed"
        STAGE_STATUSES+=("SUCCESS")
        STAGE_DURATIONS+=("$elapsed")
        return 0
    else
        local exit_code=$?
        local elapsed=$(( $(date +%s) - start_ts ))
        printf "[%s] [FAILURE] Stage %s failed with exit code %d after %ds! Continuing to next stage...\n" \
            "$(log_time)" "$stage_id" "$exit_code" "$elapsed" >&2
        STAGE_STATUSES+=("FAILED(code=$exit_code)")
        STAGE_DURATIONS+=("$elapsed")
        return 0
    fi
}

TRAIN_EPISODES=($(seq 0 31))
VAL_EPISODES=($(seq 32 39))

# ==============================================================================
# PRIORITY 1: COSMOS 3 EDGE (Clean `und_seq` 600-token feature extraction & head)
# ==============================================================================
C3_FEAT_DIR="$OUTPUTS_FEAT/cosmos3_edge_undseq_pure"
C3_RUN_DIR="$OUTPUTS_TRAIN/cosmos3-edge-undseq-smolexpert"

run_stage "C3-train" "Cosmos 3 Edge und_seq Train Cache Extraction" \
    "$C3_FEAT_DIR/train/manifest.json" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/extract_cosmos3_edge_pure_vision.py \
        --episodes ${TRAIN_EPISODES[@]} \
        --stride 3 \
        --output-dir "$C3_FEAT_DIR/train"

run_stage "C3-val" "Cosmos 3 Edge und_seq Val Cache Extraction" \
    "$C3_FEAT_DIR/val/manifest.json" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/extract_cosmos3_edge_pure_vision.py \
        --episodes ${VAL_EPISODES[@]} \
        --stride 20 \
        --output-dir "$C3_FEAT_DIR/val"

run_stage "C3-head" "Cosmos 3 Edge und_seq SmolExpert Policy Training" \
    "$C3_RUN_DIR/best.safetensors" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/train_smolexpert.py \
        --backbone cosmos \
        --train-manifest "$C3_FEAT_DIR/train/manifest.json" \
        --val-manifest "$C3_FEAT_DIR/val/manifest.json" \
        --context-transform none \
        --batch-size 8 \
        --lr 1e-4 \
        --lr-scheduler cosine \
        --warmup-steps 1000 \
        --min-steps 10000 \
        --max-steps 40000 \
        --val-every 500 \
        --patience 15 \
        --device cuda \
        --output-dir "$C3_RUN_DIR" \
        --no-wandb

# ==============================================================================
# PRIORITY 2: COSMOS 2B MULTI-LAYER (Layers 14 + 20)
# ==============================================================================
MIMIC_DIR="/home/anton/.cache/video-vam/mimic-video-f2833903"
COSMOS2B_PT="$MIMIC_DIR/video_backbone/v2w_pretrained_cosmos.pt"
COSMOS2B_TOKENIZER="$MIMIC_DIR/video_backbone/tokenizer/tokenizer.pth"
COSMOS2B_VIDEO_LORA="$REPO_ROOT/outputs/train/cosmos-video-lora-step6000/best_lora.safetensors"
C2B_MULTI_FEAT="$OUTPUTS_FEAT/cosmos2b_layers14_20_unpooled"
C2B_MULTI_RUN="$OUTPUTS_TRAIN/cosmos2b-t2-layers14-20-smolexpert"

run_stage "C2B-M-train" "Cosmos 2B T=2 Layers 14+20 Train Cache Extraction" \
    "$C2B_MULTI_FEAT/train/manifest.json" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/build_cosmos_feature_cache.py \
        --root "$DATASET_ROOT" \
        --episodes ${TRAIN_EPISODES[@]} \
        --stride 3 \
        --state-t 2 \
        --hidden-layers "14,20" \
        --context-transform none \
        --vae-input-mode observed_prefix \
        --sigma 80 \
        --seed 0 \
        --checkpoint "$COSMOS2B_PT" \
        --tokenizer "$COSMOS2B_TOKENIZER" \
        --lora-weights "$COSMOS2B_VIDEO_LORA" \
        --output-dir "$C2B_MULTI_FEAT/train" \
        --resume

run_stage "C2B-M-val" "Cosmos 2B T=2 Layers 14+20 Val Cache Extraction" \
    "$C2B_MULTI_FEAT/val/manifest.json" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/build_cosmos_feature_cache.py \
        --root "$DATASET_ROOT" \
        --episodes ${VAL_EPISODES[@]} \
        --stride 20 \
        --state-t 2 \
        --hidden-layers "14,20" \
        --context-transform none \
        --vae-input-mode observed_prefix \
        --sigma 80 \
        --seed 0 \
        --checkpoint "$COSMOS2B_PT" \
        --tokenizer "$COSMOS2B_TOKENIZER" \
        --lora-weights "$COSMOS2B_VIDEO_LORA" \
        --output-dir "$C2B_MULTI_FEAT/val" \
        --resume

run_stage "C2B-M-head" "Cosmos 2B T=2 Layers 14+20 SmolExpert Training" \
    "$C2B_MULTI_RUN/best.safetensors" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/train_smolexpert.py \
        --backbone cosmos \
        --train-manifest "$C2B_MULTI_FEAT/train/manifest.json" \
        --val-manifest "$C2B_MULTI_FEAT/val/manifest.json" \
        --context-transform none \
        --batch-size 8 \
        --lr 1e-4 \
        --lr-scheduler cosine \
        --warmup-steps 1000 \
        --min-steps 10000 \
        --max-steps 40000 \
        --val-every 500 \
        --patience 15 \
        --device cuda \
        --output-dir "$C2B_MULTI_RUN" \
        --no-wandb

# ==============================================================================
# PRIORITY 3: COSMOS 7B MULTI-LAYER (Layers 14 + 20)
# ==============================================================================
# In Cosmos 7B (28 blocks), layers 14 and 20 are the standard multi-depth tap.
CACHE_COSMOS7B="$CACHE_ROOT/cosmos7b-protocol1-cache"
C7B_RUN="$OUTPUTS_TRAIN/cosmos7b-protocol1-smolexpert"

run_stage "C7B-train" "Cosmos 7B Layers 14+20 SmolExpert Training" \
    "$C7B_RUN/best.safetensors" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/train_smolexpert.py \
        --backbone cosmos7b \
        --train-manifest "$CACHE_COSMOS7B/train/manifest.json" \
        --val-manifest "$CACHE_COSMOS7B/val/manifest.json" \
        --output-dir "$C7B_RUN" \
        --batch-size 8 \
        --lr 1e-4 \
        --lr-scheduler cosine \
        --warmup-steps 1000 \
        --min-steps 10000 \
        --max-steps 40000 \
        --val-every 500 \
        --patience 15 \
        --device cuda \
        --no-wandb

# ==============================================================================
# PRIORITY 4: COSMOS 14B MULTI-LAYER (Proportional Layers 18 + 30)
# ==============================================================================
# In Cosmos 14B (42 blocks), layers 18 and 30 are the proportional taps (18/42 ≈ 14/28, 30/42 ≈ 20/28).
CACHE_COSMOS14B="$CACHE_ROOT/cosmos14b-protocol1-cache"
C14B_RUN="$OUTPUTS_TRAIN/cosmos14b-protocol1-smolexpert"

run_stage "C14B-train" "Cosmos 14B Layers 18+30 SmolExpert Training" \
    "$C14B_RUN/best.safetensors" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/train_smolexpert.py \
        --backbone cosmos14b \
        --train-manifest "$CACHE_COSMOS14B/train/manifest.json" \
        --val-manifest "$CACHE_COSMOS14B/val/manifest.json" \
        --output-dir "$C14B_RUN" \
        --batch-size 8 \
        --lr 1e-4 \
        --lr-scheduler cosine \
        --warmup-steps 1000 \
        --min-steps 10000 \
        --max-steps 40000 \
        --val-every 500 \
        --patience 15 \
        --device cuda \
        --no-wandb

printf "\n[%s] ====================================================================\n" "$(log_time)"
printf "[%s] ALL FOLLOW-UP STAGES FINISHED!\n" "$(log_time)"
printf "[%s] Master Log: %s\n" "$(log_time)" "$MASTER_LOG"
printf "[%s] ====================================================================\n" "$(log_time)"
