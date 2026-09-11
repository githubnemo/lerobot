#!/usr/bin/env bash
# ==============================================================================
# Expanded 100-Episode Video-LoRA Dual-Evaluation Queue Runner (Phase 5 Scaling)
#
# Scales robot demonstration data from 40 to 100 episodes using Orellius/cube_out_of_box_v2:
#   - Train Split (82 episodes): Episodes 0-31 + 50 new demonstration episodes (40-89)
#   - Eval-Set 1 (Historical Benchmark, 8 episodes): Episodes 32-39 (strict Protocol 1.0, stride 20, 88 anchors)
#   - Eval-Set 2 (New Distribution Benchmark, 10 episodes): Episodes 90-99 (stride 20, 51 anchors)
#
# Sequential Execution Order:
#   STAGE 1: FLUX.2 [klein] Video-LoRA (Multi-Reference conditioning, 5,000 steps)
#     - 1.1: Train LoRA adapters (double blocks, rank 16, alpha 16) on train split
#     - 1.2: Extract adapted features on 100 episodes (train, val, eval2) using build_vam_feature_cache.py
#     - 1.3: Train SmolExpert on adapted features with dual-eval until convergence (min_steps=15000, max_steps=50000, patience=25)
#   STAGE 2: Cosmos 14B Quantized Video-LoRA (FP8 QLoRA, 4,000 steps)
#     - 2.1: Train QLoRA adapters (all 36 blocks, rank 16, alpha 16) on train split
#     - 2.2: Extract adapted features on 100 episodes (train, val, eval2) using build_vam_feature_cache.py
#     - 2.3: Train SmolExpert on adapted features with dual-eval until convergence (min_steps=20000, max_steps=60000, patience=25)
# ==============================================================================

set -Eeuo pipefail

REPO="/home/anton/lerobot-video-vam"
CACHE="/home/anton/.cache/video-vam"
LOG_DIR="$REPO/logs"
RUNS_DIR="$CACHE/runs"
MASTER_LOG="$RUNS_DIR/scale100_videolora_pipeline.log"

mkdir -p "$LOG_DIR" "$RUNS_DIR"
exec > >(tee -a "$MASTER_LOG") 2>&1

cd "$REPO"
# shellcheck source=/dev/null
source scripts/video_vam/cosmos_cuda_env.sh
# shellcheck source=/dev/null
source scripts/video_vam/gpu_lock.sh

export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH=".:src:${PYTHONPATH:-}"
DEVICE="${DEVICE:-cuda:0}"

timestamp() {
    date --iso-8601=seconds
}

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] STARTING EXPANDED 100-EPISODE VIDEO-LORA DUAL-EVALUATION PIPELINE\n" "$(timestamp)"
printf "[%s] Master Log: %s\n" "$(timestamp)" "$MASTER_LOG"
printf "[%s] Device: %s\n" "$(timestamp)" "$DEVICE"
printf "[%s] Dataset: Orellius/cube_out_of_box_v2 (100 episodes)\n" "$(timestamp)"
printf "[%s] Train Split: Episodes 0-31 + 40-89 (82 eps, stride 3, 3065 samples)\n" "$(timestamp)"
printf "[%s] Eval-Set 1 (Historical): Episodes 32-39 (8 eps, stride 20, 88 anchors)\n" "$(timestamp)"
printf "[%s] Eval-Set 2 (New): Episodes 90-99 (10 eps, stride 20, 51 anchors)\n" "$(timestamp)"
printf "[%s] ====================================================================\n" "$(timestamp)"

acquire_gpu_lock scale100-videolora-queue

# ==============================================================================
# STAGE 1: FLUX.2 [klein] Video-LoRA Pipeline (100 Episodes)
# ==============================================================================
FLUX2_LORA_RUN_DIR="$RUNS_DIR/flux2-klein-videolora-scale100"
FLUX2_LORA_FILE="$FLUX2_LORA_RUN_DIR/flux2_klein_lora.safetensors"
FLUX2_ADAPTED_CACHE="$CACHE/flux2-klein-scale100-videolora-cache"
FLUX2_SMOL_RUN_DIR="$RUNS_DIR/flux2-klein-scale100-videolora-smolexpert"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 1: FLUX.2 [klein] Video-LoRA Pipeline (100 Episodes) ===\n" "$(timestamp)"
printf "[%s] LoRA Output Dir:   %s\n" "$(timestamp)" "$FLUX2_LORA_RUN_DIR"
printf "[%s] Adapted Cache Dir: %s\n" "$(timestamp)" "$FLUX2_ADAPTED_CACHE"
printf "[%s] SmolExpert Run:    %s\n" "$(timestamp)" "$FLUX2_SMOL_RUN_DIR"
printf "[%s] ====================================================================\n" "$(timestamp)"

# 1.1 Train FLUX.2 klein Video-LoRA
if [[ ! -f "$FLUX2_LORA_FILE" ]]; then
    printf "[%s] [Stage 1.1] Training FLUX.2 klein Video-LoRA (5,000 steps)...\n" "$(timestamp)"
    uv run python scripts/video_vam/train_flux2_klein_lora.py \
        --output-dir "$FLUX2_LORA_RUN_DIR" \
        --dataset-repo-id Orellius/cube_out_of_box_v2 \
        --train-episodes "0-31, 40-89" \
        --lora-rank 16 \
        --lora-alpha 16.0 \
        --device "$DEVICE" \
        --dtype bfloat16 \
        --max-steps 5000 \
        --warmup-steps 250
    printf "[%s] [Stage 1.1] Video-LoRA weights successfully saved to %s\n" "$(timestamp)" "$FLUX2_LORA_FILE"
else
    printf "[%s] [Stage 1.1] Found existing Video-LoRA weights at %s; skipping training.\n" "$(timestamp)" "$FLUX2_LORA_FILE"
fi

# 1.2 Extract adapted features on 100 episodes
if [[ ! -f "$FLUX2_ADAPTED_CACHE/train/manifest.json" || ! -f "$FLUX2_ADAPTED_CACHE/eval2/manifest.json" ]]; then
    printf "[%s] [Stage 1.2] Extracting adapted FLUX.2 klein features across 100 episodes...\n" "$(timestamp)"
    uv run python scripts/video_vam/build_vam_feature_cache.py \
        --backbone flux2_klein \
        --lora-weights "$FLUX2_LORA_FILE" \
        --output-dir "$FLUX2_ADAPTED_CACHE" \
        --dataset-repo-id Orellius/cube_out_of_box_v2 \
        --train-episodes "0-31, 40-89" \
        --val-episodes "32-39" \
        --eval2-episodes "90-99" \
        --train-stride 3 \
        --val-stride 20 \
        --eval2-stride 20 \
        --batch-size 8 \
        --device "$DEVICE" \
        --dtype bfloat16 \
        --resume
    printf "[%s] [Stage 1.2] Feature extraction completed.\n" "$(timestamp)"
else
    printf "[%s] [Stage 1.2] Found existing adapted feature cache at %s.\n" "$(timestamp)" "$FLUX2_ADAPTED_CACHE"
fi

# 1.3 Train SmolExpert to convergence
if [[ ! -f "$FLUX2_SMOL_RUN_DIR/final_dual_eval_metrics.json" ]]; then
    printf "[%s] [Stage 1.3] Training SmolExpert on FLUX.2 klein Video-LoRA features...\n" "$(timestamp)"
    uv run python scripts/video_vam/train_smolexpert.py \
        --train-manifest "$FLUX2_ADAPTED_CACHE/train/manifest.json" \
        --val-manifest "$FLUX2_ADAPTED_CACHE/val/manifest.json" \
        --eval2-manifest "$FLUX2_ADAPTED_CACHE/eval2/manifest.json" \
        --output-dir "$FLUX2_SMOL_RUN_DIR" \
        --backbone flux2_klein \
        --batch-size 8 \
        --lr 1e-4 \
        --lr-scheduler cosine \
        --warmup-steps 1000 \
        --min-steps 15000 \
        --max-steps 50000 \
        --val-every 500 \
        --patience 25 \
        --device "$DEVICE" \
        --no-wandb
    printf "[%s] STAGE 1 COMPLETED.\n" "$(timestamp)"
    uv run python scripts/video_vam/update_scale100_leaderboard.py
else
    printf "[%s] STAGE 1 already completed: %s/final_dual_eval_metrics.json found.\n" "$(timestamp)" "$FLUX2_SMOL_RUN_DIR"
fi

# ==============================================================================
# STAGE 2: Cosmos 14B Quantized Video-LoRA Pipeline (100 Episodes)
# ==============================================================================
COSMOS14B_LORA_RUN_DIR="$RUNS_DIR/cosmos14b-videolora-scale100"
COSMOS14B_LORA_FILE="$COSMOS14B_LORA_RUN_DIR/cosmos14b_video_lora.safetensors"
COSMOS14B_ADAPTED_CACHE="$CACHE/cosmos14b-scale100-videolora-cache"
COSMOS14B_SMOL_RUN_DIR="$RUNS_DIR/cosmos14b-scale100-videolora-smolexpert"

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] === STAGE 2: Cosmos 14B Quantized Video-LoRA Pipeline (100 Episodes) ===\n" "$(timestamp)"
printf "[%s] LoRA Output Dir:   %s\n" "$(timestamp)" "$COSMOS14B_LORA_RUN_DIR"
printf "[%s] Adapted Cache Dir: %s\n" "$(timestamp)" "$COSMOS14B_ADAPTED_CACHE"
printf "[%s] SmolExpert Run:    %s\n" "$(timestamp)" "$COSMOS14B_SMOL_RUN_DIR"
printf "[%s] ====================================================================\n" "$(timestamp)"

# 2.1 Train Cosmos 14B QLoRA
if [[ ! -f "$COSMOS14B_LORA_FILE" ]]; then
    printf "[%s] [Stage 2.1] Training Cosmos 14B QLoRA (4,000 steps)...\n" "$(timestamp)"
    uv run python scripts/video_vam/train_cosmos14b_video_lora.py \
        --checkpoint-path "$CACHE/cosmos-14b" \
        --output-dir "$COSMOS14B_LORA_RUN_DIR" \
        --dataset-repo-id Orellius/cube_out_of_box_v2 \
        --train-episodes "0-31, 40-89" \
        --lora-rank 16 \
        --lora-alpha 16.0 \
        --target-blocks "all" \
        --target-modules "to_q,to_k,to_v,to_out.0" \
        --lr 1e-4 \
        --max-steps 4000 \
        --warmup-steps 200 \
        --batch-size 1 \
        --device "$DEVICE" \
        --dtype bfloat16
    printf "[%s] [Stage 2.1] Cosmos 14B QLoRA weights successfully saved to %s\n" "$(timestamp)" "$COSMOS14B_LORA_FILE"
else
    printf "[%s] [Stage 2.1] Found existing Cosmos 14B QLoRA weights at %s; skipping training.\n" "$(timestamp)" "$COSMOS14B_LORA_FILE"
fi

# 2.2 Extract adapted features on 100 episodes
if [[ ! -f "$COSMOS14B_ADAPTED_CACHE/train/manifest.json" || ! -f "$COSMOS14B_ADAPTED_CACHE/eval2/manifest.json" ]]; then
    printf "[%s] [Stage 2.2] Extracting adapted Cosmos 14B features across 100 episodes...\n" "$(timestamp)"
    uv run python scripts/video_vam/build_vam_feature_cache.py \
        --backbone cosmos14b \
        --lora-weights "$COSMOS14B_LORA_FILE" \
        --output-dir "$COSMOS14B_ADAPTED_CACHE" \
        --dataset-repo-id Orellius/cube_out_of_box_v2 \
        --train-episodes "0-31, 40-89" \
        --val-episodes "32-39" \
        --eval2-episodes "90-99" \
        --train-stride 3 \
        --val-stride 20 \
        --eval2-stride 20 \
        --batch-size 16 \
        --device "$DEVICE" \
        --dtype bfloat16 \
        --resume
    printf "[%s] [Stage 2.2] Feature extraction completed.\n" "$(timestamp)"
else
    printf "[%s] [Stage 2.2] Found existing adapted feature cache at %s.\n" "$(timestamp)" "$COSMOS14B_ADAPTED_CACHE"
fi

# 2.3 Train SmolExpert to convergence
if [[ ! -f "$COSMOS14B_SMOL_RUN_DIR/final_dual_eval_metrics.json" ]]; then
    printf "[%s] [Stage 2.3] Training SmolExpert on Cosmos 14B Video-LoRA features...\n" "$(timestamp)"
    uv run python scripts/video_vam/train_smolexpert.py \
        --train-manifest "$COSMOS14B_ADAPTED_CACHE/train/manifest.json" \
        --val-manifest "$COSMOS14B_ADAPTED_CACHE/val/manifest.json" \
        --eval2-manifest "$COSMOS14B_ADAPTED_CACHE/eval2/manifest.json" \
        --output-dir "$COSMOS14B_SMOL_RUN_DIR" \
        --backbone cosmos14b \
        --batch-size 8 \
        --lr 1e-4 \
        --lr-scheduler cosine \
        --warmup-steps 1000 \
        --min-steps 20000 \
        --max-steps 60000 \
        --val-every 500 \
        --patience 25 \
        --device "$DEVICE" \
        --no-wandb
    printf "[%s] STAGE 2 COMPLETED.\n" "$(timestamp)"
    uv run python scripts/video_vam/update_scale100_leaderboard.py
else
    printf "[%s] STAGE 2 already completed: %s/final_dual_eval_metrics.json found.\n" "$(timestamp)" "$COSMOS14B_SMOL_RUN_DIR"
fi

printf "\n[%s] ====================================================================\n" "$(timestamp)"
printf "[%s] ALL 100-EPISODE VIDEO-LORA BENCHMARKS FINISHED SUCCESSFULLY!\n" "$(timestamp)"
printf "[%s] Master Log: %s\n" "$(timestamp)" "$MASTER_LOG"
printf "[%s] ====================================================================\n" "$(timestamp)"

uv run python scripts/video_vam/update_scale100_leaderboard.py
