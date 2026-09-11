#!/usr/bin/env bash
# ==============================================================================
# Robust Overnight Training and Evaluation Queue (Protocol 1.0)
# Priority: Cosmos 2B (T=2 undistilled -> T=16 distillation), then FLUX.2 & 14B
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
MASTER_LOG="$LOGS_DIR/overnight_robust_queue_${TIMESTAMP}.log"
LATEST_LOG="$LOGS_DIR/overnight_robust_queue_latest.log"
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
printf "[%s] STARTING COSMOS 2B FIRST QUEUE (Protocol 1.0)\n" "$(log_time)"
printf "[%s] Master Log: %s\n" "$(log_time)" "$MASTER_LOG"
printf "[%s] ====================================================================\n" "$(log_time)"

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

MIMIC_DIR="/home/anton/.cache/video-vam/mimic-video-f2833903"
COSMOS2B_PT="$MIMIC_DIR/video_backbone/v2w_pretrained_cosmos.pt"
COSMOS2B_TOKENIZER="$MIMIC_DIR/video_backbone/tokenizer/tokenizer.pth"
COSMOS2B_VIDEO_LORA="$REPO_ROOT/outputs/train/cosmos-video-lora-step6000/best_lora.safetensors"

TRAIN_EPISODES=($(seq 0 31))
VAL_EPISODES=($(seq 32 39))

# ==============================================================================
# PRIORITY 1: COSMOS 2B T=2 UNDISTILLED BASELINE
# ==============================================================================
FEAT_T2_UNPOOLED="$OUTPUTS_FEAT/cosmos2b_videolora_t2_unpooled"
RUN_COSMOS2B_T2_UNDISTILLED="$OUTPUTS_TRAIN/cosmos2b-t2-undistilled-smolexpert"

run_stage "1A-train" "Cosmos 2B T=2 Undistilled Train Cache Extraction" \
    "$FEAT_T2_UNPOOLED/train/manifest.json" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/build_cosmos_feature_cache.py \
        --root "$DATASET_ROOT" \
        --episodes ${TRAIN_EPISODES[@]} \
        --stride 3 \
        --state-t 2 \
        --context-transform none \
        --vae-input-mode observed_prefix \
        --sigma 80 \
        --seed 0 \
        --checkpoint "$COSMOS2B_PT" \
        --tokenizer "$COSMOS2B_TOKENIZER" \
        --lora-weights "$COSMOS2B_VIDEO_LORA" \
        --output-dir "$FEAT_T2_UNPOOLED/train" \
        --resume

run_stage "1A-val" "Cosmos 2B T=2 Undistilled Val Cache Extraction" \
    "$FEAT_T2_UNPOOLED/val/manifest.json" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/build_cosmos_feature_cache.py \
        --root "$DATASET_ROOT" \
        --episodes ${VAL_EPISODES[@]} \
        --stride 20 \
        --state-t 2 \
        --context-transform none \
        --vae-input-mode observed_prefix \
        --sigma 80 \
        --seed 0 \
        --checkpoint "$COSMOS2B_PT" \
        --tokenizer "$COSMOS2B_TOKENIZER" \
        --lora-weights "$COSMOS2B_VIDEO_LORA" \
        --output-dir "$FEAT_T2_UNPOOLED/val" \
        --resume

run_stage "1B" "Cosmos 2B T=2 Undistilled SmolExpert Training" \
    "$RUN_COSMOS2B_T2_UNDISTILLED/best.safetensors" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/train_smolexpert.py \
        --backbone cosmos \
        --train-manifest "$FEAT_T2_UNPOOLED/train/manifest.json" \
        --val-manifest "$FEAT_T2_UNPOOLED/val/manifest.json" \
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
        --output-dir "$RUN_COSMOS2B_T2_UNDISTILLED" \
        --no-wandb

# ==============================================================================
# PRIORITY 2: COSMOS 2B T=16 -> T=2 DIRECT REPRESENTATION DISTILLATION
# ==============================================================================
FEAT_TEACHER_CONDFRAMES="$OUTPUTS_FEAT/cosmos2b_teacher_condframes_unpooled"
RUN_DISTILLED_LORA="$OUTPUTS_TRAIN/cosmos2b-t2-direct-distilled-lora"
FEAT_T2_DISTILLED="$OUTPUTS_FEAT/cosmos2b_distilled_t2_unpooled"
RUN_COSMOS2B_T2_DISTILLED="$OUTPUTS_TRAIN/cosmos2b-t2-distilled-smolexpert"

run_stage "2A-train" "Cosmos 2B T=16 Teacher cond_frames Train Extraction" \
    "$FEAT_TEACHER_CONDFRAMES/train/manifest.json" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/build_cosmos_feature_cache.py \
        --root "$DATASET_ROOT" \
        --episodes ${TRAIN_EPISODES[@]} \
        --stride 3 \
        --state-t 16 \
        --context-transform cond_frames \
        --vae-input-mode observed_prefix \
        --sigma 80 \
        --seed 0 \
        --checkpoint "$COSMOS2B_PT" \
        --tokenizer "$COSMOS2B_TOKENIZER" \
        --lora-weights "$COSMOS2B_VIDEO_LORA" \
        --output-dir "$FEAT_TEACHER_CONDFRAMES/train" \
        --resume

run_stage "2A-val" "Cosmos 2B T=16 Teacher cond_frames Val Extraction" \
    "$FEAT_TEACHER_CONDFRAMES/val/manifest.json" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/build_cosmos_feature_cache.py \
        --root "$DATASET_ROOT" \
        --episodes ${VAL_EPISODES[@]} \
        --stride 20 \
        --state-t 16 \
        --context-transform cond_frames \
        --vae-input-mode observed_prefix \
        --sigma 80 \
        --seed 0 \
        --checkpoint "$COSMOS2B_PT" \
        --tokenizer "$COSMOS2B_TOKENIZER" \
        --lora-weights "$COSMOS2B_VIDEO_LORA" \
        --output-dir "$FEAT_TEACHER_CONDFRAMES/val" \
        --resume

run_stage "2B" "Cosmos 2B T=2 Direct Manifold Distillation Training" \
    "$RUN_DISTILLED_LORA/best_lora.safetensors" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/train_cosmos_t2_distillation.py \
        --train-manifest "$FEAT_TEACHER_CONDFRAMES/train/manifest.json" \
        --val-manifest "$FEAT_TEACHER_CONDFRAMES/val/manifest.json" \
        --dataset-root "$DATASET_ROOT" \
        --backbone-checkpoint "$COSMOS2B_PT" \
        --tokenizer "$COSMOS2B_TOKENIZER" \
        --merge-lora-weights "$COSMOS2B_VIDEO_LORA" \
        --lora-rank 16 \
        --lora-alpha 16.0 \
        --lora-lr 1e-4 \
        --max-steps 15000 \
        --val-every 500 \
        --patience 12 \
        --device cuda \
        --output-dir "$RUN_DISTILLED_LORA" \
        --no-wandb

run_stage "2C-train" "Cosmos 2B Distilled T=2 Train Feature Extraction" \
    "$FEAT_T2_DISTILLED/train/manifest.json" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/build_cosmos_feature_cache.py \
        --root "$DATASET_ROOT" \
        --episodes ${TRAIN_EPISODES[@]} \
        --stride 3 \
        --state-t 2 \
        --context-transform none \
        --vae-input-mode observed_prefix \
        --sigma 80 \
        --seed 0 \
        --checkpoint "$COSMOS2B_PT" \
        --tokenizer "$COSMOS2B_TOKENIZER" \
        --merge-lora-weights "$COSMOS2B_VIDEO_LORA" \
        --lora-weights "$RUN_DISTILLED_LORA/best_lora.safetensors" \
        --lora-blocks 0-19 \
        --output-dir "$FEAT_T2_DISTILLED/train" \
        --resume

run_stage "2C-val" "Cosmos 2B Distilled T=2 Val Feature Extraction" \
    "$FEAT_T2_DISTILLED/val/manifest.json" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/build_cosmos_feature_cache.py \
        --root "$DATASET_ROOT" \
        --episodes ${VAL_EPISODES[@]} \
        --stride 20 \
        --state-t 2 \
        --context-transform none \
        --vae-input-mode observed_prefix \
        --sigma 80 \
        --seed 0 \
        --checkpoint "$COSMOS2B_PT" \
        --tokenizer "$COSMOS2B_TOKENIZER" \
        --merge-lora-weights "$COSMOS2B_VIDEO_LORA" \
        --lora-weights "$RUN_DISTILLED_LORA/best_lora.safetensors" \
        --lora-blocks 0-19 \
        --output-dir "$FEAT_T2_DISTILLED/val" \
        --resume

run_stage "2D" "Cosmos 2B Distilled T=2 SmolExpert Training" \
    "$RUN_COSMOS2B_T2_DISTILLED/best.safetensors" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/train_smolexpert.py \
        --backbone cosmos \
        --train-manifest "$FEAT_T2_DISTILLED/train/manifest.json" \
        --val-manifest "$FEAT_T2_DISTILLED/val/manifest.json" \
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
        --output-dir "$RUN_COSMOS2B_T2_DISTILLED" \
        --no-wandb

# Retrying Stage 1B now that manifest hash has been repaired:
run_stage "1B-retry" "Cosmos 2B T=2 Undistilled SmolExpert Training" \
    "/best.safetensors" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/train_smolexpert.py \
        --backbone cosmos \
        --train-manifest "/train/manifest.json" \
        --val-manifest "/val/manifest.json" \
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
        --output-dir "" \
        --no-wandb

# ==============================================================================
# PRIORITY 3: FLUX.2 [klein] PROTOCOL 1.0
# ==============================================================================
CACHE_FLUX2="$CACHE_ROOT/flux2-klein-protocol1-cache"
RUN_FLUX2="$OUTPUTS_TRAIN/flux2-klein-protocol1-smolexpert"
run_stage "3" "FLUX.2 klein Protocol 1.0 SmolExpert Training" \
    "$RUN_FLUX2/best.safetensors" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/train_smolexpert.py \
        --backbone flux2_klein \
        --train-manifest "$CACHE_FLUX2/train/manifest.json" \
        --val-manifest "$CACHE_FLUX2/val/manifest.json" \
        --output-dir "$RUN_FLUX2" \
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
# PRIORITY 4: COSMOS 14B PROTOCOL 1.0
# ==============================================================================
CACHE_COSMOS14B="$CACHE_ROOT/cosmos14b-protocol1-cache"
RUN_COSMOS14B="$OUTPUTS_TRAIN/cosmos14b-protocol1-smolexpert"
run_stage "4" "Cosmos 14B Protocol 1.0 SmolExpert Training" \
    "$RUN_COSMOS14B/best.safetensors" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/train_smolexpert.py \
        --backbone cosmos14b \
        --train-manifest "$CACHE_COSMOS14B/train/manifest.json" \
        --val-manifest "$CACHE_COSMOS14B/val/manifest.json" \
        --output-dir "$RUN_COSMOS14B" \
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
# PROTOCOL 1.1 DUAL-EVALUATION BENCHMARK (V1 vs V2 across all models)
# ==============================================================================
run_stage "EVAL-DUAL" "Protocol 1.1 Dual-Evaluation Benchmark (V1 vs V2)" \
    "/protocol1_1_dual_eval_matrix.json" \
    env PYTHONPATH=".:src" uv run python scripts/video_vam/run_protocol1_1_dual_eval.py --device cuda

# ==============================================================================
# SUMMARY REPORT
# ==============================================================================
printf "\n[%s] ====================================================================\n" "$(log_time)"
printf "[%s] ALL QUEUE STAGES COMPLETED! COMPILING SUMMARY REPORT\n" "$(log_time)"
printf "[%s] ====================================================================\n" "$(log_time)"

SUMMARY_FILE="$OUTPUTS_EVAL/protocol1_overnight_results_${TIMESTAMP}.json"
python3 -c "
import json
from pathlib import Path

runs = [
    ('cosmos2b_t2_undistilled', '$RUN_COSMOS2B_T2_UNDISTILLED'),
    ('cosmos2b_t2_distilled', '$RUN_COSMOS2B_T2_DISTILLED'),
    ('flux2_klein_protocol1', '$RUN_FLUX2'),
    ('cosmos14b_protocol1', '$RUN_COSMOS14B'),
]

summary = {'timestamp': '$TIMESTAMP', 'runs': {}}
for name, path_str in runs:
    p = Path(path_str)
    metrics_file = p / 'best_metrics.json'
    if metrics_file.is_file():
        try:
            summary['runs'][name] = json.loads(metrics_file.read_text())
            summary['runs'][name]['status'] = 'COMPLETED'
        except Exception as e:
            summary['runs'][name] = {'status': 'ERROR_READING_METRICS', 'error': str(e)}
    else:
        summary['runs'][name] = {'status': 'NO_METRICS_FOUND'}

with open('$SUMMARY_FILE', 'w') as f:
    json.dump(summary, f, indent=2)
print('Summary saved to $SUMMARY_FILE')
"

ln -sf "$SUMMARY_FILE" "$OUTPUTS_EVAL/protocol1_overnight_results_latest.json"

for i in "${!STAGE_NAMES[@]}"; do
    printf "  - %-50s : %-15s (%ds)\n" "${STAGE_NAMES[$i]}" "${STAGE_STATUSES[$i]}" "${STAGE_DURATIONS[$i]}"
done

printf "\n[%s] Finished! Master log: %s\n" "$(log_time)" "$MASTER_LOG"
