#!/usr/bin/env bash
# ==============================================================================
# COSMOS 2B SCALE-100 PIPELINE (T=2 UNDISTILLED & T=16 COND_FRAMES REFERENCE)
# ==============================================================================

set -Eeuo pipefail
REPO_ROOT="/home/anton/lerobot-video-vam"
cd "$REPO_ROOT"

source scripts/video_vam/cosmos_cuda_env.sh
export PYTHONPATH=".:src"
export PATH="/home/anton/.local/bin:$PATH"
PYTHON="/home/anton/lerobot-video-vam/.venv/bin/python"

V2_DATASET="/home/anton/.cache/huggingface/lerobot/hub/datasets--Orellius--cube_out_of_box_v2/snapshots/5d0325cc1412f4774223a0beb528958108814962"
COSMOS2B_PT="/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
COSMOS2B_TOKENIZER="/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
COSMOS2B_PROMPT="/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors"
COSMOS2B_BASE_LORA="/home/anton/lerobot-video-vam/outputs/train/cosmos-video-lora-step6000/best_lora.safetensors"

mkdir -p outputs/logs outputs/train outputs/features outputs/evaluation

LOG_FILE="outputs/logs/c2b_scale100_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "LAUNCHING COSMOS 2B SCALE-100 PIPELINE AT $(date)"
echo "Log file: $LOG_FILE"
echo "================================================================================"

TRAIN_EPS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
           40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 \
           70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89)
VAL_EPS=(32 33 34 35 36 37 38 39)
EVAL2_EPS=(90 91 92 93 94 95 96 97 98 99)

run_task() {
    local task_name="$1"
    shift
    echo ""
    echo "================================================================================"
    echo ">>> STARTING TASK: $task_name at $(date)"
    echo "================================================================================"
    set +e
    "$@"
    local status=$?
    set -e
    if [ $status -eq 0 ]; then
        echo ">>> TASK COMPLETED: $task_name at $(date)"
    else
        echo ">>> TASK FAILED (exit code $status): $task_name at $(date)"
    fi
    return 0
}

# ------------------------------------------------------------------------------
# 1. COSMOS 2B T=2 UNDISTILLED BASELINE (SCALE-100)
# ------------------------------------------------------------------------------
C2B_T2_FEAT="outputs/features/v2-cosmos2b-t2-undistilled"
C2B_T2_POLICY="outputs/train/v2-cosmos2b-t2-undistilled-smolexpert"
mkdir -p "$C2B_T2_FEAT/train" "$C2B_T2_FEAT/val" "$C2B_T2_FEAT/eval2"

run_task "Extract Cosmos 2B T=2 Undistilled Train Features (V2 82 eps)" \
    "$PYTHON" scripts/video_vam/build_cosmos_feature_cache.py \
    --dataset-repo-id Orellius/cube_out_of_box_v2 \
    --root "$V2_DATASET" \
    --episodes "${TRAIN_EPS[@]}" \
    --stride 3 \
    --state-t 2 \
    --checkpoint "$COSMOS2B_PT" \
    --tokenizer "$COSMOS2B_TOKENIZER" \
    --lora-weights "$COSMOS2B_BASE_LORA" \
    --output-dir "$C2B_T2_FEAT/train" \
    --resume

if [ -f "$C2B_T2_FEAT/train/manifest.json" ] && [ -f "$C2B_T2_FEAT/val/manifest.json" ] && [ -f "$C2B_T2_FEAT/eval2/manifest.json" ]; then
    run_task "Train SmolExpert on Cosmos 2B T=2 Undistilled (Scale-100 Dual-Eval)" \
        "$PYTHON" scripts/video_vam/train_smolexpert.py \
        --backbone cosmos \
        --protocol scale100 \
        --train-manifest "$C2B_T2_FEAT/train/manifest.json" \
        --val-manifest "$C2B_T2_FEAT/val/manifest.json" \
        --eval2-manifest "$C2B_T2_FEAT/eval2/manifest.json" \
        --output-dir "$C2B_T2_POLICY" \
        --batch-size 8 \
        --lr 1e-4 \
        --warmup-steps 1000 \
        --min-steps 15000 \
        --max-steps 50000 \
        --val-every 500 \
        --patience 20 \
        --seed 0 \
        --overwrite
fi

# ------------------------------------------------------------------------------
# 2. COSMOS 2B T=16 COND_FRAMES REFERENCE (SCALE-100)
# ------------------------------------------------------------------------------
C2B_T16_FEAT="outputs/features/v2-cosmos2b-t16-condframes"
C2B_T16_POLICY="outputs/train/v2-cosmos2b-t16-condframes-smolexpert"
mkdir -p "$C2B_T16_FEAT/train" "$C2B_T16_FEAT/val" "$C2B_T16_FEAT/eval2"

run_task "Extract Cosmos 2B T=16 cond_frames Train Features (V2 82 eps)" \
    "$PYTHON" scripts/video_vam/build_cosmos_feature_cache.py \
    --dataset-repo-id Orellius/cube_out_of_box_v2 \
    --root "$V2_DATASET" \
    --episodes "${TRAIN_EPS[@]}" \
    --stride 3 \
    --state-t 16 \
    --context-transform cond_frames \
    --checkpoint "$COSMOS2B_PT" \
    --tokenizer "$COSMOS2B_TOKENIZER" \
    --lora-weights "$COSMOS2B_BASE_LORA" \
    --output-dir "$C2B_T16_FEAT/train" \
    --resume

if [ -f "$C2B_T16_FEAT/train/manifest.json" ] && [ -f "$C2B_T16_FEAT/val/manifest.json" ] && [ -f "$C2B_T16_FEAT/eval2/manifest.json" ]; then
    run_task "Train SmolExpert on Cosmos 2B T=16 Reference (Scale-100 Dual-Eval)" \
        "$PYTHON" scripts/video_vam/train_smolexpert.py \
        --backbone cosmos \
        --protocol scale100 \
        --train-manifest "$C2B_T16_FEAT/train/manifest.json" \
        --val-manifest "$C2B_T16_FEAT/val/manifest.json" \
        --eval2-manifest "$C2B_T16_FEAT/eval2/manifest.json" \
        --output-dir "$C2B_T16_POLICY" \
        --batch-size 8 \
        --lr 1e-4 \
        --warmup-steps 1000 \
        --min-steps 15000 \
        --max-steps 50000 \
        --val-every 500 \
        --patience 20 \
        --seed 0 \
        --overwrite
fi

# ------------------------------------------------------------------------------
# 3. GENERATE DEPLOYMENT CONFIG WRAPPERS
# ------------------------------------------------------------------------------
run_task "Generate Deployment Config Wrappers" \
    "$PYTHON" -c "
import json
from pathlib import Path

REPO = Path('/home/anton/lerobot-video-vam')
base_cfg = {
    'type': 'video_vam', 'backend': 'cosmos', 'device': 'cuda',
    'input_features': {'observation.images.front': {'type': 'VISUAL', 'shape': [3, 480, 640]}, 'observation.state': {'type': 'STATE', 'shape': [6]}},
    'output_features': {'action': {'type': 'ACTION', 'shape': [6]}},
    'camera_key': 'observation.images.front', 'action_seed': 0,
    'cosmos_compile_friendly': False, 'cosmos_torch_compile': False,
}

c2b_t2 = dict(base_cfg)
c2b_t2.update({'cosmos_state_t': 2, 'cosmos_context_transform': 'none'})
p2 = REPO / 'outputs/train/v2-cosmos2b-t2-undistilled-smolexpert/config.json'
if p2.parent.is_dir(): p2.write_text(json.dumps(c2b_t2, indent=2))

c2b_t16 = dict(base_cfg)
c2b_t16.update({'cosmos_state_t': 16, 'cosmos_context_transform': 'none'})
p16 = REPO / 'outputs/train/v2-cosmos2b-t16-condframes-smolexpert/config.json'
if p16.parent.is_dir(): p16.write_text(json.dumps(c2b_t16, indent=2))

print('Deployment configs written successfully!')
"

echo ""
echo "================================================================================"
echo "COSMOS 2B SCALE-100 PIPELINE COMPLETED SUCCESSFULLY AT $(date)"
echo "================================================================================"
