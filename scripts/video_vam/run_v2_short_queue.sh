#!/usr/bin/env bash
# ==============================================================================
# FAST V2 (SCALE-100) MODEL TRAINING QUEUE (ZERO DISTILLATION)
#
# Runs:
# 1. Cosmos 3 Edge 15k-LoRA: Extract V2 features -> Train SmolExpert (Dual-Eval)
# 2. Cosmos 2B T=2 Undistilled: Extract V2 features -> Train SmolExpert (Dual-Eval)
# 3. Cosmos 2B T=16 Reference: Extract V2 features -> Train SmolExpert (Dual-Eval)
# 4. Generate config.json deployment wrappers & update Scale-100 leaderboard
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
C3_15K_LORA="/home/anton/lerobot-video-vam/outputs/train/v2-cosmos3-edge-video-lora-15k/best_lora.safetensors"

mkdir -p outputs/logs outputs/train outputs/features outputs/evaluation

LOG_FILE="outputs/logs/v2_short_queue_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "LAUNCHING FAST V2 TRAINING QUEUE AT $(date)"
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

# ==============================================================================
# 1. COSMOS 3 EDGE (15K LORA POLICY ON SCALE-100)
# ==============================================================================
C3_15K_FEAT="outputs/features/v2-cosmos3-edge-lora-15k"
C3_15K_POLICY="outputs/train/v2-cosmos3-edge-lora-15k-smolexpert"

if [ ! -f "$C3_15K_FEAT/train/manifest.json" ]; then
    run_task "Extract Cosmos 3 Edge 15k Train Features (82 eps, stride 3)" \
        "$PYTHON" scripts/video_vam/extract_cosmos3_edge_pure_vision.py \
        --dataset-repo-id Orellius/cube_out_of_box_v2 \
        --dataset-root "$V2_DATASET" \
        --episodes "${TRAIN_EPS[@]}" \
        --stride 3 \
        --lora-checkpoint "$C3_15K_LORA" \
        --output-dir "$C3_15K_FEAT/train"
fi

if [ ! -f "$C3_15K_FEAT/val/manifest.json" ]; then
    run_task "Extract Cosmos 3 Edge 15k Val Features (8 eps, stride 20)" \
        "$PYTHON" scripts/video_vam/extract_cosmos3_edge_pure_vision.py \
        --dataset-repo-id Orellius/cube_out_of_box_v2 \
        --dataset-root "$V2_DATASET" \
        --episodes "${VAL_EPS[@]}" \
        --stride 20 \
        --lora-checkpoint "$C3_15K_LORA" \
        --output-dir "$C3_15K_FEAT/val"
fi

if [ ! -f "$C3_15K_FEAT/eval2/manifest.json" ]; then
    run_task "Extract Cosmos 3 Edge 15k Eval2 Features (10 eps, stride 20)" \
        "$PYTHON" scripts/video_vam/extract_cosmos3_edge_pure_vision.py \
        --dataset-repo-id Orellius/cube_out_of_box_v2 \
        --dataset-root "$V2_DATASET" \
        --episodes "${EVAL2_EPS[@]}" \
        --stride 20 \
        --lora-checkpoint "$C3_15K_LORA" \
        --output-dir "$C3_15K_FEAT/eval2"
fi

if [ -f "$C3_15K_FEAT/train/manifest.json" ] && [ -f "$C3_15K_FEAT/val/manifest.json" ] && [ -f "$C3_15K_FEAT/eval2/manifest.json" ]; then
    run_task "Train SmolExpert on Cosmos 3 Edge 15k LoRA (Scale-100 Dual-Eval)" \
        "$PYTHON" scripts/video_vam/train_smolexpert.py \
        --backbone cosmos3-edge \
        --protocol scale100 \
        --train-manifest "$C3_15K_FEAT/train/manifest.json" \
        --val-manifest "$C3_15K_FEAT/val/manifest.json" \
        --eval2-manifest "$C3_15K_FEAT/eval2/manifest.json" \
        --output-dir "$C3_15K_POLICY" \
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

# ==============================================================================
# 2. COSMOS 2B T=2 UNDISTILLED BASELINE ON SCALE-100
# ==============================================================================
C2B_T2_FEAT="outputs/features/v2-cosmos2b-t2-undistilled"
C2B_T2_POLICY="outputs/train/v2-cosmos2b-t2-undistilled-smolexpert"
mkdir -p "$C2B_T2_FEAT/train" "$C2B_T2_FEAT/val" "$C2B_T2_FEAT/eval2"

if [ ! -f "$C2B_T2_FEAT/train/manifest.json" ]; then
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
        --output-dir "$C2B_T2_FEAT/train"
fi

if [ ! -f "$C2B_T2_FEAT/val/manifest.json" ]; then
    run_task "Extract Cosmos 2B T=2 Undistilled Val Features (V2 8 eps)" \
        "$PYTHON" scripts/video_vam/build_cosmos_feature_cache.py \
        --dataset-repo-id Orellius/cube_out_of_box_v2 \
        --root "$V2_DATASET" \
        --episodes "${VAL_EPS[@]}" \
        --stride 20 \
        --state-t 2 \
        --checkpoint "$COSMOS2B_PT" \
        --tokenizer "$COSMOS2B_TOKENIZER" \
        --lora-weights "$COSMOS2B_BASE_LORA" \
        --output-dir "$C2B_T2_FEAT/val"
fi

if [ ! -f "$C2B_T2_FEAT/eval2/manifest.json" ]; then
    run_task "Extract Cosmos 2B T=2 Undistilled Eval2 Features (V2 10 eps)" \
        "$PYTHON" scripts/video_vam/build_cosmos_feature_cache.py \
        --dataset-repo-id Orellius/cube_out_of_box_v2 \
        --root "$V2_DATASET" \
        --episodes "${EVAL2_EPS[@]}" \
        --stride 20 \
        --state-t 2 \
        --checkpoint "$COSMOS2B_PT" \
        --tokenizer "$COSMOS2B_TOKENIZER" \
        --lora-weights "$COSMOS2B_BASE_LORA" \
        --output-dir "$C2B_T2_FEAT/eval2"
fi

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

# ==============================================================================
# 3. COSMOS 2B T=16 COND_FRAMES REFERENCE ON SCALE-100
# ==============================================================================
C2B_T16_FEAT="outputs/features/v2-cosmos2b-t16-condframes"
C2B_T16_POLICY="outputs/train/v2-cosmos2b-t16-condframes-smolexpert"
mkdir -p "$C2B_T16_FEAT/train" "$C2B_T16_FEAT/val" "$C2B_T16_FEAT/eval2"

if [ ! -f "$C2B_T16_FEAT/train/manifest.json" ]; then
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
        --output-dir "$C2B_T16_FEAT/train"
fi

if [ ! -f "$C2B_T16_FEAT/val/manifest.json" ]; then
    run_task "Extract Cosmos 2B T=16 cond_frames Val Features (V2 8 eps)" \
        "$PYTHON" scripts/video_vam/build_cosmos_feature_cache.py \
        --dataset-repo-id Orellius/cube_out_of_box_v2 \
        --root "$V2_DATASET" \
        --episodes "${VAL_EPS[@]}" \
        --stride 20 \
        --state-t 16 \
        --context-transform cond_frames \
        --checkpoint "$COSMOS2B_PT" \
        --tokenizer "$COSMOS2B_TOKENIZER" \
        --lora-weights "$COSMOS2B_BASE_LORA" \
        --output-dir "$C2B_T16_FEAT/val"
fi

if [ ! -f "$C2B_T16_FEAT/eval2/manifest.json" ]; then
    run_task "Extract Cosmos 2B T=16 cond_frames Eval2 Features (V2 10 eps)" \
        "$PYTHON" scripts/video_vam/build_cosmos_feature_cache.py \
        --dataset-repo-id Orellius/cube_out_of_box_v2 \
        --root "$V2_DATASET" \
        --episodes "${EVAL2_EPS[@]}" \
        --stride 20 \
        --state-t 16 \
        --context-transform cond_frames \
        --checkpoint "$COSMOS2B_PT" \
        --tokenizer "$COSMOS2B_TOKENIZER" \
        --lora-weights "$COSMOS2B_BASE_LORA" \
        --output-dir "$C2B_T16_FEAT/eval2"
fi

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

# ==============================================================================
# 4. GENERATE DEPLOYMENT CONFIGS & UPDATE LEADERBOARD
# ==============================================================================
run_task "Generate Deployment Config Wrappers & Update Leaderboard" \
    "$PYTHON" -c "
import json
from pathlib import Path

REPO = Path('/home/anton/lerobot-video-vam')
base_cfg = {
    'type': 'video_vam',
    'n_obs_steps': 1,
    'input_features': {'observation.images.front': {'type': 'VISUAL', 'shape': [3, 480, 640]}, 'observation.state': {'type': 'STATE', 'shape': [6]}},
    'output_features': {'action': {'type': 'ACTION', 'shape': [6]}},
    'device': 'cuda', 'use_amp': False, 'use_peft': False, 'camera_key': 'observation.images.front',
    'action_seed': 0, 'cosmos_compile_friendly': False, 'cosmos_torch_compile': False,
}

# 1. Cosmos 3 Edge 15k
c3_15k = dict(base_cfg)
c3_15k.update({'backend': 'cosmos3_edge', 'cosmos3_checkpoint': '/home/anton/.cache/video-vam/cosmos3-edge', 'cosmos3_lora_weights': str(REPO / 'outputs/train/v2-cosmos3-edge-video-lora-15k/best_lora.safetensors')})
p = REPO / 'outputs/train/v2-cosmos3-edge-lora-15k-smolexpert/config.json'
if p.parent.is_dir(): p.write_text(json.dumps(c3_15k, indent=2))

# 2. Cosmos 2B T=2 Undistilled
c2b_t2 = dict(base_cfg)
c2b_t2.update({'backend': 'cosmos', 'cosmos_state_t': 2, 'cosmos_context_transform': 'none'})
p = REPO / 'outputs/train/v2-cosmos2b-t2-undistilled-smolexpert/config.json'
if p.parent.is_dir(): p.write_text(json.dumps(c2b_t2, indent=2))

# 3. Cosmos 2B T=16 Reference
c2b_t16 = dict(base_cfg)
c2b_t16.update({'backend': 'cosmos', 'cosmos_state_t': 16, 'cosmos_context_transform': 'none'})
p = REPO / 'outputs/train/v2-cosmos2b-t16-condframes-smolexpert/config.json'
if p.parent.is_dir(): p.write_text(json.dumps(c2b_t16, indent=2))

print('Deployment configs written successfully!')
"

echo ""
echo "================================================================================"
echo "FAST V2 TRAINING QUEUE COMPLETED AT $(date)"
echo "================================================================================"
