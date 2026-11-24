#!/usr/bin/env bash
# exp004: FLUX 3 Action SO-101 LoRA (W1 wrist-duplicated, recalibrated quantiles) + Protocol 1.0 eval + HF upload.
# Target host: RunPod RTX 5090 (32 GB) | Dataset: Orellius/so101_sort_cubes_no_top@a7c156de
set -Eeuo pipefail

export RUN_TAG="exp004_flux3_sort_cubes_lora_w1"
export VAM_CACHE_DIR="${VAM_CACHE_DIR:-$HOME/.cache/video-vam}"
export HF_UPLOAD_OWNER="${HF_UPLOAD_OWNER:-Orellius}"
PYTHON="${VAM_VENV:-.venv}/bin/python"

RECALIBRATED_DIR="outputs/bases/flux3-so101-sortcubes-recal"
if [[ ! -d "${RECALIBRATED_DIR}" ]]; then
  mkdir -p outputs/bases
  "${PYTHON}" scripts/video_vam/export_flux3_recalibrated_base.py \
    --policy-path "${VAM_CACHE_DIR}/flux3-action-so101" \
    --dataset-repo-id Orellius/so101_sort_cubes_no_top \
    --dataset-revision a7c156de0e84a81e19457e6d783e4316e9f901e1 \
    --train-episodes 0:68 \
    --output "${RECALIBRATED_DIR}"
fi

# Delay auto-upload until after offline Protocol 1.0 RMSE eval is written into the run dir
HF_AUTO_UPLOAD=0 bash scripts/video_vam/run_experiment.sh f3a_sort_cubes_lora_w1 \
  --policy.path="${RECALIBRATED_DIR}" \
  "$@"

"${PYTHON}" scripts/video_vam/evaluate_flux3_action_rmse.py \
  --policy-path "outputs/train/${RUN_TAG}/checkpoints/last/pretrained_model" \
  --dataset-repo-id Orellius/so101_sort_cubes_no_top \
  --dataset-revision a7c156de0e84a81e19457e6d783e4316e9f901e1 \
  --train-episodes 0:68 \
  --eval-episodes 68:77 \
  --stride 20 \
  --scene-mode duplicate \
  --recalibrate none \
  --output "outputs/train/${RUN_TAG}/flux3_eval.json"

if [[ "${HF_AUTO_UPLOAD:-1}" == "1" ]]; then
  "${PYTHON}" scripts/video_vam/push_checkpoint_to_hf.py --run-dir "outputs/train/${RUN_TAG}"
fi
