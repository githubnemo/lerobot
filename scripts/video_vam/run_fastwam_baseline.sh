#!/usr/bin/env bash
# Guarded one-hour native FastWAM baseline. This script intentionally does not launch the run.
set -euo pipefail

cd /home/anton/lerobot-video-vam

DATASET_ROOT=/home/anton/.cache/video-vam/cube-out-of-box-dataset
TEXT_ARTIFACT=/home/anton/.cache/video-vam/fastwam-text/cube-out-of-box.safetensors
TEXT_MARKER=${TEXT_ARTIFACT}.complete
OUTPUT_DIR=/home/anton/.cache/video-vam/runs/fastwam-cube-out-of-box

# The precompute process loads only UMT5/tokenizer, then exits before the policy process
# constructs Wan VAE/video/action components. The marker is written only after validation succeeds.
if [[ ! -f "$TEXT_MARKER" || ! -f "$TEXT_ARTIFACT" ]]; then
  .venv/bin/python scripts/video_vam/precompute_fastwam_text.py \
    --dataset-root "$DATASET_ROOT" \
    --repo-id hubnemo/cube_out_of_box_dataset \
    --episodes 0:32 \
    --output "$TEXT_ARTIFACT" \
    --dtype bfloat16 \
    --device cuda
  touch "$TEXT_MARKER"
fi

# Do not mkdir OUTPUT_DIR here: lerobot-train owns its output lifecycle. W&B stays online
# for the real baseline, and timeout provides the hard one-hour cap.
timeout --signal=TERM 3600s .venv/bin/lerobot-train \
  --dataset.repo_id=hubnemo/cube_out_of_box_dataset \
  --dataset.root="$DATASET_ROOT" \
  --dataset.episodes='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]' \
  --policy.type=fastwam \
  --policy.load_text_encoder=false \
  --policy.text_context_path="$TEXT_ARTIFACT" \
  --policy.torch_dtype=bfloat16 \
  --policy.freeze_video_expert=true \
  --policy.use_gradient_checkpointing=true \
  --policy.mot_checkpoint_mixed_attn=true \
  --policy.loss.lambda_video=0.0 \
  --policy.loss.lambda_action=1.0 \
  --policy.proprio_dim=6 \
  --policy.action_dim=6 \
  --policy.action_horizon=32 \
  --policy.n_action_steps=32 \
  --policy.num_video_frames=33 \
  --policy.action_video_freq_ratio=4 \
  --policy.image_size='[224,448]' \
  --batch_size=1 \
  --steps=310 \
  --save_freq=100 \
  --output_dir="$OUTPUT_DIR" \
  --job_name=fastwam_cube_out_of_box \
  --wandb.enable=true \
  --wandb.project=fastwam-cube-out-of-box \
  --wandb.mode=online

# Native-policy training runners always follow a successful run with the frozen RMSE helper.
scripts/video_vam/post_train_rmse.sh \
  --fastwam-checkpoint "$OUTPUT_DIR/checkpoints/last/pretrained_model" \
  "$OUTPUT_DIR/action-rmse.json"
