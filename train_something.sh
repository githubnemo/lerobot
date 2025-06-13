#!/bin/sh

python lerobot/scripts/train.py \
  --dataset.repo_id=blanchon/play_orange \
  --policy.type=smolvla \
  --output_dir=outputs/train/play_orange \
  --job_name=play_orange \
  --policy.device=cuda \
  --steps=500 \
  --wandb.enable=false

