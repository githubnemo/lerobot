#!/bin/sh


# rm -r outputs/train/matchbox/


python lerobot/scripts/train.py \
  --dataset.repo_id=hubnemo/so101_matchbox \
  --policy.type=smolvla \
  --output_dir=outputs/train/matchbox \
  --job_name=matchbox \
  --policy.device=cuda \
  --steps=200 \
  --wandb.enable=false \

