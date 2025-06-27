#!/bin/sh

rm -r datasets/foo
python -m lerobot.record  \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM4 \
  --robot.cameras="{ up: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --robot.id=shabby \
  --dataset.repo_id=blanchon/eval_play_orange \
  --dataset.single_task='Play orange note' \
  --dataset.root='datasets/foo' \
  --display_data=false \
  --policy.path=outputs/train/play_orange/checkpoints/last/pretrained_model/ \
