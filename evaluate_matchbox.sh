#!/bin/sh

rm -r datasets/foo
python -m lerobot.record  \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM4 \
  --robot.cameras="{ up: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --robot.id=shabby \
  --dataset.repo_id=nemo/eval_matchbox \
  --dataset.single_task='Put the matchbox on the bag' \
  --dataset.root='datasets/foo' \
  --display_data=false \
  --policy.path=outputs/train/matchbox/checkpoints/last/pretrained_model/ \
