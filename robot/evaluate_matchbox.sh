#!/bin/sh

set -eu

policy_path=outputs/train/matchbox/checkpoints/last/pretrained_model
use_peft=false

while [ $# -gt 0 ]; do
	case $1 in
		--policy-path)
			shift 1
			policy_path="$1"
			;;
		--use-peft)
			use_peft=true
			;;
		*)
			echo "Usage: $0 [--policy-path <id>]"
			exit 1
			;;
	esac
	shift 1
done


(rm -r datasets/foo || exit 0)
python -m pdb -m  lerobot.record  \
  --policy.use_peft=$use_peft \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM4 \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --robot.id=shabby \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM3 \
  --teleop.id=shabby_leader \
  --dataset.repo_id=hubnemo/eval_so101_matchbox_reward_fpv \
  --dataset.single_task='Put the matchbox on the bag' \
  --dataset.root='datasets/foo' \
  --dataset.episode_time_s=200 \
  --dataset.reset_time_s=10 \
  --dataset.num_episodes=2 \
  --display_data=true \
  --policy.path="$policy_path"
