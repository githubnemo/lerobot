#!/bin/sh

set -eu

policy_path=outputs/train/sort/checkpoints/last/pretrained_model
#use_peft=false

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

TOP_CAMERA=$(v4l2-ctl --list-devices | grep 'Webcam C920' -A1 | tail -1 | tr -d "\t")
FPV_CAMERA=$(v4l2-ctl --list-devices | grep 'Innomaker' -A1 | tail -1 | tr -d "\t")

v4l2-ctl -d "${TOP_CAMERA}" -c focus_automatic_continuous=0
v4l2-ctl -d "${TOP_CAMERA}" -c focus_absolute=19

(rm -r datasets/foo || exit 0)
lerobot-record  \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.cameras="{ front: {type: opencv, index_or_path: $FPV_CAMERA, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: $TOP_CAMERA, width: 640, height: 480, fps: 30} }" \
  --robot.id=shabby \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=shabby_leader \
  --dataset.repo_id=hubnemo/eval_so101_sort_cubes \
  --dataset.single_task="Grab the purple cube and put it into the cardboard box, ignore the other objects" \
  --dataset.root='datasets/foo' \
  --dataset.episode_time_s=200 \
  --dataset.reset_time_s=2 \
  --dataset.num_episodes=10 \
  --dataset.fps=30 \
  --display_data=true \
  --policy.path="$policy_path" \
  --policy.n_action_steps=10

