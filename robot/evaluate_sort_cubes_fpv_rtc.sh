#!/bin/sh

set -eu

policy_path=outputs/train/sort/checkpoints/last/pretrained_model
use_peft=false
execution_horizon=20
duration=120

while [ $# -gt 0 ]; do
	case $1 in
		--policy-path)
			shift 1
			policy_path="$1"
			;;
		--use-peft)
			use_peft=true
			;;
		--execution-horizon)
			shift 1
			execution_horizon="$1"
			;;
		--duration)
			shift 1
			duration="$1"
			;;
		*)
			echo "Usage: $0 [--policy-path <id>]"
			exit 1
			;;
	esac
	shift 1
done

FPV_CAMERA=$(v4l2-ctl --list-devices | grep 'Innomaker' -A1 | tail -1 | tr -d "\t")

test -n "$FPV_CAMERA"

python examples/rtc/eval_with_real_robot.py \
        --policy.path="$policy_path" \
	--policy.device=cuda \
        --rtc.enabled=true \
	--robot.id=shabby \
	  --robot.type=so101_follower \
	  --robot.port=/dev/ttyACM1 \
	--robot.cameras="{ front: {type: opencv, index_or_path: $FPV_CAMERA, width: 640, height: 480, fps: 30} }" \
	--task="Grab the purple cube and put it into the cardboard box, ignore the other objects" \
        --duration=$duration \
	--rtc.execution_horizon=${execution_horizon} \
	--policy.n_action_steps=30
