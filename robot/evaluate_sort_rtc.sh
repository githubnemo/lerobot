#!/bin/sh

set -eu

policy_path=outputs/train/sort/checkpoints/last/pretrained_model
use_peft=false
execution_horizon=20

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
			execution_horizon="$1"
			;;
		*)
			echo "Usage: $0 [--policy-path <id>]"
			exit 1
			;;
	esac
	shift 1
done


python examples/rtc/eval_with_real_robot.py \
        --policy.path="$policy_path" \
	--policy.device=cuda \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
	--robot.id=shabby \
	  --robot.type=so101_follower \
	  --robot.port=/dev/ttyACM1 \
	  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
	--task="Grab the matchbox and put it into the cardboard box, ignore the other objects" \
        --duration=120 \
	--rtc.execution_horizon=${execution_horizon}

