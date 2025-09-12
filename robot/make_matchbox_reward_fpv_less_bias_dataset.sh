#!/bin/bash

script_dir="$(dirname "$(readlink -e "$0")")"

export DISPLAY=:0.0

(rm -r "$script_dir/../datasets/so101_matchbox_reward_fpv_less_bias_dataset" || exit 0)
python -m pdb lerobot/scripts/rl/gym_manipulator.py \
	--config_path "$script_dir/matchbox_env_config_reward_fpv_less_bias.json" 
