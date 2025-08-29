#!/bin/bash

script_dir="$(dirname "$(readlink -e "$0")")"

export DISPLAY=:0.0

rm -r /home/nemo/.cache/pysandbox-lerobot/huggingface/lerobot/your_username/shabby_task_dataset
python lerobot/scripts/rl/gym_manipulator.py --config_path "$script_dir/shabby_env_config_no_reward.json"

