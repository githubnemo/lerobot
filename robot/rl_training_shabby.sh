#!/bin/bash

script_dir="$(dirname "$(readlink -e "$0")")"

if [ -z "$script_dir" ]; then
	echo "cannot find script dir"
	exit 1
fi

export DISPLAY=:0.0

python lerobot/scripts/rl/gym_manipulator.py --config_path "$script_dir/shabby_env_config.json" $@

