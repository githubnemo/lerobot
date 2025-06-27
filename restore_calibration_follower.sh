#!/bin/sh

# not needed! the problem is, that calibration data is saved to bus. when switching between _follower and _follower_endeffector, they calibrations are slightly different so the bus responsds that it is uncalibrated.

cp ~/shabby.json /home/nemo/.cache/pysandbox-lerobot/huggingface/lerobot/calibration/robots/so101_follower/shabby.json
