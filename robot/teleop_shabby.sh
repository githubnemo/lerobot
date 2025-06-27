#!/bin/sh

python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM4 \
    --robot.id=shabby \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM3 \
    --teleop.id=shabby_leader \
    --display_data=true \
    --robot.cameras="{ side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
