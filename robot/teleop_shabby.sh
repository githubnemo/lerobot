#!/bin/sh


export DISPLAY=:0.0

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=shabby \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=shabby_leader \
    --display_data=true \
    --robot.cameras="{ side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, rotation: ROTATE_180}}" \
