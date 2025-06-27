#!/bin/sh

python -m lerobot.scripts.find_joint_limits \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM4 \
    --robot.id=shabby \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM3 \
    --teleop.id=shabby_leader
