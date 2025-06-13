#!/bin/sh

python -m lerobot.calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM3 \
    --teleop.id=shabby_leader
