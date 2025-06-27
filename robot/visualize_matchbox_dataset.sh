python -m lerobot.replay \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM4 \
    --robot.id=shabby \
    --dataset.repo_id=nemo/matchbox \
    --dataset.episode=1 # choose the episode you want to replay