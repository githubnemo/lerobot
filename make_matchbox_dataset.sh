python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM4 \
    --robot.id=shabby \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM3 \
    --teleop.id=shabby_leader \
    --display_data=true \
    --dataset.repo_id=hubnemo/so101_matchbox \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=10 \
    --dataset.num_episodes=2 \
    --dataset.single_task="Put the matchbox on the bag" \
    --dataset.push_to_hub=true \
    --resume=true
    