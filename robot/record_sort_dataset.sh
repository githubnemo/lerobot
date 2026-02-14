# can be set to true if we have a token that can access the repo
resume=${resume:-"false"}

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=shabby \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=shabby_leader \
    --display_data=true \
    --play_sounds=false \
    --dataset.repo_id=hubnemo/so101_sort \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=5 \
    --dataset.num_episodes=20 \
    --dataset.single_task="Grab the matchbox and put it into the cardboard box, ignore the other objects" \
    --dataset.push_to_hub=true \
    --display_data=true \
    --resume=$resume
    
