# can be set to true if we have a token that can access the repo
resume=${resume:-"false"}

TOP_CAMERA=$(v4l2-ctl --list-devices | grep 'Webcam C920' -A1 | tail -1 | tr -d "\t")
FPV_CAMERA=$(v4l2-ctl --list-devices | grep 'Innomaker' -A1 | tail -1 | tr -d "\t")

v4l2-ctl -d "${TOP_CAMERA}" -c focus_automatic_continuous=0
v4l2-ctl -d "${TOP_CAMERA}" -c focus_absolute=19

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=shabby \
    --robot.cameras="{ front: {type: opencv, index_or_path: $FPV_CAMERA, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: $TOP_CAMERA, width: 640, height: 480, fps: 30} }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=shabby_leader \
    --display_data=true \
    --play_sounds=false \
    --dataset.repo_id=hubnemo/so101_sort_cubes \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=5 \
    --dataset.num_episodes=20 \
    --dataset.single_task="Grab the purple cube and put it into the cardboard box, ignore the other objects" \
    --dataset.push_to_hub=true \
    --display_data=true \
    --resume=$resume
    
