
# gets latest checkpoint for given model type
model_type="smolvla"

if [ "$model_type" = "act" ]; then
  repo_id="orellius/so101_matchbox_fpv_act"
elif [ "$model_type" = "smolvla" ]; then
  repo_id="orellius/so101_matchbox_fpv_smolvla"
fi

# Look for the latest training directory for the given model type.
# The `-d` flag ensures that `ls` lists directory names rather than their contents.
latest_experiment_dir=$(ls -td outputs/train/matchbox_so101_matchbox_${model_type}_* | head -n 1)

# Find the latest step by numerically sorting the checkpoint directories.
latest_step=$(ls "$latest_experiment_dir/checkpoints" | sort -n | tail -n 1)

# Construct the path to the pretrained model of the latest checkpoint.
checkpoint_path="$latest_experiment_dir/checkpoints/$latest_step/pretrained_model"

# print
echo "Uploading $checkpoint_path to $repo_id"
echo "Latest experiment dir: $latest_experiment_dir"
echo "Step: $latest_step"

huggingface-cli upload $repo_id \
  $checkpoint_path

