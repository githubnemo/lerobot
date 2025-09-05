#!/bin/sh


# rm -r outputs/train/matchbox/

set -eu

do_resume=false
num_steps=1000
use_peft=false
wandb=false
suffix=""
batch_size=8
val_batch_size=32

while [ "$#" -gt 0 ]; do
	case $1 in
		resume)
			do_resume=true
			;;
		--steps)
			shift 1
			num_steps=$1
			;;
		--suffix)
			shift 1
			suffix=$1
			;;
		--batch-size)
			shift 1
			batch_size=$1
			;;
        --wandb)
            wandb=true
            ;;
		--use-peft)
			use_peft=true
			;;
		*)
			echo "Usage: $0 [--steps=n] [resume]"
			exit 1
			;;
	esac
	shift 1
done

num_warmup_steps=$(( num_steps / 20 ))
num_decay_steps="$num_steps"

echo "do resume: $do_resume"
echo "num_warmup_steps: $num_warmup_steps"

# Clear cache for updated dataset (uncomment if dataset was updated)
# rm -r ~/.cache/huggingface/lerobot/hubnemo/so101_matchbox

POLICY_TYPE=act
MODEL_NAME="so101_matchbox_fpv_${POLICY_TYPE}_$(date +%Y%m%d_%H%M)"
echo "MODEL_NAME: $MODEL_NAME"

# Common arguments for both resume and fresh training
common_args=(
  --dataset.repo_id=hubnemo/so101_matchbox_reward_fpv
  --dataset.root=datasets/so101_matchbox_reward_fpv
  --policy.type=${POLICY_TYPE}
  --output_dir=outputs/train/matchbox_${MODEL_NAME}
  --job_name=matchbox_${MODEL_NAME}
  --policy.device=cuda
#  --policy.scheduler_warmup_steps="$num_warmup_steps"
#  --policy.scheduler_decay_steps="$num_decay_steps"
  --steps="$num_steps"
  --wandb.enable=$wandb
  --dataset.image_transforms.enable=true
  --policy.optimizer_lr=1e-3
  --batch_size=$batch_size
  --policy.push_to_hub=False
  --log_freq=100
  # Validation settings for supervised learning
  --use_validation=true
  --val_split=0.02
  --val_freq=100
  --val_batch_size=$val_batch_size
  --use_peft=$use_peft
)
run_name="matchbox${suffix}"

if $do_resume; then
  echo "Resuming for $num_steps steps"
  python lerobot/scripts/train.py \
    "${common_args[@]}" \
    --resume=true \
    --config_path=outputs/train/matchbox_fpv/checkpoints/last/pretrained_model/train_config.json
else
  echo "Training from scratch with validation"
  python lerobot/scripts/train.py "${common_args[@]}"
fi
