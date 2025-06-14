#!/bin/sh


# rm -r outputs/train/matchbox/

set -eu

do_resume=false
num_steps=200

while [ "$#" -gt 0 ]; do
	case $1 in
		resume)
			do_resume=true
			;;
		--steps)
			shift 1
			num_steps=$1
			;;
		*)
			echo "Usage: $0 [--steps=n] [resume]"
			exit 1
			;;
	esac
	shift 1
done

num_warmup_steps="$(( $num_steps / 20 ))"
num_decay_steps="$num_steps"

if $do_resume; then
	echo "Resuming for $num_steps steps"
	python lerobot/scripts/train.py \
	  --dataset.repo_id=hubnemo/so101_matchbox \
	  --policy.type=smolvla \
	  --output_dir=outputs/train/matchbox \
	  --job_name=matchbox \
	  --policy.device=cuda \
	  --policy.scheduler_warmup_steps=$num_warmup_steps \
	  --policy.scheduler_decay_steps=$num_decay_steps \
	  --steps=$num_steps \
	  --wandb.enable=false \
	  --resume=true \
	  --config_path=outputs/train/matchbox/checkpoints/last/pretrained_model/train_config.json
else
	echo Training from scratch
	python lerobot/scripts/train.py \
	  --dataset.repo_id=hubnemo/so101_matchbox \
	  --policy.type=smolvla \
	  --output_dir=outputs/train/matchbox \
	  --job_name=matchbox \
	  --policy.device=cuda \
	  --policy.warmup_steps=$num_warmup_steps \
	  --policy.decay_steps=$num_decay_steps \
	  --steps=$num_steps \
	  --wandb.enable=false \
fi
