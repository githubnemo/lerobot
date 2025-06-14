#!/bin/sh


# rm -r outputs/train/matchbox/

set -eu

do_resume=false
num_steps=200
use_peft=false
suffix=""

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

num_warmup_steps="$(( $num_steps / 20 ))"
num_decay_steps="$num_steps"
run_name="matchbox${suffix}"

if $do_resume; then
	echo "Resuming for $num_steps steps"
	python -m pdb lerobot/scripts/train.py \
	  --dataset.repo_id=hubnemo/so101_matchbox \
	  --policy.type=smolvla \
	  --output_dir=outputs/train/${run_name} \
	  --job_name=${run_name} \
	  --policy.device=cuda \
	  --policy.scheduler_warmup_steps=$num_warmup_steps \
	  --policy.scheduler_decay_steps=$num_decay_steps \
	  --steps=$num_steps \
	  --wandb.enable=false \
	  --resume=true \
	  --config_path=outputs/train/${run_name}/checkpoints/last/pretrained_model/train_config.json \
	  --use_peft=$use_peft
else
	echo Training from scratch
	python -m pdb lerobot/scripts/train.py \
	  --dataset.repo_id=hubnemo/so101_matchbox \
	  --policy.type=smolvla \
	  --output_dir=outputs/train/${run_name} \
	  --job_name=${run_name} \
	  --policy.device=cuda \
	  --policy.scheduler_warmup_steps=$num_warmup_steps \
	  --policy.scheduler_decay_steps=$num_decay_steps \
	  --steps=$num_steps \
	  --wandb.enable=false \
	  --use_peft=$use_peft
fi
