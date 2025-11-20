#!/bin/sh


# rm -r outputs/train/matchbox/

set -eu

do_resume=false
num_steps=5000
resume_from_dir=""
POLICY_TYPE=smolvla
n_obs_steps=""
image_resolution="512,512"

wandb=true
suffix=""
batch_size=4

while [ "$#" -gt 0 ]; do
	case $1 in
		resume)
			do_resume=true
			;;
		--steps=*)
			num_steps="${1#*=}"
			;;
		--steps)
			shift 1
			num_steps=$1
			;;
		--suffix=*)
			suffix="${1#*=}"
			;;
		--suffix)
			shift 1
			suffix=$1
			;;
		--batch-size=*)
			batch_size="${1#*=}"
			;;
		--batch-size)
			shift 1
			batch_size=$1
			;;
		--policy-type=*)
			POLICY_TYPE="${1#*=}"
			;;
		--policy-type)
			shift 1
			POLICY_TYPE=$1
			;;
		--n-obs-steps=*)
			n_obs_steps="${1#*=}"
			;;
		--n-obs-steps)
			shift 1
			n_obs_steps=$1
			;;
		--image-resolution=*)
			image_resolution="${1#*=}"
			;;
		--image-resolution)
			shift 1
			image_resolution=$1
			;;
        --wandb)
            wandb=true
            ;;
		--resume-from=*)
			resume_from_dir="${1#*=}"
			do_resume=true
			;;
		--resume-from)
			shift 1
			resume_from_dir=$1
			do_resume=true
			;;
		*)
			echo "Usage: $0 [--steps=n] [--policy-type=TYPE] [--n-obs-steps=N] [--image-resolution=WxH] [resume] [--resume-from=DIR]"
			exit 1
			;;
	esac
	shift 1
done

num_warmup_steps=$(( num_steps / 20 ))
num_decay_steps="$num_steps"

echo "do resume: $do_resume"
echo "num_warmup_steps: $num_warmup_steps"
echo "policy type: $POLICY_TYPE"
echo "n_obs_steps: $n_obs_steps"

REPO_URL="https://huggingface.co/datasets/hubnemo/so101_sort"
REPO_ID="hubnemo/so101_sort"
REPO_NAME="so101_sort"

# Build model repo ID with context suffix if n_obs_steps > 1 and batch size
MODEL_REPO_ID="orellius/so101_sort_${POLICY_TYPE}"

# Add context suffix for temporal observations
if [ -n "$n_obs_steps" ] && [ "$n_obs_steps" -gt 1 ]; then
  MODEL_REPO_ID="${MODEL_REPO_ID}_context${n_obs_steps}"
fi

# Add batch size suffix
MODEL_REPO_ID="${MODEL_REPO_ID}_bs${batch_size}"

# Add user provided suffix
if [ -n "$suffix" ]; then
  MODEL_REPO_ID="${MODEL_REPO_ID}_${suffix}"
fi

echo "================================"
echo "MODEL_REPO_ID: $MODEL_REPO_ID"
echo "Create this repo on HuggingFace before training:"
echo "https://huggingface.co/new?name=$(echo $MODEL_REPO_ID | cut -d'/' -f2)"
echo "================================"

# Determine output directory
if $do_resume; then
  if [ -n "$resume_from_dir" ]; then
    OUTPUT_DIR="$resume_from_dir"
  else
    # Default to the most recent 5k step model
    OUTPUT_DIR="outputs/train/so101_sort_so101_sort_smolvla_20251107_2019"
  fi
  echo "Resuming from: $OUTPUT_DIR"
  JOB_NAME=$(basename ${OUTPUT_DIR})
  # Clear cache for updated dataset (new samples added)
  # The dataset is stored at datasets/${REPO_NAME} (from --dataset.root)
  # We need to clear the meta directory to force re-download of metadata
  echo "Clearing dataset cache for updated dataset..."
  DATASET_ROOT="datasets/${REPO_NAME}"
  if [ -d "$DATASET_ROOT/meta" ]; then
    echo "Removing cached metadata to force refresh from hub..."
    rm -rf "$DATASET_ROOT/meta"
  fi
  # Also clear the HuggingFace cache as backup
  rm -rf ~/.cache/huggingface/lerobot/hubnemo/${REPO_ID}
  
  # Check if WandB was enabled in the previous run
  # If not, disable it to avoid WandB resume errors
  CHECKPOINT_CONFIG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
  if [ -f "$CHECKPOINT_CONFIG" ]; then
    PREVIOUS_WANDB_ENABLED=$(python3 -c "import json; print(json.load(open('$CHECKPOINT_CONFIG'))['wandb']['enable'])" 2>/dev/null || echo "false")
    if [ "$PREVIOUS_WANDB_ENABLED" = "False" ] || [ "$PREVIOUS_WANDB_ENABLED" = "false" ]; then
      echo "Previous run had WandB disabled. Disabling WandB for resume to avoid errors."
      wandb=false
    fi
  fi
else
  MODEL_NAME="${REPO_NAME}_${POLICY_TYPE}_$(date +%Y%m%d_%H%M)"
  OUTPUT_DIR="outputs/train/${REPO_NAME}_${MODEL_NAME}"
  JOB_NAME="${REPO_NAME}_${MODEL_NAME}"
  echo "MODEL_NAME: $MODEL_NAME"
fi

# Set learning rate based on policy type (using recommended defaults)
case $POLICY_TYPE in
  act)
    OPTIMIZER_LR=1e-5
    ;;
  smolvla)
    OPTIMIZER_LR=1e-4
    ;;
  *)
    # Default to 3e-4 for other policies (can be overridden)
    OPTIMIZER_LR=3e-4
    ;;
esac

# Common arguments for both resume and fresh training
common_args=(
  --dataset.repo_id=${REPO_ID}
  --dataset.root=datasets/${REPO_NAME}
  --policy.type=${POLICY_TYPE}
  --policy.repo_id=${MODEL_REPO_ID}
  --output_dir=${OUTPUT_DIR}
  --job_name=${JOB_NAME}
  --policy.device=cuda
  --steps="$num_steps"
  --wandb.enable=$wandb
  --wandb.project=lerobot-shabby
  --dataset.image_transforms.enable=true
  --policy.optimizer_lr=${OPTIMIZER_LR}
  --batch_size=$batch_size
  --policy.push_to_hub=true
  --log_freq=100
  --eval_freq=200
)

# Add n_obs_steps - only for policies that support it and if specified
# Policies that support n_obs_steps > 1: diffusion (default 2), vqbet (default 5), act (newly added support)
# Policies that DON'T support n_obs_steps > 1: smolvla, pi0, pi05, pi0fast, tdmpc
if [ -n "$n_obs_steps" ]; then
  case $POLICY_TYPE in
    diffusion|vqbet|act)
      # These policies support n_obs_steps > 1
      common_args+=(--policy.n_obs_steps=$n_obs_steps)
      ;;
    smolvla|pi0|pi05|pi0fast|tdmpc)
      # These policies only support n_obs_steps = 1
      if [ "$n_obs_steps" != "1" ]; then
        echo "Warning: Policy type '$POLICY_TYPE' only supports n_obs_steps=1. Ignoring n_obs_steps=$n_obs_steps"
      fi
      # Don't add n_obs_steps parameter for these policies (they use default of 1)
      ;;
    *)
      # Unknown policy - try to add it anyway, will fail if not supported
      common_args+=(--policy.n_obs_steps=$n_obs_steps)
      ;;
  esac
fi

# Add image resolution based on policy type
if [ -n "$image_resolution" ]; then
  # Parse resolution (expects format like "512x512" or "512,512")
  if echo "$image_resolution" | grep -q "x"; then
    # Format: 512x512 -> convert to [512,512] (JSON array format for draccus)
    WIDTH=$(echo "$image_resolution" | cut -d'x' -f1)
    HEIGHT=$(echo "$image_resolution" | cut -d'x' -f2)
    RESOLUTION="[$WIDTH,$HEIGHT]"
  else
    # Format: 512,512 -> convert to [512,512] (JSON array format for draccus)
    WIDTH=$(echo "$image_resolution" | cut -d',' -f1)
    HEIGHT=$(echo "$image_resolution" | cut -d',' -f2)
    RESOLUTION="[$WIDTH,$HEIGHT]"
  fi
  
  case $POLICY_TYPE in
    smolvla)
      common_args+=(--policy.resize_imgs_with_padding=$RESOLUTION)
      ;;
    pi0|pi05|pi0fast)
      # For pi0 policies, keep comma-separated format (image_resolution might use different format)
      RESOLUTION_PI0=$(echo "$image_resolution" | tr 'x' ',')
      common_args+=(--policy.image_resolution=$RESOLUTION_PI0)
      ;;
    *)
      echo "Warning: Image resolution parameter not supported for policy type: $POLICY_TYPE"
      ;;
  esac
elif [ "$POLICY_TYPE" = "smolvla" ]; then
  # Default smolvla to 512x512 if not specified (JSON array format for draccus)
  common_args+=(--policy.resize_imgs_with_padding=[512,512])
fi

# Add scheduler parameters only for policies that support them
# smolvla, pi0, pi05, pi0fast support both scheduler_warmup_steps and scheduler_decay_steps
# diffusion and vqbet only support scheduler_warmup_steps
# ACT and others don't support these parameters
case $POLICY_TYPE in
  smolvla|pi0|pi05|pi0fast)
    common_args+=(
      --policy.scheduler_warmup_steps="$num_warmup_steps"
      --policy.scheduler_decay_steps="$num_decay_steps"
    )
    ;;
  diffusion|vqbet)
    common_args+=(
      --policy.scheduler_warmup_steps="$num_warmup_steps"
    )
    ;;
  *)
    # ACT and other policies don't support scheduler parameters
    ;;
esac
run_name="${REPO_NAME}${suffix}"

if $do_resume; then
  echo "Resuming for $num_steps steps"
  lerobot-train \
    "${common_args[@]}" \
    --resume=true \
    --config_path=${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json
else
  echo "Training from scratch with validation"
  #python -m pdb `which lerobot-train` "${common_args[@]}"
  lerobot-train "${common_args[@]}"
fi
