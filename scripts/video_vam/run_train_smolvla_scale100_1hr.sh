#!/usr/bin/env bash
# 1-Hour SmolVLA Training on Scale-100 Demonstration Dataset (Orellius/cube_out_of_box_v2)
set -Eeuo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAINER="$REPO/.venv/bin/lerobot-train"
# Native training requires a non-existent output child inside the fresh run.
RUN_DIR="${RUN_DIR:-$REPO/outputs/train/smolvla-scale100-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/training}"
if [[ -e "$RUN_DIR" || -L "$RUN_DIR" || -e "$OUTPUT_DIR" || -L "$OUTPUT_DIR" ]]; then
    printf 'Refusing to overwrite existing RUN_DIR or OUTPUT_DIR\n' >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$RUN_DIR")"
mkdir -- "$RUN_DIR"
RUN_DIR="$(cd -- "$RUN_DIR" && pwd)"
OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"
LOG_FILE="$RUN_DIR/train.log"

exec > >(tee -a "$LOG_FILE") 2>&1

cd "$REPO"
source scripts/video_vam/cosmos_cuda_env.sh
source scripts/video_vam/gpu_lock.sh

acquire_gpu_lock smolvla-scale100-train

# 82 training episodes: 0..31 (canonical) + 40..89 (extended), holding out 32..39 and 90..99
TRAIN_EPISODES=$(python3 -c "print(str(list(range(32)) + list(range(40, 90))))")

printf "[%s] STARTING SMOLVLA 1-HOUR TRAINING ON SCALE-100 DATASET\n" "$(date --iso-8601=seconds)"
printf "[%s] Output directory: %s\n" "$(date --iso-8601=seconds)" "$OUTPUT_DIR"

"$TRAINER" \
    --dataset.repo_id=Orellius/cube_out_of_box_v2 \
    --dataset.episodes="$TRAIN_EPISODES" \
    --dataset.use_imagenet_stats=true \
    --dataset.image_transforms.enable=false \
    --dataset.video_backend=torchcodec \
    --dataset.streaming=false \
    --dataset.eval_split=0.0 \
    --policy.type=smolvla \
    --policy.device=cuda \
    --policy.pretrained_path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --policy.load_vlm_weights=false \
    --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.use_amp=true \
    --policy.chunk_size=50 \
    --policy.n_action_steps=10 \
    --policy.n_obs_steps=1 \
    --policy.num_steps=10 \
    --policy.resize_imgs_with_padding=[512,512] \
    --batch_size=8 \
    --optimizer.type=adamw \
    --optimizer.lr=0.0001 \
    --optimizer.betas=[0.9,0.95] \
    --optimizer.eps=1e-8 \
    --optimizer.weight_decay=1e-10 \
    --optimizer.grad_clip_norm=10 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.peak_lr=0.0001 \
    --scheduler.decay_lr=2.5e-6 \
    --scheduler.num_decay_steps=25000 \
    --scheduler.num_warmup_steps=1000 \
    --steps=25000 \
    --seed=42 \
    --num_workers=4 \
    --log_freq=100 \
    --eval_steps=0 \
    --save_checkpoint=true \
    --save_freq=0 \
    --output_dir="$OUTPUT_DIR" \
    --wandb.enable=false

date --iso-8601=seconds > "$RUN_DIR/COMPLETE"
printf "[%s] SMOLVLA SCALE-100 TRAINING COMPLETED SUCCESSFULLY!\n" "$(date --iso-8601=seconds)"
