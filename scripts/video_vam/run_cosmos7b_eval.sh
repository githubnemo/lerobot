cd /home/anton/lerobot-video-vam
source scripts/video_vam/cosmos_cuda_env.sh
source scripts/video_vam/gpu_lock.sh
acquire_gpu_lock eval_cosmos7b

mkdir -p /home/anton/lerobot-video-vam/outputs/evaluation/cosmos7b-protocol1-reeval

for seed in 0 1 2; do
  echo "Evaluating Cosmos 7B Protocol 1.0 seed $seed..."
  /home/anton/lerobot-video-vam/.venv/bin/python scripts/video_vam/evaluate_cosmos_world2action_cache.py \
    --checkpoint /home/anton/lerobot-video-vam/outputs/train/cosmos7b-protocol1-smolexpert/best.safetensors \
    --manifest /home/anton/.cache/video-vam/cosmos7b-protocol1-cache/val/manifest.json \
    --normalizer /home/anton/lerobot-video-vam/outputs/train/cosmos7b-protocol1-smolexpert/normalizer.safetensors \
    --expected-step 17000 \
    --seed $seed \
    --overwrite \
    --output-json /home/anton/lerobot-video-vam/outputs/evaluation/cosmos7b-protocol1-reeval/eval_seed${seed}.json
done
echo "Cosmos 7B evaluation completed!"
