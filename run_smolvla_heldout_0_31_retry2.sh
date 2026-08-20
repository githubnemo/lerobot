#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/anton/lerobot-video-vam
PYTHON="$PROJECT/.venv/bin/python"
TRAIN="$PROJECT/.venv/bin/lerobot-train"
OUT="$PROJECT/outputs/train/cube_out_of_box_il_smolvla_heldout_0_31_20260818_retry2"
LOG="$PROJECT/outputs/train/cube_out_of_box_il_smolvla_heldout_0_31_20260818_retry2.launch.log"
SPLIT_SOURCE=/home/anton/.cache/video-vam/splits/rehearsal-stride20.json
DATASET_REVISION=243370c3c08bcbd860133c4a0d658ea7c1d2e77e
DATASET_SNAPSHOT="/home/anton/.cache/huggingface/lerobot/hub/datasets--hubnemo--cube_out_of_box_dataset/snapshots/$DATASET_REVISION"
RUN_ID_FILE="$PROJECT/outputs/train/cube_out_of_box_il_smolvla_heldout_0_31_20260818_retry2.wandb_run_id"

mkdir -p "$PROJECT/outputs/train"
SPLIT_COPY="$PROJECT/outputs/train/cube_out_of_box_il_smolvla_heldout_0_31_20260818_retry2.split.json"
MANIFEST_STAGING="$PROJECT/outputs/train/cube_out_of_box_il_smolvla_heldout_0_31_20260818_retry2.manifest.json"
exec > >(tee -a "$LOG") 2>&1

echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "HOST $(hostname)"
echo "GPU_WAIT threshold=2000MiB util<5%"

if [[ ! -f "$SPLIT_SOURCE" ]]; then
  echo "ERROR missing split source: $SPLIT_SOURCE"
  exit 20
fi
cp -f "$SPLIT_SOURCE" "$SPLIT_COPY"

RUN_ID="$($PYTHON - <<'PY'
import secrets
print(secrets.token_hex(4))
PY
)"
printf '%s\n' "$RUN_ID" > "$RUN_ID_FILE"

$PYTHON - "$MANIFEST_STAGING" "$RUN_ID" "$SPLIT_SOURCE" "$DATASET_REVISION" <<'PY'
import json, sys
from pathlib import Path
out, run_id, split, revision = sys.argv[1:]
train_episodes = list(range(32))
val_episodes = list(range(32, 40))
manifest = {
    "run_id": run_id,
    "entrypoint": "/home/anton/lerobot-video-vam/.venv/bin/lerobot-train",
    "dataset": {"repo_id": "hubnemo/cube_out_of_box_dataset", "revision": revision},
    "split_source": split,
    "train_episodes": train_episodes,
    "held_out_episodes": val_episodes,
    "normalization_mapping": {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"},
    "policy": {"type": "smolvla", "pretrained_path": "lerobot/smolvla_base", "chunk_size": 50, "n_action_steps": 10, "n_obs_steps": 1, "resize_imgs_with_padding": [512, 512], "use_amp": True},
    "training": {"batch_size": 8, "steps": 5000, "seed": 1000, "image_transforms": False},
    "normalization_stats_source": f"{revision}/meta/stats.json",
}
Path(out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

# Wait for the other job to release the GPU. Require two clean observations.
clean=0
while (( clean < 2 )); do
  read -r mem util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  echo "GPU_STATUS memory=${mem}MiB utilization=${util}%"
  if (( mem < 2000 && util < 5 )); then
    clean=$((clean + 1))
  else
    clean=0
  fi
  if (( clean < 2 )); then sleep 10; fi
done
echo "GPU_FREE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

CMD=(
  "$TRAIN"
  --dataset.repo_id=hubnemo/cube_out_of_box_dataset
  --dataset.revision="$DATASET_REVISION"
  --dataset.episodes='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]'
  --dataset.use_imagenet_stats=true
  --dataset.image_transforms.enable=false
  --dataset.video_backend=torchcodec
  --dataset.streaming=false
  --dataset.eval_split=0.0
  --policy.type=smolvla
  --policy.device=cuda
  --policy.pretrained_path=lerobot/smolvla_base
  --policy.push_to_hub=false
  --policy.load_vlm_weights=true
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct
  --policy.freeze_vision_encoder=true
  --policy.train_expert_only=true
  --policy.use_amp=true
  --policy.chunk_size=50
  --policy.n_action_steps=10
  --policy.n_obs_steps=1
  --policy.num_steps=10
  --policy.resize_imgs_with_padding='[512,512]'
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
  --policy.optimizer_lr=0.0001
  --policy.optimizer_betas='[0.9,0.95]'
  --policy.optimizer_eps=1e-8
  --policy.optimizer_weight_decay=1e-10
  --policy.optimizer_grad_clip_norm=10
  --policy.scheduler_decay_lr=2.5e-6
  --policy.scheduler_decay_steps=30000
  --policy.scheduler_warmup_steps=1000
  --batch_size=8
  --optimizer.type=adamw
  --optimizer.lr=0.0001
  --optimizer.betas='[0.9,0.95]'
  --optimizer.eps=1e-8
  --optimizer.weight_decay=1e-10
  --optimizer.grad_clip_norm=10
  --scheduler.type=cosine_decay_with_warmup
  --scheduler.peak_lr=0.0001
  --scheduler.decay_lr=2.5e-6
  --scheduler.num_decay_steps=30000
  --scheduler.num_warmup_steps=1000
  --steps=5000
  --seed=1000
  --num_workers=4
  --log_freq=100
  --eval_steps=0
  --save_checkpoint=true
  --save_freq=5000
  --save_checkpoint_to_hub=false
  --use_policy_training_preset=true
  --output_dir="$OUT"
  --job_name=cube_out_of_box_il_smolvla_heldout_0_31
  --wandb.enable=true
  --wandb.entity=hubnemo-hugging-face
  --wandb.project=cube_out_of_box_il
  --wandb.run_id="$RUN_ID"
  --wandb.disable_artifact=false
)

echo "COMMAND_START"
printf ' %q' "${CMD[@]}"
printf '\nCOMMAND_END\n'
cd "$PROJECT"
set +e
"${CMD[@]}"
TRAIN_STATUS=$?
set -e
echo "TRAIN_STATUS $TRAIN_STATUS"

if (( TRAIN_STATUS != 0 )); then
  exit "$TRAIN_STATUS"
fi

cp -f "$SPLIT_COPY" "$OUT/split_source.json"
cp -f "$MANIFEST_STAGING" "$OUT/run_manifest.json"

# Attach the reload-critical processor files, dataset stats, revision, and split
# to the same W&B run. The built-in logger uploads model.safetensors only.
$PYTHON - "$OUT" "$RUN_ID" "$DATASET_SNAPSHOT" "$SPLIT_SOURCE" <<'PY'
import json, sys
from pathlib import Path
import wandb
out, run_id, snapshot, split_source = map(Path, sys.argv[1:])
checkpoint_candidates = sorted(out.glob("checkpoints/*/pretrained_model"), key=lambda p: p.stat().st_mtime)
if not checkpoint_candidates:
    raise SystemExit(f"No pretrained_model checkpoint found under {out}/checkpoints")
checkpoint = checkpoint_candidates[-1]
print("CHECKPOINT", checkpoint)
required = [checkpoint / "policy_preprocessor.json", checkpoint / "policy_postprocessor.json", checkpoint / "config.json"]
required += [out / "run_manifest.json", out / "split_source.json"]
required += [snapshot / "meta/stats.json", snapshot / "meta/info.json"]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit("Missing reload artifact files: " + ", ".join(missing))
metadata = {
    "dataset_repo_id": "hubnemo/cube_out_of_box_dataset",
    "dataset_revision": snapshot.name,
    "train_episodes": list(range(32)),
    "held_out_episodes": list(range(32, 40)),
    "normalization_mapping": {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"},
    "checkpoint_path": str(checkpoint),
    "source_run_id": run_id,
}
run = wandb.init(id=run_id, entity="hubnemo-hugging-face", project="cube_out_of_box_il", resume="must")
artifact_name = f"smolvla-heldout-normalization-{run_id}"
artifact = wandb.Artifact(artifact_name, type="policy-metadata", metadata=metadata)
files = {
    "policy_preprocessor.json": checkpoint / "policy_preprocessor.json",
    "policy_postprocessor.json": checkpoint / "policy_postprocessor.json",
    "policy_config.json": checkpoint / "config.json",
    "run_manifest.json": out / "run_manifest.json",
    "split_source.json": out / "split_source.json",
    "dataset_meta/stats.json": snapshot / "meta/stats.json",
    "dataset_meta/info.json": snapshot / "meta/info.json",
}
for name, path in files.items():
    artifact.add_file(str(path), name=name)
run.log_artifact(artifact)
run.finish()
print("NORMALIZATION_ARTIFACT", artifact_name)
print("NORMALIZATION_FILES", sorted(files))
PY

echo "FINISH $(date -u +%Y-%m-%dT%H:%M:%SZ)"
