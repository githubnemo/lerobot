#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/anton/lerobot-video-vam
PYTHON="$REPO/.venv/bin/python"
TRAINER="$REPO/.venv/bin/lerobot-train"
CACHE=/home/anton/.cache/video-vam
RUN_ROOT="$CACHE/runs/smolvla-train-only-stats-20260826"
SOURCE_DATASET="$CACHE/cube-out-of-box-dataset"
STATS_DATASET="$CACHE/cube-out-of-box-dataset-train0-31-stats"
SPLIT="$CACHE/splits/rehearsal-stride20.json"
MANIFEST="$CACHE/cosmos-rehearsal-stride20-sigma80/manifest.json"
REFERENCE_OUTPUT="$REPO/outputs/train/cube_out_of_box_il_smolvla_heldout_0_31_20260819_1hr"
OUTPUT_DIR="$REPO/outputs/train/cube_out_of_box_il_smolvla_train_only_stats_0_31_20260826_1hr"
CHECKPOINT="$OUTPUT_DIR/checkpoints/last/pretrained_model"
REVISION=243370c3c08bcbd860133c4a0d658ea7c1d2e77e
EPISODES='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]'

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/queue.log") 2>&1
stage=initializing
trap 'status=$?; set +e; printf "[%s] FAILED stage=%s status=%s\n" "$(date --iso-8601=seconds)" "$stage" "$status"; printf "%s\n" "$stage" > "$RUN_ROOT/FAILED"; df -h / > "$RUN_ROOT/disk-at-failure.txt"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-at-failure.txt" || true; exit "$status"' ERR

if [[ -f "$RUN_ROOT/COMPLETE" ]]; then
    printf "[%s] SmolVLA corrected rerun already complete\n" "$(date --iso-8601=seconds)"
    exit 0
fi
if [[ -f "$RUN_ROOT/FAILED" ]]; then
    printf "[%s] removing this queue's stale retry marker %s\n" \
        "$(date --iso-8601=seconds)" "$RUN_ROOT/FAILED"
    rm -f -- "$RUN_ROOT/FAILED"
fi

cd "$REPO"
source scripts/video_vam/cosmos_cuda_env.sh
source scripts/video_vam/gpu_lock.sh
export PYTHONHASHSEED=0

disk_audit() {
    "$PYTHON" - "$REFERENCE_OUTPUT" "$OUTPUT_DIR" <<'PY'
import shutil
import sys
from pathlib import Path

reference = Path(sys.argv[1])
output = Path(sys.argv[2])
if not reference.is_dir():
    raise FileNotFoundError(f"reference output is missing: {reference}")
if output.exists():
    raise FileExistsError(f"refusing to overwrite an existing corrected run: {output}")
reference_bytes = sum(item.stat().st_size for item in reference.rglob("*") if item.is_file())
free_bytes = shutil.disk_usage("/").free
retained_floor_bytes = 20 * 2**30
required_free_bytes = reference_bytes + retained_floor_bytes
print(
    "DISK_AUDIT "
    f"free_gib={free_bytes / 2**30:.2f} "
    f"reference_run_gib={reference_bytes / 2**30:.2f} "
    f"retained_floor_gib={retained_floor_bytes / 2**30:.2f} "
    f"required_free_gib={required_free_bytes / 2**30:.2f}"
)
if free_bytes < required_free_bytes:
    raise RuntimeError(
        "refusing SmolVLA training: free disk is below one reference-run footprint "
        "+ 20 GiB retained floor; no files were deleted"
    )
PY
    df -h /
}

stage=disk-audit
disk_audit

stage=prepare-train-only-stats
"$PYTHON" scripts/video_vam/prepare_smolvla_train_only_stats.py \
    --source-root "$SOURCE_DATASET" \
    --output-root "$STATS_DATASET" \
    --repo-id hubnemo/cube_out_of_box_dataset \
    --revision "$REVISION" \
    --episodes {0..31} \
    > "$RUN_ROOT/stats-delta.json"
touch "$RUN_ROOT/STATS_COMPLETE"

stage=acquire-gpu-lock
acquire_gpu_lock smolvla-train-only-stats-20260826

stage=training
TRAIN_COMMAND=(
    "$TRAINER"
    --dataset.repo_id=hubnemo/cube_out_of_box_dataset
    --dataset.revision="$REVISION"
    --dataset.root="$STATS_DATASET"
    --dataset.episodes="$EPISODES"
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
    --steps=29200
    --seed=1000
    --num_workers=4
    --log_freq=100
    --eval_steps=0
    --save_checkpoint=true
    --save_freq=7000
    --save_checkpoint_to_hub=false
    --use_policy_training_preset=true
    --output_dir="$OUTPUT_DIR"
    --job_name=cube_out_of_box_il_smolvla_train_only_stats_0_31_1hr
    --wandb.enable=true
    --wandb.entity=hubnemo-hugging-face
    --wandb.project=cube_out_of_box_il
    --wandb.run_id=smolvla-trainonly-20260826
    --wandb.disable_artifact=true
)
printf "[%s] TRAIN_COMMAND:" "$(date --iso-8601=seconds)"
printf " %q" "${TRAIN_COMMAND[@]}"
printf "\n"
"${TRAIN_COMMAND[@]}"
touch "$RUN_ROOT/TRAINING_COMPLETE"

stage=verify-checkpoint-stats
"$PYTHON" - "$CHECKPOINT" "$STATS_DATASET/meta/stats.json" "$RUN_ROOT/checkpoint-stats-verification.json" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

checkpoint = Path(sys.argv[1])
stats = json.loads(Path(sys.argv[2]).read_text())
output = Path(sys.argv[3])
preprocessor = checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors"
postprocessor = checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
report = {}
for path in (preprocessor, postprocessor):
    tensors = load_file(path)
    for key in ("action", "observation.state"):
        count = int(tensors[f"{key}.count"].item())
        expected_count = int(stats[key]["count"][0])
        if count != expected_count or count != 4816:
            raise RuntimeError(f"{path.name} {key} count mismatch: {count} != {expected_count}")
        for stat_name in ("mean", "std"):
            actual = tensors[f"{key}.{stat_name}"].numpy()
            expected = np.asarray(stats[key][stat_name], dtype=np.float32)
            if not np.allclose(actual, expected, atol=1e-6, rtol=1e-6):
                raise RuntimeError(f"{path.name} {key}.{stat_name} does not match train-only stats")
    report[path.name] = {"action_count": 4816, "state_count": 4816, "verified": True}
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
PY
touch "$RUN_ROOT/STATS_VERIFIED"

stage=evaluation
"$PYTHON" scripts/video_vam/evaluate_action_rmse.py \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --output "$RUN_ROOT/corrected-smolvla-rmse.json" \
    --dataset-root "$SOURCE_DATASET" \
    --smolvla-checkpoint "$CHECKPOINT" \
    --device cuda \
    --batch-size 1 \
    --seed 0
touch "$RUN_ROOT/EVALUATION_COMPLETE"

stage=summary
"$PYTHON" - "$RUN_ROOT/corrected-smolvla-rmse.json" "$RUN_ROOT/summary.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
metric = result["backends"]["smolvla"]
summary = {
    "protocol_version": result["protocol"]["protocol_version"],
    "sample_count": metric["sample_count"],
    "aggregate_rmse_mixed_units": metric["aggregate_rmse_deg"],
    "prior_leaked_aggregate_rmse_mixed_units": 14.8323011648351,
    "aggregate_delta_corrected_minus_leaked": metric["aggregate_rmse_deg"] - 14.8323011648351,
    "per_joint_rmse_mixed_units": metric["per_joint_rmse_deg"],
    "per_joint_units": metric["per_joint_units"],
    "executed_prefix": metric["executed_prefix"],
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

stage=verification
.venv/bin/ruff check \
    scripts/video_vam/prepare_smolvla_train_only_stats.py \
    src/lerobot/policies/vam/action_rmse.py
bash -n scripts/video_vam/run_smolvla_train_only_stats_queue.sh
df -h / > "$RUN_ROOT/final-disk.txt"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader \
    > "$RUN_ROOT/final-gpu-state.txt"
stage=complete
touch "$RUN_ROOT/COMPLETE"
printf "[%s] corrected SmolVLA rerun and frozen evaluation complete\n" "$(date --iso-8601=seconds)"
