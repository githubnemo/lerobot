#!/usr/bin/env bash
# Train video-only Cosmos LoRA, rebuild adapted pool2 caches, then train SmolExpert.
set -Eeuo pipefail

REPO=/home/anton/lerobot-video-vam
PYTHON="$REPO/.venv/bin/python"
CACHE=/home/anton/.cache/video-vam
RUN_ROOT="$CACHE/runs/cosmos2b-videolora-smolexpert-20260828"
LORA_RUN="$CACHE/runs/cosmos2b-video-lora-20260828"
BEST_LORA="$LORA_RUN/best_lora.safetensors"
TRAIN_CACHE="$CACHE/cosmos-videolora-train0-31-stride3-pool2"
VAL_CACHE="$CACHE/cosmos-videolora-val32-39-stride20-pool2"
BASELINE_TRAIN_MANIFEST="$CACHE/cosmos-train0-31-stride3-sigma80-prefix-pool2/manifest.json"
BASELINE_VAL_MANIFEST="$CACHE/cosmos-rehearsal-stride20-sigma80-prefix-pool2/manifest.json"
SPLIT="$CACHE/splits/rehearsal-stride20.json"
DATASET_ROOT="$CACHE/cube-out-of-box-dataset"
CHECKPOINT="$CACHE/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
TOKENIZER="$CACHE/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
PROMPT="$CACHE/prompt-embeddings/cube-out-of-box-t5-11b.safetensors"
SESSION=cosmos-video-lora-20260828

cd "$REPO"
source scripts/video_vam/cosmos_cuda_env.sh
source scripts/video_vam/gpu_lock.sh
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$RUN_ROOT" "$LORA_RUN"
exec > >(tee -a "$RUN_ROOT/queue.log") 2>&1
stage=initializing
trap 'status=$?; set +e; printf "[%s] FAILED stage=%s status=%s\n" "$(date --iso-8601=seconds)" "$stage" "$status"; printf "%s\n" "$stage" > "$RUN_ROOT/FAILED"; exit "$status"' ERR

if [[ -f "$RUN_ROOT/COMPLETE" ]]; then
    printf "[%s] queue already complete; leaving marker intact and exiting\n" "$(date --iso-8601=seconds)"
    exit 0
fi
if [[ -f "$LORA_RUN/PAUSED" ]]; then
    printf "[%s] old video-LoRA queue paused via %s; refusing to retrain adapters\n" "$(date --iso-8601=seconds)" "$LORA_RUN/PAUSED"
    exit 0
fi
rm -f -- "$RUN_ROOT/FAILED"

TRAIN_EPISODES=({0..31})
VAL_EPISODES=({32..39})

disk_audit() {
    "$PYTHON" - "$BASELINE_TRAIN_MANIFEST" "$BASELINE_VAL_MANIFEST" "$TRAIN_CACHE" "$VAL_CACHE" <<'PY_INNER'
import json
import shutil
import sys
from pathlib import Path

baseline = [Path(value) for value in sys.argv[1:3]]
targets = [Path(value) for value in sys.argv[3:5]]
missing = [str(path) for path in baseline if not path.is_file()]
if missing:
    raise FileNotFoundError(f"baseline manifests missing for disk estimate: {missing}")
estimated = sum(json.loads(path.read_text())["total_bytes"] for path in baseline)
existing = sum(
    child.stat().st_size
    for root in targets
    if root.exists()
    for child in root.rglob("*")
    if child.is_file()
)
remaining = max(0, estimated - existing)
reserved = 20 * 2**30
overhead = 2 * 2**30
required = remaining + reserved + overhead
free = shutil.disk_usage("/").free
print(
    "DISK_AUDIT "
    f"free_gib={free / 2**30:.2f} estimated_cache_gib={estimated / 2**30:.2f} "
    f"existing_target_gib={existing / 2**30:.2f} remaining_gib={remaining / 2**30:.2f} "
    f"guard_floor_gib={reserved / 2**30:.2f} overhead_gib={overhead / 2**30:.2f} "
    f"required_gib={required / 2**30:.2f}"
)
if free < required:
    raise RuntimeError("refusing cache rebuild: free disk is below cache estimate plus guarded floor")
print("DISK_AUDIT_OK no deletion performed")
PY_INNER
    df -h /
}

sha256_file() {
    "$PYTHON" - "$1" <<'PY_INNER'
import hashlib
import sys
from pathlib import Path
path = Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY_INNER
}

verify_lora() {
    "$PYTHON" - "$BEST_LORA" <<'PY_INNER'
import hashlib
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
sidecar = path.with_suffix(".json")
if not path.is_file() or not sidecar.is_file():
    raise FileNotFoundError(f"best LoRA checkpoint or provenance sidecar missing: {path}")
payload = json.loads(sidecar.read_text())
lora = payload.get("lora")
if not isinstance(lora, dict) or lora.get("rank") != 16 or float(lora.get("alpha", 0)) != 16.0:
    raise ValueError("best LoRA provenance does not declare rank=16 and alpha=16")
digest = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        digest.update(chunk)
print(f"BEST_LORA path={path} sha256={digest.hexdigest()} rank={lora['rank']} alpha={lora['alpha']}")
PY_INNER
}

hash_audit() {
    "$PYTHON" - "$1" "$BEST_LORA" <<'PY_INNER'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
lora_path = Path(sys.argv[2])
payload = json.loads(manifest_path.read_text())
provenance = payload.get("provenance", {})
lora = provenance.get("lora_weights")
if not provenance.get("adapters_applied") or not isinstance(lora, dict):
    raise ValueError(f"manifest is not marked as adapter-applied: {manifest_path}")
digest = hashlib.sha256(lora_path.read_bytes()).hexdigest()
if lora.get("sha256") != digest:
    raise ValueError(f"manifest LoRA sha256 mismatch: {manifest_path}")
failures = []
for entry in payload["entries"]:
    for key in ("safetensors", "sidecar"):
        path = manifest_path.parent / entry[key]
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry[f"{key}_sha256"]:
            failures.append(f"hash mismatch {path}")
if failures:
    raise RuntimeError("; ".join(failures[:8]))
print(f"CACHE_HASH_AUDIT_OK manifest={manifest_path} samples={len(payload['entries'])} lora_sha256={digest}")
PY_INNER
}

run_lora_training() {
    "$PYTHON" -m scripts.video_vam.train_cosmos_video_lora \
        --manifest "$BASELINE_TRAIN_MANIFEST" \
        --val-manifest "$BASELINE_VAL_MANIFEST" \
        --split "$SPLIT" \
        --dataset-root "$DATASET_ROOT" \
        --backbone-checkpoint "$CHECKPOINT" \
        --tokenizer "$TOKENIZER" \
        --prompt "$PROMPT" \
        --output-dir "$LORA_RUN" \
        --latent-cache-dir "$LORA_RUN/latent-cache" \
        --train-episodes "${TRAIN_EPISODES[@]}" \
        --max-steps 500000 \
        --max-hours 12 \
        --batch-size 1 \
        --val-every 1000 \
        --save-every 1000 \
        --patience 10 \
        --min-delta 0 \
        --lora-rank 16 \
        --lora-alpha 16 \
        --lora-lr 1e-4 \
        --weight-decay 0.1 \
        --grad-clip 1 \
        --eval-sigma 80 \
        --seed 0 \
        --wandb-project video-vam-world2action \
        --run-name cosmos2b-video-lora-20260828 \
        --overwrite
}

build_cache() {
    local episodes=($1)
    local stride=$2
    local output=$3
    "$PYTHON" -m scripts.video_vam.build_cosmos_feature_cache \
        --root "$DATASET_ROOT" \
        --episodes "${episodes[@]}" \
        --stride "$stride" \
        --prompt "$PROMPT" \
        --checkpoint "$CHECKPOINT" \
        --lora-weights "$BEST_LORA" \
        --tokenizer "$TOKENIZER" \
        --output-dir "$output" \
        --sigma 80 \
        --seed 0 \
        --vae-input-mode observed_prefix \
        --context-transform pool2 \
        --overwrite
}

run_smolexpert() {
    "$PYTHON" -m scripts.video_vam.train_smolexpert_on_cosmos \
        --manifest "$TRAIN_CACHE/manifest.json" \
        --val-manifest "$VAL_CACHE/manifest.json" \
        --split "$SPLIT" \
        --train-episodes "${TRAIN_EPISODES[@]}" \
        --output-dir "$RUN_ROOT/smolexpert" \
        --expert-checkpoint lerobot/smolvla_base \
        --batch-size 8 \
        --max-steps 500000 \
        --max-hours 4 \
        --val-every 1000 \
        --patience 10 \
        --min-delta 0 \
        --lr 1e-4 \
        --weight-decay 1e-10 \
        --grad-clip 10 \
        --warmup-steps 1000 \
        --num-steps 10 \
        --context-transform auto \
        --seed 0 \
        --wandb-project video-vam-world2action \
        --run-name cosmos2b-videolora-pool2-smolexpert-20260828 \
        --overwrite
}

stage=acquire-gpu-lock
acquire_gpu_lock cosmos-video-lora-20260828
stage=disk-audit
disk_audit
stage=video-lora-training
run_lora_training
stage=verify-best-lora
verify_lora
sha256_file "$BEST_LORA" > "$LORA_RUN/best_lora.sha256"
stage=build-train-cache
build_cache "${TRAIN_EPISODES[*]}" 3 "$TRAIN_CACHE"
stage=build-val-cache
build_cache "${VAL_EPISODES[*]}" 20 "$VAL_CACHE"
stage=hash-audit
hash_audit "$TRAIN_CACHE/manifest.json"
hash_audit "$VAL_CACHE/manifest.json"
touch "$RUN_ROOT/CACHES_COMPLETE"
stage=smolexpert-training
run_smolexpert
touch "$RUN_ROOT/SMOLEXPERT_COMPLETE"
stage=complete
touch "$RUN_ROOT/COMPLETE"
printf "[%s] COMPLETE run_root=%s lora_root=%s\n" "$(date --iso-8601=seconds)" "$RUN_ROOT" "$LORA_RUN"
