#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/anton/lerobot-video-vam
PYTHON="$REPO/.venv/bin/python"
CACHE=/home/anton/.cache/video-vam
RUN_ROOT="$CACHE/runs/cosmos-bridge-retry-20260826-rerun"
TRAIN_CACHE="$CACHE/cosmos-bridge-train0-31-stride3-pool2"
VAL_CACHE="$CACHE/cosmos-bridge-val32-39-stride20-pool2"
BASELINE_TRAIN_MANIFEST="$CACHE/cosmos-train0-31-stride3-sigma80-prefix-pool2/manifest.json"
BASELINE_VAL_MANIFEST="$CACHE/cosmos-rehearsal-stride20-sigma80-prefix-pool2/manifest.json"
SPLIT="$CACHE/splits/rehearsal-stride20.json"
BRIDGE_CHECKPOINT="$CACHE/mimic-video-f2833903/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt"
TRAINER=scripts.video_vam.train_smolexpert_on_cosmos
TRAIN_EPISODES=({0..31})

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/queue.log") 2>&1
stage=initializing
trap 'status=$?; set +e; printf "[%s] FAILED stage=%s status=%s\n" "$(date --iso-8601=seconds)" "$stage" "$status"; printf "%s\n" "$stage" > "$RUN_ROOT/FAILED"; exit "$status"' ERR

if [[ -f "$RUN_ROOT/COMPLETE" ]]; then
    printf "[%s] bridge queue already complete; leaving marker intact and exiting\n" \
        "$(date --iso-8601=seconds)"
    exit 0
fi
if [[ -f "$RUN_ROOT/FAILED" ]]; then
    printf "[%s] deleting stale retry marker %s because a new queue attempt is starting\n" \
        "$(date --iso-8601=seconds)" "$RUN_ROOT/FAILED"
    rm -f -- "$RUN_ROOT/FAILED"
fi

cd "$REPO"
source scripts/video_vam/cosmos_cuda_env.sh
source scripts/video_vam/gpu_lock.sh
export PYTHONHASHSEED=0

disk_audit() {
    "$PYTHON" - "$BASELINE_TRAIN_MANIFEST" "$BASELINE_VAL_MANIFEST" \
        "$TRAIN_CACHE" "$VAL_CACHE" <<'PY'
import json
import shutil
import sys
from pathlib import Path

baseline_paths = [Path(sys.argv[1]), Path(sys.argv[2])]
target_paths = [Path(sys.argv[3]), Path(sys.argv[4])]
missing = [str(path) for path in baseline_paths if not path.is_file()]
if missing:
    raise FileNotFoundError(f"baseline manifests missing for disk estimate: {missing}")
estimated_cache_bytes = sum(json.loads(path.read_text())["total_bytes"] for path in baseline_paths)
existing_target_bytes = sum(
    item.stat().st_size
    for root in target_paths
    if root.exists()
    for item in root.rglob("*")
    if item.is_file()
)
remaining_cache_bytes = max(0, estimated_cache_bytes - existing_target_bytes)
guard_floor_bytes = 20 * 2**30
overhead_bytes = 2 * 2**30
required_free_bytes = remaining_cache_bytes + guard_floor_bytes + overhead_bytes
free_bytes = shutil.disk_usage("/").free
print(
    "DISK_AUDIT "
    f"free_bytes={free_bytes} free_gib={free_bytes / 2**30:.2f} "
    f"estimated_cache_bytes={estimated_cache_bytes} "
    f"estimated_cache_gib={estimated_cache_bytes / 2**30:.2f} "
    f"existing_target_bytes={existing_target_bytes} "
    f"remaining_cache_bytes={remaining_cache_bytes} "
    f"guard_floor_gib={guard_floor_bytes / 2**30:.2f} "
    f"overhead_gib={overhead_bytes / 2**30:.2f} "
    f"required_free_bytes={required_free_bytes} "
    f"required_free_gib={required_free_bytes / 2**30:.2f}"
)
if free_bytes < required_free_bytes:
    raise RuntimeError(
        "refusing to build bridge caches: free disk is below the estimated remaining cache "
        "+ 20 GiB retained floor + 2 GiB overhead; no files were deleted"
    )
print("DISK_AUDIT_OK no deletion needed; bridge caches fit above the guarded floor")
PY
    df -h /
}

verify_manifest_hashes() {
    local manifest=$1 report=$2
    "$PYTHON" - "$manifest" "$report" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
payload = json.loads(manifest_path.read_text())
failures = []
for entry in payload["entries"]:
    tensor_path = manifest_path.parent / entry["safetensors"]
    sidecar_path = manifest_path.parent / entry["sidecar"]
    if not tensor_path.is_file():
        failures.append((tensor_path, sidecar_path, entry["safetensors_sha256"], "MISSING"))
        continue
    digest = hashlib.sha256()
    with tensor_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != entry["safetensors_sha256"]:
        failures.append((tensor_path, sidecar_path, entry["safetensors_sha256"], actual))
report_path.write_text(
    "".join(f"{tensor}\t{sidecar}\t{expected}\t{actual}\n" for tensor, sidecar, expected, actual in failures)
)
print(
    f"HASH_AUDIT manifest={manifest_path} entries={len(payload['entries'])} "
    f"mismatches={len(failures)}"
)
for tensor, _sidecar, expected, actual in failures:
    print(f"HASH_MISMATCH file={tensor} expected={expected} actual={actual}")
if failures:
    raise SystemExit(42)
PY
}

print_and_check_provenance() {
    local manifest=$1
    "$PYTHON" - "$manifest" "$BRIDGE_CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
expected_checkpoint = str(Path(sys.argv[2]).resolve())
payload = json.loads(manifest_path.read_text())
provenance = dict(payload["provenance"])
selection = dict(provenance.get("selection", {}))
if "ordered_pairs" in selection:
    selection["ordered_pair_count"] = len(selection.pop("ordered_pairs"))
provenance["selection"] = selection
print(f"MANIFEST_PROVENANCE {manifest_path}")
print(json.dumps(provenance, indent=2, sort_keys=True))
weights = provenance.get("weights")
if not isinstance(weights, dict) or weights.get("checkpoint_kind") != "bridge_fused":
    raise RuntimeError("ABORT: manifest checkpoint_kind is not bridge_fused")
bridge = weights.get("bridge_lora")
if (
    not isinstance(bridge, dict)
    or bridge.get("checkpoint_path") != expected_checkpoint
    or bridge.get("fused") is not True
    or bridge.get("rank") != 256
    or bridge.get("checkpoint_sha256") != weights.get("checkpoint_sha256")
):
    raise RuntimeError("ABORT: bridge LoRA provenance is absent or inconsistent")
print(
    "BRIDGE_PROVENANCE_CONFIRMED "
    f"checkpoint_kind={weights['checkpoint_kind']} "
    f"bridge_lora_recorded=true checkpoint={bridge['checkpoint_path']} "
    f"sha256={bridge['checkpoint_sha256']}"
)
PY
}

run_smoke() {
    printf "[%s] SMOKE_COMMAND: %q -m %q --manifest %q --val-manifest %q --split %q --train-episodes 0..31 --output-dir %q --expert-checkpoint lerobot/smolvla_base --batch-size 8 --max-steps 200 --val-every 200 --patience 1 --min-delta 0 --lr 1e-4 --weight-decay 1e-10 --grad-clip 10 --warmup-steps 100 --num-steps 10 --context-transform auto --seed 0 --no-wandb --no-save-checkpoints --overwrite\n" \
        "$(date --iso-8601=seconds)" "$PYTHON" "$TRAINER" "$TRAIN_CACHE/manifest.json" \
        "$VAL_CACHE/manifest.json" "$SPLIT" "$RUN_ROOT/smoke"
    "$PYTHON" -m "$TRAINER" \
        --manifest "$TRAIN_CACHE/manifest.json" \
        --val-manifest "$VAL_CACHE/manifest.json" \
        --split "$SPLIT" \
        --train-episodes "${TRAIN_EPISODES[@]}" \
        --output-dir "$RUN_ROOT/smoke" \
        --expert-checkpoint lerobot/smolvla_base \
        --batch-size 8 \
        --max-steps 200 \
        --val-every 200 \
        --patience 1 \
        --min-delta 0 \
        --lr 1e-4 \
        --weight-decay 1e-10 \
        --grad-clip 10 \
        --warmup-steps 100 \
        --num-steps 10 \
        --context-transform auto \
        --seed 0 \
        --no-wandb \
        --no-save-checkpoints \
        --overwrite
}

run_training() {
    printf "[%s] TRAIN_COMMAND: %q -m %q --manifest %q --val-manifest %q --split %q --train-episodes 0..31 --output-dir %q --expert-checkpoint lerobot/smolvla_base --batch-size 8 --max-steps 500000 --max-hours 4 --val-every 1000 --patience 10 --min-delta 0 --lr 1e-4 --weight-decay 1e-10 --grad-clip 10 --warmup-steps 1000 --num-steps 10 --context-transform auto --seed 0 --wandb-project video-vam-world2action --run-name cosmos2b-bridge-pool2-smolexpert-20260826-rerun --overwrite\n" \
        "$(date --iso-8601=seconds)" "$PYTHON" "$TRAINER" "$TRAIN_CACHE/manifest.json" \
        "$VAL_CACHE/manifest.json" "$SPLIT" "$RUN_ROOT/train"
    "$PYTHON" -m "$TRAINER" \
        --manifest "$TRAIN_CACHE/manifest.json" \
        --val-manifest "$VAL_CACHE/manifest.json" \
        --split "$SPLIT" \
        --train-episodes "${TRAIN_EPISODES[@]}" \
        --output-dir "$RUN_ROOT/train" \
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
        --run-name cosmos2b-bridge-pool2-smolexpert-20260826-rerun \
        --overwrite
}

stage=disk-audit
disk_audit

stage=train-cache-hash-audit
verify_manifest_hashes "$TRAIN_CACHE/manifest.json" "$RUN_ROOT/train-hash-mismatches.tsv"
stage=train-cache-provenance
print_and_check_provenance "$TRAIN_CACHE/manifest.json"

stage=val-cache-hash-audit
verify_manifest_hashes "$VAL_CACHE/manifest.json" "$RUN_ROOT/val-hash-mismatches.tsv"
stage=val-cache-provenance
print_and_check_provenance "$VAL_CACHE/manifest.json"

stage=acquire-gpu-lock
acquire_gpu_lock cosmos-bridge-retry-20260826-rerun

stage=smoke
run_smoke
touch "$RUN_ROOT/SMOKE_OK"

stage=train
run_training
touch "$RUN_ROOT/TRAIN_COMPLETE"

stage=complete
touch "$RUN_ROOT/COMPLETE"
rm -f -- "$RUN_ROOT/FAILED"
printf "[%s] bridge queue complete\n" "$(date --iso-8601=seconds)"
