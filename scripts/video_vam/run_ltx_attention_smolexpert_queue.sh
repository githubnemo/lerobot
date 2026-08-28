#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/anton/lerobot-video-vam
PYTHON="$REPO/.venv/bin/python"
CACHE=/home/anton/.cache/video-vam
RUN_ROOT="$CACHE/runs/ltx25-attention-smolexpert-20260827"
TRAIN_MANIFEST="$CACHE/ltx25-layer-probe-train0-31-stride3-pool2/manifest.json"
VAL_MANIFEST="$CACHE/ltx25-layer-probe-val32-39-stride20-pool2/manifest.json"
SPLIT="$CACHE/splits/rehearsal-stride20.json"
RUN_NAME=ltx25-attention-smolexpert-pool2-s0-20260827

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/queue.log") 2>&1
stage=setup

on_error() {
    local status=$?
    trap - ERR
    printf '[%s] FAILED stage=%s status=%s\n' "$(date --iso-8601=seconds)" "$stage" "$status"
    printf 'stage=%s\nstatus=%s\ntime=%s\n' \
        "$stage" "$status" "$(date --iso-8601=seconds)" > "$RUN_ROOT/FAILED"
    exit "$status"
}
trap on_error ERR

cd "$REPO"
source scripts/video_vam/cosmos_cuda_env.sh
source scripts/video_vam/gpu_lock.sh
export PYTHONHASHSEED=0

stage=gpu-lock
acquire_gpu_lock ltx25-attention-smolexpert-20260827

stage=disk-audit
free_bytes=$(df --output=avail -B1 "$RUN_ROOT" | awk 'NR == 2 {print $1}')
required_bytes=$((12 * 1024 * 1024 * 1024))
printf '[%s] disk audit free_bytes=%s required_bytes=%s\n' \
    "$(date --iso-8601=seconds)" "$free_bytes" "$required_bytes"
df -h "$RUN_ROOT"
du -sh "${TRAIN_MANIFEST%/manifest.json}" "${VAL_MANIFEST%/manifest.json}"
if (( free_bytes < required_bytes )); then
    printf 'disk audit failed: less than 12 GiB free\n' >&2
    exit 72
fi
touch "$RUN_ROOT/DISK_AUDIT_OK"

stage=cache-verification
"$PYTHON" - "$TRAIN_MANIFEST" "$VAL_MANIFEST" <<'PY'
from pathlib import Path
import sys

from lerobot.policies.vam.ltx_layer_mix_cache import (
    LTXLayerMixCacheDataset,
    load_layer_mix_manifest,
    sha256_file,
)

expected = (
    (Path(sys.argv[1]), 1_572, "train"),
    (Path(sys.argv[2]), 88, "validation"),
)
for path, count, label in expected:
    manifest = load_layer_mix_manifest(path)
    if len(manifest.entries) != count:
        raise ValueError(f"{label} cache count is {len(manifest.entries)}, expected {count}")
    dataset = LTXLayerMixCacheDataset(manifest)
    for index, entry in enumerate(manifest.entries, 1):
        dataset.load_entry(entry)
        if index % 100 == 0 or index == count:
            print(f"verified {label} cache {index}/{count}", flush=True)
    print(f"{label}_manifest_sha256={sha256_file(path)}", flush=True)
PY
touch "$RUN_ROOT/CACHE_HASHES_OK"

stage=training
"$PYTHON" -m scripts.video_vam.train_smolexpert_on_cosmos \
    --manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --split "$SPLIT" \
    --output-dir "$RUN_ROOT" \
    --mixer attention \
    --attn-width 2048 \
    --attn-heads 8 \
    --batch-size 2 \
    --grad-accum-steps 4 \
    --max-steps 500000 \
    --max-hours 12 \
    --val-every 1000 \
    --patience 10 \
    --min-delta 0.02 \
    --lr 1e-4 \
    --weight-decay 1e-10 \
    --grad-clip 10 \
    --warmup-steps 1000 \
    --num-steps 10 \
    --seed 0 \
    --wandb-project video-vam-world2action \
    --run-name "$RUN_NAME" \
    --overwrite

stage=complete
rm -f -- "$RUN_ROOT/FAILED"
touch "$RUN_ROOT/COMPLETE"
printf '[%s] training queue complete\n' "$(date --iso-8601=seconds)"
