#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/anton/lerobot-video-vam
PYTHON="$REPO/.venv/bin/python"
CACHE=/home/anton/.cache/video-vam
RUN_ROOT="$CACHE/runs/ltx25-mixer-variants-20260827"
TRAIN_CACHE="$CACHE/ltx25-layer-probe-train0-31-stride3-pool2"
VAL_CACHE="$CACHE/ltx25-layer-probe-val32-39-stride20-pool2"
SPLIT="$CACHE/splits/rehearsal-stride20.json"
NORMALIZER="$CACHE/runs/vam-hour-k8/normalizer.safetensors"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/queue.log") 2>&1
stage=initializing
trap 'status=$?; set +e; printf "[%s] FAILED stage=%s status=%s\n" "$(date --iso-8601=seconds)" "$stage" "$status"; printf "%s\n" "$stage" > "$RUN_ROOT/FAILED"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-at-failure.txt" || true; exit "$status"' ERR

rm -f -- "$RUN_ROOT/FAILED"
cd "$REPO"
source scripts/video_vam/cosmos_cuda_env.sh
source scripts/video_vam/gpu_lock.sh
export PYTHONHASHSEED=0

stage=disk-audit
"$PYTHON" - "$TRAIN_CACHE/manifest.json" "$VAL_CACHE/manifest.json" <<'PY_AUDIT'
import shutil
import sys
from pathlib import Path

manifests = [Path(value) for value in sys.argv[1:]]
missing = [str(path) for path in manifests if not path.is_file()]
if missing:
    raise FileNotFoundError(f"existing layer-mix cache manifests are missing: {missing}")
free_bytes = shutil.disk_usage("/").free
required_bytes = 2 * 9 * 2**30 + 20 * 2**30
print(
    "DISK_AUDIT "
    f"free_bytes={free_bytes} free_gib={free_bytes / 2**30:.2f} "
    f"required_bytes={required_bytes} required_gib={required_bytes / 2**30:.2f} "
    "basis=two_9GiB_checkpoint_reserves_plus_20GiB_floor caches=reused"
)
if free_bytes < required_bytes:
    raise RuntimeError("insufficient disk for two mixer runs while retaining the 20 GiB floor")
print("DISK_AUDIT_OK existing layer-mix caches will be reused; no cache build scheduled")
PY_AUDIT
df -h / > "$RUN_ROOT/disk-before.txt"

stage=acquire-gpu-lock
acquire_gpu_lock ltx-mixer-variants-20260827
touch "$RUN_ROOT/GPU_LOCK_ACQUIRED"

run_mixer() {
    local mixer=$1 output_dir=$2 wandb_name=$3
    shift 3
    printf '[%s] TRAIN_START mixer=%s output=%s\n' \
        "$(date --iso-8601=seconds)" "$mixer" "$output_dir"
    "$PYTHON" -m scripts.video_vam.train_ltx_layer_mix \
        --train-manifest "$TRAIN_CACHE/manifest.json" \
        --val-manifest "$VAL_CACHE/manifest.json" \
        --split "$SPLIT" \
        --normalizer "$NORMALIZER" \
        --output-dir "$output_dir" \
        --context-transform pool2 \
        --layers 8,14,20,26,34,40 \
        --max-steps 200000 \
        --max-hours 2 \
        --batch-size 2 \
        --grad-accum-steps 2 \
        --flow-draws 8 \
        --flow-draw-chunk 4 \
        --val-every 300 \
        --patience 20 \
        --min-delta 0 \
        --warmup-steps 200 \
        --lr 1e-4 \
        --weight-decay 0.1 \
        --grad-clip 10 \
        --seed 0 \
        --mixer "$mixer" \
        --wandb-project video-vam-world2action \
        --wandb-run-name "$wandb_name" \
        --overwrite \
        "$@"
    printf '[%s] TRAIN_COMPLETE mixer=%s output=%s\n' \
        "$(date --iso-8601=seconds)" "$mixer" "$output_dir"
}

stage=attention
run_mixer \
    attention \
    "$RUN_ROOT/attention" \
    ltx25-layer-attention-pool2-w2a-s0-20260827 \
    --attn-width 2048 \
    --attn-heads 8
touch "$RUN_ROOT/ATTENTION_COMPLETE"

stage=gated
run_mixer \
    gated \
    "$RUN_ROOT/gated" \
    ltx25-layer-gated-pool2-w2a-s0-20260827
touch "$RUN_ROOT/GATED_COMPLETE"

stage=complete
df -h / > "$RUN_ROOT/disk-after.txt"
touch "$RUN_ROOT/COMPLETE"
rm -f -- "$RUN_ROOT/FAILED"
printf '[%s] mixer variant queue complete\n' "$(date --iso-8601=seconds)"
