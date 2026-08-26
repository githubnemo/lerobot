#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/anton/lerobot-video-vam
PYTHON="$REPO/.venv/bin/python"
CACHE=/home/anton/.cache/video-vam
RUN_ROOT="$CACHE/runs/ltx25-layer-probe-20260826"
TRAIN_CACHE="$CACHE/ltx25-layer-probe-train0-31-stride3-pool2"
VAL_CACHE="$CACHE/ltx25-layer-probe-val32-39-stride20-pool2"
SPLIT="$CACHE/splits/rehearsal-stride20.json"
NORMALIZER="$CACHE/runs/vam-hour-k8/normalizer.safetensors"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/queue.log") 2>&1
stage=initializing
trap 'status=$?; printf "[%s] FAILED stage=%s status=%s\n" "$(date --iso-8601=seconds)" "$stage" "$status"; printf "%s\n" "$stage" > "$RUN_ROOT/FAILED"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader > "$RUN_ROOT/gpu-at-failure.txt" || true; exit "$status"' ERR

cd "$REPO"
source scripts/video_vam/cosmos_cuda_env.sh
source scripts/video_vam/gpu_lock.sh
export PYTHONHASHSEED=0

stage=acquire-gpu-lock
acquire_gpu_lock ltx-layer-probe-20260826
touch "$RUN_ROOT/GPU_LOCK_ACQUIRED"

stage=extraction-cost
"$PYTHON" -m scripts.video_vam.benchmark_ltx_layer_mix \
    --iterations 3 \
    --output "$RUN_ROOT/extraction-cost.json"
touch "$RUN_ROOT/EXTRACTION_COST_COMPLETE"

stage=cache
cache_complete=0
for attempt in 1 2; do
    cache_log="$RUN_ROOT/cache-pool2-attempt${attempt}.log"
    if "$PYTHON" -m scripts.video_vam.build_ltx_layer_mix_cache \
        --train-output-dir "$TRAIN_CACHE" \
        --val-output-dir "$VAL_CACHE" \
        --train-stride 3 \
        --val-stride 20 \
        --context-transform pool2 \
        --min-free-gib 20 \
        --seed 0 \
        --resume 2>&1 | tee "$cache_log"; then
        cache_complete=1
        break
    fi
    status=${PIPESTATUS[0]}
    if rg -q "deleted [0-9]+ corrupt cache tensor/sidecar pairs" "$cache_log"; then
        printf "[%s] cache hash audit found corruption; retrying with --resume\n" \
            "$(date --iso-8601=seconds)"
    else
        exit "$status"
    fi
done
(( cache_complete == 1 ))
touch "$RUN_ROOT/CACHE_COMPLETE"

stage=full-mix
TRAIN_BATCH_SIZE=0
TRAIN_GRAD_ACCUM=0
TRAIN_FLOW_CHUNK=0
for setup in "2 2 4" "2 2 2" "2 2 1" "1 4 1"; do
    read -r batch_size grad_accum flow_chunk <<<"$setup"
    attempt_log="$RUN_ROOT/mix-pool2-b${batch_size}-ga${grad_accum}-chunk${flow_chunk}.log"
    if "$PYTHON" -m scripts.video_vam.train_ltx_layer_mix \
        --train-manifest "$TRAIN_CACHE/manifest.json" \
        --val-manifest "$VAL_CACHE/manifest.json" \
        --split "$SPLIT" \
        --normalizer "$NORMALIZER" \
        --output-dir "$RUN_ROOT/mix" \
        --context-transform pool2 \
        --layers 8,14,20,26,34,40 \
        --max-steps 200000 \
        --max-hours 72 \
        --batch-size "$batch_size" \
        --grad-accum-steps "$grad_accum" \
        --flow-draws 8 \
        --flow-draw-chunk "$flow_chunk" \
        --val-every 300 \
        --patience 20 \
        --min-delta 0 \
        --warmup-steps 200 \
        --lr 1e-4 \
        --weight-decay 0.1 \
        --grad-clip 10 \
        --seed 0 \
        --wandb-project video-vam-world2action \
        --wandb-run-name ltx25-layer-mix-pool2-w2a-s0-20260826 \
        --overwrite 2>&1 | tee "$attempt_log"; then
        TRAIN_BATCH_SIZE=$batch_size
        TRAIN_GRAD_ACCUM=$grad_accum
        TRAIN_FLOW_CHUNK=$flow_chunk
        break
    fi
    status=${PIPESTATUS[0]}
    if rg -q "CUDA OOM" "$attempt_log"; then
        printf "[%s] pool2 OOM with batch=%s grad_accum=%s chunk=%s; trying lower-memory setup\n" \
            "$(date --iso-8601=seconds)" "$batch_size" "$grad_accum" "$flow_chunk"
    else
        exit "$status"
    fi
done
if (( TRAIN_BATCH_SIZE == 0 )); then
    printf "pool2 failed at chunk=1,batch=1; measured attempts are in %s/mix-pool2-*.log\n" "$RUN_ROOT"
    exit 70
fi
printf "POOL2_OPTIMIZATION_SETUP=batch_size=%s grad_accum_steps=%s effective_batch=4 flow_draws=8 flow_draw_chunk=%s\n" \
    "$TRAIN_BATCH_SIZE" "$TRAIN_GRAD_ACCUM" "$TRAIN_FLOW_CHUNK" | tee "$RUN_ROOT/optimization-setup.txt"
touch "$RUN_ROOT/MIX_COMPLETE"

stage=select-layers
"$PYTHON" - "$RUN_ROOT/mix/result.json" "$RUN_ROOT/selected_layers.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
weights = result["best_mix_diagnostics"]["weights"]
ranked = sorted((int(layer) for layer in weights), key=lambda layer: (-float(weights[str(layer)]), layer))
selection = {"ranking_by_softmax_weight": ranked, "top1": ranked[0], "top2": sorted(ranked[:2])}
Path(sys.argv[2]).write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
print("LAYER_SELECTION=" + json.dumps(selection, sort_keys=True), flush=True)
PY
read -r TOP1 TOP2A TOP2B < <(
    "$PYTHON" - "$RUN_ROOT/selected_layers.json" <<'PY'
import json
import sys
from pathlib import Path

selection = json.loads(Path(sys.argv[1]).read_text())
print(selection["top1"], *selection["top2"])
PY
)

stage=top1-confirmation
"$PYTHON" -m scripts.video_vam.train_ltx_layer_mix \
    --train-manifest "$TRAIN_CACHE/manifest.json" \
    --val-manifest "$VAL_CACHE/manifest.json" \
    --split "$SPLIT" \
    --normalizer "$NORMALIZER" \
    --output-dir "$RUN_ROOT/top1" \
    --context-transform pool2 \
    --layers "$TOP1" \
    --max-steps 30000 \
    --max-hours 12 \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --grad-accum-steps "$TRAIN_GRAD_ACCUM" \
    --flow-draws 8 \
    --flow-draw-chunk "$TRAIN_FLOW_CHUNK" \
    --val-every 300 \
    --patience 10 \
    --min-delta 0 \
    --warmup-steps 200 \
    --lr 1e-4 \
    --weight-decay 0.1 \
    --grad-clip 10 \
    --seed 0 \
    --wandb-project video-vam-world2action \
    --wandb-run-name "ltx25-layer-top1-l${TOP1}-pool2-w2a-s0-20260826" \
    --overwrite
touch "$RUN_ROOT/TOP1_COMPLETE"

stage=top2-confirmation
"$PYTHON" -m scripts.video_vam.train_ltx_layer_mix \
    --train-manifest "$TRAIN_CACHE/manifest.json" \
    --val-manifest "$VAL_CACHE/manifest.json" \
    --split "$SPLIT" \
    --normalizer "$NORMALIZER" \
    --output-dir "$RUN_ROOT/top2" \
    --context-transform pool2 \
    --layers "${TOP2A},${TOP2B}" \
    --max-steps 30000 \
    --max-hours 12 \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --grad-accum-steps "$TRAIN_GRAD_ACCUM" \
    --flow-draws 8 \
    --flow-draw-chunk "$TRAIN_FLOW_CHUNK" \
    --val-every 300 \
    --patience 10 \
    --min-delta 0 \
    --warmup-steps 200 \
    --lr 1e-4 \
    --weight-decay 0.1 \
    --grad-clip 10 \
    --seed 0 \
    --wandb-project video-vam-world2action \
    --wandb-run-name "ltx25-layer-top2-l${TOP2A}-l${TOP2B}-pool2-w2a-s0-20260826" \
    --overwrite
touch "$RUN_ROOT/TOP2_COMPLETE"

stage=documentation
"$PYTHON" -m scripts.video_vam.report_ltx_layer_probe --run-root "$RUN_ROOT" --repo "$REPO"
touch "$RUN_ROOT/DOCS_COMPLETE"

stage=verification
.venv/bin/ruff check \
    src/lerobot/policies/vam/ltx_extractor.py \
    src/lerobot/policies/vam/ltx_layer_mix.py \
    src/lerobot/policies/vam/ltx_layer_mix_cache.py \
    scripts/video_vam/benchmark_ltx_layer_mix.py \
    scripts/video_vam/build_ltx_layer_mix_cache.py \
    scripts/video_vam/train_ltx_layer_mix.py \
    scripts/video_vam/report_ltx_layer_probe.py \
    tests/policies/test_ltx_layer_mix.py
git diff --check
df -h / > "$RUN_ROOT/final-disk.txt"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader \
    > "$RUN_ROOT/final-gpu-state.txt"
stage=complete
touch "$RUN_ROOT/COMPLETE"
printf "[%s] layer probe complete\n" "$(date --iso-8601=seconds)"
