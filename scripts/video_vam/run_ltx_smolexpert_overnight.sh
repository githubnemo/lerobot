#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/anton/lerobot-video-vam
PYTHON="$REPO/.venv/bin/python"
CACHE=/home/anton/.cache/video-vam
RUN_ROOT="$CACHE/runs/ltx25-smolexpert-20260826"
ACTIVE_SESSION=ltx25-plateau-20260825
TRAINER=scripts.video_vam.train_smolexpert_on_cosmos
BUILD=scripts.video_vam.build_ltx_feature_cache
TRAIN_POOL2="$CACHE/ltx25-train0-31-stride3-pool2/manifest.json"
VAL_POOL2="$CACHE/ltx25-val32-39-stride20-pool2/manifest.json"
TRAIN_RAW="$CACHE/ltx25-train0-31-stride3-unpooled/manifest.json"
VAL_RAW="$CACHE/ltx25-val32-39-stride20-unpooled/manifest.json"
SPLIT="$CACHE/splits/rehearsal-stride20.json"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/queue.log") 2>&1
stage=waiting
trap 'status=$?; printf "[%s] FAILED stage=%s status=%s\n" "$(date --iso-8601=seconds)" "$stage" "$status"; printf "%s\n" "$stage" > "$RUN_ROOT/FAILED"; exit "$status"' ERR

gpu_idle() {
    local utilization memory seen=0
    while IFS=',' read -r utilization memory; do
        seen=1
        utilization=${utilization// /}
        memory=${memory// /}
        [[ "$utilization" =~ ^[0-9]+$ && "$memory" =~ ^[0-9]+$ ]] || return 1
        (( utilization <= 5 && memory <= 1000 )) || return 1
    done < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
    (( seen == 1 ))
}

wait_for_idle_gpu() {
    local consecutive=0
    while (( consecutive < 3 )); do
        if gpu_idle; then
            ((consecutive += 1))
        else
            consecutive=0
        fi
        sleep 20
    done
}

run_train() {
    local output=$1 name=$2 train_manifest=$3 val_manifest=$4 max_hours=$5
    "$PYTHON" -m "$TRAINER" \
        --manifest "$train_manifest" \
        --val-manifest "$val_manifest" \
        --split "$SPLIT" \
        --output-dir "$output" \
        --batch-size 8 \
        --max-steps 500000 \
        --max-hours "$max_hours" \
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
        --run-name "$name" \
        --overwrite
}

run_smoke() {
    local output=$1 train_manifest=$2 val_manifest=$3
    "$PYTHON" -m "$TRAINER" \
        --manifest "$train_manifest" \
        --val-manifest "$val_manifest" \
        --split "$SPLIT" \
        --output-dir "$output" \
        --batch-size 2 \
        --max-steps 2 \
        --val-every 2 \
        --patience 1 \
        --warmup-steps 1 \
        --num-steps 2 \
        --seed 0 \
        --no-wandb \
        --no-save-checkpoints \
        --overwrite
}

free_completed_smoke_optimizer_states() {
    local run state free required
    local -a runs=(
        "$CACHE/runs/night2-20260819/smoke-runA"
        "$CACHE/runs/night2-20260819/smoke-runB"
        "$CACHE/runs/night3-20260819/smoke-cached"
    )
    : > "$RUN_ROOT/disk-actions.log"
    free=$(df --output=avail -B1 / | awk 'NR == 2 {print $1}')
    required=$((1660 * (2400 * 4096 * 2 + (6 + 30 * 6) * 4 + 30 + 16384) + 22 * 1024 * 1024 * 1024))
    printf "[%s] raw-cache disk audit: free=%s required=%s\n" \
        "$(date --iso-8601=seconds)" "$free" "$required" | tee -a "$RUN_ROOT/disk-actions.log"
    if (( free >= required )); then
        printf "[%s] no deletion needed; full stride-3 cache fits guarded floor\n" \
            "$(date --iso-8601=seconds)" | tee -a "$RUN_ROOT/disk-actions.log"
        df -h / | tee -a "$RUN_ROOT/disk-actions.log"
        return
    fi
    for run in "${runs[@]}"; do
        [[ -s "$run/best.safetensors" ]]
        for state in "$run/best.state.pt" "$run/last.state.pt"; do
            if [[ -f "$state" ]]; then
                printf "[%s] deleting completed smoke optimizer state only: %s (%s bytes)\n" \
                    "$(date --iso-8601=seconds)" "$state" "$(stat -c %s "$state")" |
                    tee -a "$RUN_ROOT/disk-actions.log"
                rm -- "$state"
                free=$(df --output=avail -B1 / | awk 'NR == 2 {print $1}')
                if (( free >= required )); then
                    df -h / | tee -a "$RUN_ROOT/disk-actions.log"
                    return
                fi
            fi
        done
    done
    printf "insufficient disk after exhausting approved optimizer-state candidates\n" >&2
    return 1
}

cd "$REPO"
source scripts/video_vam/cosmos_cuda_env.sh
export PYTHONHASHSEED=0

# Wait on the prior training PROCESS, not on its tmux session: that session
# keeps an interactive shell after its queue script exits, so a
# has-session predicate never terminates.
printf "[%s] waiting for prior LTX training process to exit\n" \
    "$(date --iso-8601=seconds)"
while pgrep -f "scripts/video_vam/train_ltx_world2action.py" >/dev/null 2>&1; do
    sleep 60
done
printf "[%s] prior training exited; waiting for stable idle GPU\n" "$(date --iso-8601=seconds)"
wait_for_idle_gpu

stage=pool2-smoke
printf "[%s] starting %s\n" "$(date --iso-8601=seconds)" "$stage"
run_smoke "$RUN_ROOT/pool2-smoke" "$TRAIN_POOL2" "$VAL_POOL2"
touch "$RUN_ROOT/POOL2_SMOKE_OK"

stage=run-a
printf "[%s] starting %s\n" "$(date --iso-8601=seconds)" "$stage"
run_train "$RUN_ROOT/run-a-pool2" \
    ltx25-pool2-smolexpert-plateau-20260826 "$TRAIN_POOL2" "$VAL_POOL2" 10
"$PYTHON" scripts/video_vam/report_ltx_expert_result.py \
    --result "$RUN_ROOT/run-a-pool2/result.json" \
    --arm "LTX-2.5 pool2 + pretrained SmolVLA expert" \
    --confound "none; same stride-3/stride-20 anchors as the LTX World2Action arm"
touch "$RUN_ROOT/RUN_A_COMPLETE"

stage=disk-cleanup
printf "[%s] starting %s\n" "$(date --iso-8601=seconds)" "$stage"
free_completed_smoke_optimizer_states

stage=unpooled-cache
printf "[%s] starting %s\n" "$(date --iso-8601=seconds)" "$stage"
"$PYTHON" -m "$BUILD" \
    --train-output-dir "${TRAIN_RAW%/manifest.json}" \
    --val-output-dir "${VAL_RAW%/manifest.json}" \
    --train-stride 3 \
    --val-stride 20 \
    --context-transform none \
    --min-free-gib 20 \
    --seed 0 \
    --resume
touch "$RUN_ROOT/UNPOOLED_CACHE_COMPLETE"

stage=unpooled-smoke
printf "[%s] starting %s\n" "$(date --iso-8601=seconds)" "$stage"
run_smoke "$RUN_ROOT/unpooled-smoke" "$TRAIN_RAW" "$VAL_RAW"
touch "$RUN_ROOT/UNPOOLED_SMOKE_OK"

stage=run-b
printf "[%s] starting %s\n" "$(date --iso-8601=seconds)" "$stage"
run_train "$RUN_ROOT/run-b-unpooled" \
    ltx25-unpooled-stride3-smolexpert-plateau-20260826 "$TRAIN_RAW" "$VAL_RAW" 14
"$PYTHON" scripts/video_vam/report_ltx_expert_result.py \
    --result "$RUN_ROOT/run-b-unpooled/result.json" \
    --arm "LTX-2.5 unpooled + pretrained SmolVLA expert" \
    --confound "none; disk guard retained the full stride-3 train cache"
touch "$RUN_ROOT/RUN_B_COMPLETE"

stage=complete
df -h / | tee -a "$RUN_ROOT/disk-actions.log"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader |
    tee "$RUN_ROOT/final-gpu-state.txt"
touch "$RUN_ROOT/COMPLETE"
printf "[%s] all stages complete\n" "$(date --iso-8601=seconds)"
