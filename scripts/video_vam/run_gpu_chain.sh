#!/usr/bin/env bash
set -Euo pipefail

REPO=/home/anton/lerobot-video-vam
CACHE=/home/anton/.cache/video-vam
RUN_ROOT="$CACHE/runs/gpu-chain-20260826"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/progress.log") 2>&1
cd "$REPO"
source scripts/video_vam/cosmos_cuda_env.sh

failures=0
run_arm() {
    local arm_name=$1 script=$2 status
    printf '[%s] GPU_CHAIN_START arm=%s script=%s\n' \
        "$(date --iso-8601=seconds)" "$arm_name" "$script"
    if bash "$script"; then
        printf '[%s] GPU_CHAIN_COMPLETE arm=%s\n' \
            "$(date --iso-8601=seconds)" "$arm_name"
    else
        status=$?
        failures=$((failures + 1))
        printf '[%s] GPU_CHAIN_FAILED arm=%s status=%s; continuing\n' \
            "$(date --iso-8601=seconds)" "$arm_name" "$status"
    fi
}

run_arm ltx-layer-probe scripts/video_vam/run_ltx_layer_probe_queue.sh
run_arm wrapper-validation scripts/video_vam/run_wrapper_validation_queue.sh
run_arm cosmos-bridge-retry scripts/video_vam/run_cosmos_bridge_retry_queue.sh

printf '[%s] GPU_CHAIN_FINISHED failures=%s\n' \
    "$(date --iso-8601=seconds)" "$failures"
(( failures == 0 ))
