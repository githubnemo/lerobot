#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/anton/lerobot-video-vam
HELPER="$REPO/scripts/video_vam/gpu_lock.sh"
TMP_ROOT=$(mktemp -d)
EVENTS="$TMP_ROOT/events"
trap 'rm -rf -- "$TMP_ROOT"' EXIT

acquirer() {
    local arm_name=$1 hold_seconds=$2
    (
        source "$HELPER"
        GPU_LOCK_FILE="$TMP_ROOT/gpu.lock"
        GPU_LOCK_HOLDER_FILE="$TMP_ROOT/gpu.lock.holder"
        GPU_LOCK_WAIT_SECONDS=0.1
        GPU_LOCK_FREE_POLLS=2
        GPU_LOCK_FREE_POLL_SECONDS=0.05
        nvidia-smi() { return 0; }
        acquire_gpu_lock "$arm_name"
        printf '%s-enter\n' "$arm_name" >> "$EVENTS"
        sleep "$hold_seconds"
        printf '%s-exit\n' "$arm_name" >> "$EVENTS"
    )
}

acquirer first 0.4 &
first_pid=$!
for _ in {1..100}; do
    [[ -s $EVENTS ]] && break
    sleep 0.01
done
[[ -s $EVENTS ]]
acquirer second 0.05 &
second_pid=$!
wait "$first_pid"
wait "$second_pid"

mapfile -t events < "$EVENTS"
expected=(first-enter first-exit second-enter second-exit)
if [[ ${events[*]} != "${expected[*]}" ]]; then
    printf 'serialization failed: got <%s>, expected <%s>\n' \
        "${events[*]}" "${expected[*]}" >&2
    exit 1
fi
printf 'GPU_LOCK_SERIALIZATION_OK events=%s\n' "${events[*]}"
