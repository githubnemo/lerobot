#!/usr/bin/env bash
# Source this helper, then call acquire_gpu_lock <arm-name> before GPU work.

if [[ ${GPU_LOCK_HELPER_SOURCED:-0} == 1 ]]; then
    return 0 2>/dev/null || exit 0
fi
GPU_LOCK_HELPER_SOURCED=1

GPU_LOCK_FILE=${GPU_LOCK_FILE:-/home/anton/.cache/video-vam/gpu.lock}
GPU_LOCK_HOLDER_FILE=${GPU_LOCK_HOLDER_FILE:-/home/anton/.cache/video-vam/gpu.lock.holder}
GPU_LOCK_WAIT_SECONDS=${GPU_LOCK_WAIT_SECONDS:-30}
GPU_LOCK_FREE_POLLS=${GPU_LOCK_FREE_POLLS:-3}
GPU_LOCK_FREE_POLL_SECONDS=${GPU_LOCK_FREE_POLL_SECONDS:-3}
GPU_LOCK_HELD=0

_gpu_lock_timestamp() {
    date --iso-8601=seconds
}

_gpu_lock_cleanup() {
    local recorded_pid=""
    if [[ ${GPU_LOCK_HELD:-0} == 1 && -f ${GPU_LOCK_HOLDER_FILE:-} ]]; then
        while IFS='=' read -r key value; do
            if [[ $key == pid ]]; then
                recorded_pid=$value
                break
            fi
        done < "$GPU_LOCK_HOLDER_FILE"
        if [[ $recorded_pid == ${GPU_LOCK_HOLDER_PID:-} ]]; then
            rm -f -- "$GPU_LOCK_HOLDER_FILE"
        fi
    fi
}

_gpu_lock_fail() {
    local message=$1
    printf '[%s] GPU_LOCK_ERROR %s\n' "$(_gpu_lock_timestamp)" "$message" >&2
    _gpu_lock_cleanup
    GPU_LOCK_HELD=0
    if [[ -n ${GPU_LOCK_FD:-} ]]; then
        eval "exec ${GPU_LOCK_FD}>&-"
    fi
    return 70
}

acquire_gpu_lock() {
    if (( $# != 1 )); then
        printf 'usage: acquire_gpu_lock <arm-name>\n' >&2
        return 64
    fi
    local arm_name=$1
    if [[ ! $arm_name =~ ^[A-Za-z0-9._:-]+$ ]]; then
        printf 'invalid GPU arm name: %q\n' "$arm_name" >&2
        return 64
    fi
    if [[ ${GPU_LOCK_HELD:-0} == 1 ]]; then
        printf '[%s] GPU lock already held by arm=%s pid=%s\n' \
            "$(_gpu_lock_timestamp)" "$arm_name" "$GPU_LOCK_HOLDER_PID"
        return 0
    fi
    if ! command -v flock >/dev/null 2>&1; then
        printf '[%s] GPU_LOCK_ERROR flock is required but was not found\n' \
            "$(_gpu_lock_timestamp)" >&2
        return 69
    fi

    mkdir -p -- "$(dirname "$GPU_LOCK_FILE")"
    exec {GPU_LOCK_FD}>"$GPU_LOCK_FILE"
    printf '[%s] waiting for GPU lock arm=%s lock=%s\n' \
        "$(_gpu_lock_timestamp)" "$arm_name" "$GPU_LOCK_FILE"
    while ! flock -w "$GPU_LOCK_WAIT_SECONDS" "$GPU_LOCK_FD"; do
        local holder='unavailable'
        if [[ -s $GPU_LOCK_HOLDER_FILE ]]; then
            holder=$(tr '\n' ' ' < "$GPU_LOCK_HOLDER_FILE")
        fi
        printf '[%s] still waiting for GPU lock arm=%s holder={%s}\n' \
            "$(_gpu_lock_timestamp)" "$arm_name" "$holder"
    done

    GPU_LOCK_HELD=1
    GPU_LOCK_HOLDER_PID=$BASHPID
    printf 'arm=%s\npid=%s\nacquired_at=%s\nhost=%s\n' \
        "$arm_name" "$GPU_LOCK_HOLDER_PID" "$(_gpu_lock_timestamp)" "$(hostname)" \
        > "$GPU_LOCK_HOLDER_FILE"
    trap _gpu_lock_cleanup EXIT
    printf '[%s] acquired GPU lock arm=%s pid=%s fd=%s\n' \
        "$(_gpu_lock_timestamp)" "$arm_name" "$GPU_LOCK_HOLDER_PID" "$GPU_LOCK_FD"

    local poll compute_apps
    local -a busy_observations=()
    for ((poll = 1; poll <= GPU_LOCK_FREE_POLLS; poll += 1)); do
        if ! compute_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1); then
            _gpu_lock_fail "nvidia-smi compute-process query failed: $compute_apps"
            return $?
        fi
        if [[ -n ${compute_apps//[[:space:]]/} ]]; then
            busy_observations+=("poll=$poll pids=${compute_apps//$'\n'/,}")
            printf '[%s] GPU busy verification arm=%s poll=%s/%s pids=%s\n' \
                "$(_gpu_lock_timestamp)" "$arm_name" "$poll" "$GPU_LOCK_FREE_POLLS" \
                "${compute_apps//$'\n'/,}" >&2
        else
            printf '[%s] GPU free verification arm=%s poll=%s/%s\n' \
                "$(_gpu_lock_timestamp)" "$arm_name" "$poll" "$GPU_LOCK_FREE_POLLS"
        fi
        if (( poll < GPU_LOCK_FREE_POLLS )); then
            sleep "$GPU_LOCK_FREE_POLL_SECONDS"
        fi
    done
    if (( ${#busy_observations[@]} > 0 )); then
        _gpu_lock_fail "lock acquired but GPU compute processes were observed: ${busy_observations[*]}"
        return $?
    fi
    printf '[%s] GPU lock ready arm=%s; no compute applications detected\n' \
        "$(_gpu_lock_timestamp)" "$arm_name"
}
