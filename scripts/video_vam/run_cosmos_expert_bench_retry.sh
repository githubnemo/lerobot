#!/usr/bin/env bash
# Retry only the SmolExpert benchmark arms after the current latency queue exits.
set -uo pipefail

REPO_ROOT=/home/anton/lerobot-video-vam
GATE_DIR=/home/anton/.cache/video-vam/runs/cosmos-latency-bench
TERMINAL_MARKER="$GATE_DIR/TERMINAL"
RETRY_DIR="$GATE_DIR/expert-retry"
OUTPUT_ROOT=/home/anton/.cache/video-vam/cosmos-extraction-benchmark-expert-retry
POLL_SECONDS="${COSMOS_EXPERT_RETRY_POLL_SECONDS:-15}"
GPU_MEMORY_LIMIT_MIB=2000
GPU_UTILIZATION_LIMIT=5
LOG="$RETRY_DIR/queue.log"

mkdir -p "$RETRY_DIR" "$OUTPUT_ROOT"
exec > >(tee -a "$LOG") 2>&1
trap 'touch "$RETRY_DIR/TERMINAL"' EXIT

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

log "waiting for current benchmark marker: $TERMINAL_MARKER"
while [[ ! -f "$TERMINAL_MARKER" ]]; do
  sleep "$POLL_SECONDS"
done
log "current benchmark terminal marker present"

idle_samples=0
while (( idle_samples < 2 )); do
  if [[ ! -f "$TERMINAL_MARKER" ]]; then
    idle_samples=0
    log "current benchmark marker disappeared; resetting GPU idle samples"
    sleep "$POLL_SECONDS"
    continue
  fi
  if ! read -r memory_mib utilization < <(
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F, 'NR == 1 {gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); print $1, $2}'
  ); then
    idle_samples=0
    log "nvidia-smi failed; waiting"
    sleep "$POLL_SECONDS"
    continue
  fi
  if [[ "$memory_mib" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] &&
    (( memory_mib < GPU_MEMORY_LIMIT_MIB && utilization < GPU_UTILIZATION_LIMIT )); then
    idle_samples=$((idle_samples + 1))
    log "GPU idle sample $idle_samples/2 (${memory_mib} MiB, ${utilization}% utilization)"
  else
    idle_samples=0
    log "GPU busy or unreadable (${memory_mib:-unknown} MiB, ${utilization:-unknown}% utilization); resetting"
  fi
  (( idle_samples < 2 )) && sleep "$POLL_SECONDS"
done

cd "$REPO_ROOT" || exit 2
if ! source scripts/video_vam/cosmos_cuda_env.sh; then
  log "failed to source Cosmos CUDA environment"
  exit 2
fi

COMMON_ARGS=(
  --iterations 20
  --warmups 3
  --cudnn-benchmark
  --decoder-checkpoint /home/anton/.cache/video-vam/runs/smolexpert-cosmos/train-converge/best.safetensors
  --normalizer /home/anton/.cache/video-vam/runs/smolexpert-cosmos/train-converge/normalizer.safetensors
)
overall_rc=0

run_arm() {
  local name="$1"
  local arm="$2"
  local done_marker="$RETRY_DIR/${name}.done"
  local failed_marker="$RETRY_DIR/${name}.failed"
  local rc
  rm -f "$done_marker" "$failed_marker"
  log "starting arm=$arm output=$OUTPUT_ROOT/$name"
  if "$VAM_VENV/bin/python" scripts/video_vam/benchmark_cosmos_extraction.py       --arm "$arm" "${COMMON_ARGS[@]}" --output-dir "$OUTPUT_ROOT/$name"; then
    touch "$done_marker"
    log "arm=$arm completed"
  else
    rc=$?
    touch "$failed_marker"
    overall_rc=1
    log "arm=$arm failed rc=$rc; continuing"
  fi
}

run_arm context_cache context_cache
run_arm decoder_sweep decoder_sweep

if (( overall_rc == 0 )); then
  touch "$RETRY_DIR/SUCCESS"
  log "expert retry arms completed successfully"
else
  rm -f "$RETRY_DIR/SUCCESS"
  log "one or more expert retry arms failed"
fi
exit "$overall_rc"
