#!/usr/bin/env bash
# Generic launcher for Video-VAM experiments. Replaces the ~80 one-off run_*.sh / *_queue.sh wrappers.
#
#   bash scripts/video_vam/run_experiment.sh <preset> [extra CLI args...]
#   bash scripts/video_vam/run_experiment.sh --list
#   DRY_RUN=1 bash scripts/video_vam/run_experiment.sh <preset>     # print the command only
#
# A preset is scripts/video_vam/presets/<preset>.args: one CLI token group per line, '#' comments,
# blank lines ignored. Optional directives (first lines):
#   #! trainer: smolexpert     (default) -> scripts/video_vam/train_smolexpert.py
#   #! trainer: lerobot-train  -> native LeRobot trainer (e.g. SmolVLA baselines)
#   #! trainer: <script.py>    -> any scripts/video_vam/<script.py>
#   #! include: <preset>       -> prepend another preset's args (e.g. shared _base files)
# Values may reference ${RUN_TAG} (defaults to the preset name) and environment variables.
# Extra CLI args are appended, so later flags override preset flags for argparse/draccus.
#
# Environment:
#   RUN_TAG         output/log tag (default: preset name)
#   VAM_CACHE_DIR   model cache directory (default: ${HOME}/.cache/video-vam)
#   HF_AUTO_UPLOAD  set to 1 to upload outputs/train/${RUN_TAG} to HF Hub on exit 0
#   HF_REPO_ID      optional explicit HF repo ID when HF_AUTO_UPLOAD=1
#   HF_UPLOAD_OWNER HF owner/org for auto-upload (default: Orellius)
#   GPU_LOCK=0      skip scripts/video_vam/gpu_lock.sh (default: acquire when flock+nvidia-smi exist)
#   DRY_RUN=1       print the resolved command and exit
set -Eeuo pipefail

export VAM_CACHE_DIR="${VAM_CACHE_DIR:-${HOME:-/tmp}/.cache/video-vam}"
if [[ -n ${HOME:-} && -f "${HOME}/.env.ml" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/.env.ml"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
PRESET_DIR="${SCRIPT_DIR}/presets"

if [[ ${1:-} == "--list" || $# -lt 1 ]]; then
    printf 'Available presets (%s):\n' "$PRESET_DIR"
    for f in "$PRESET_DIR"/*.args; do
        name=$(basename "$f" .args)
        [[ $name == _* ]] && continue
        desc=$(grep -m1 '^# ' "$f" | sed 's/^# //')
        printf '  %-36s %s\n' "$name" "$desc"
    done
    [[ $# -lt 1 ]] && exit 64 || exit 0
fi

PRESET=$1
shift
PRESET_FILE="${PRESET_DIR}/${PRESET}.args"
[[ -f $PRESET_FILE ]] || { printf 'Unknown preset: %s (see --list)\n' "$PRESET" >&2; exit 64; }

export RUN_TAG="${RUN_TAG:-$PRESET}"
cd "$REPO_ROOT"

TRAINER=$(sed -n 's/^#! *trainer: *//p' "$PRESET_FILE" | head -n1)
TRAINER=${TRAINER:-smolexpert}

# Read preset tokens (recursively resolving '#! include:'). ${VAR} references are expanded via bash
# word parsing, so presets must not contain command substitutions.
ARGS=()
read_preset() {
    local file=$1 depth=${2:-0} line inc
    (( depth < 5 )) || { printf 'Preset include depth exceeded at %s\n' "$file" >&2; exit 65; }
    while IFS= read -r inc; do
        [[ -f "${PRESET_DIR}/${inc}.args" ]] || { printf 'Missing include %s in %s\n' "$inc" "$file" >&2; exit 65; }
        read_preset "${PRESET_DIR}/${inc}.args" $((depth + 1))
    done < <(sed -n 's/^#! *include: *//p' "$file")
    while IFS= read -r line || [[ -n $line ]]; do
        line="${line%%#*}"
        [[ -z ${line//[[:space:]]/} ]] && continue
        if [[ $line == *'$('* || $line == *'`'* ]]; then
            printf 'Refusing command substitution in preset line: %s\n' "$line" >&2
            exit 65
        fi
        eval "tokens=( $line )"
        ARGS+=("${tokens[@]}")
    done < "$file"
}
read_preset "$PRESET_FILE"
ARGS+=("$@")

PYTHON="${VAM_VENV:-${REPO_ROOT}/.venv}/bin/python"
case $TRAINER in
    smolexpert) CMD=("$PYTHON" scripts/video_vam/train_smolexpert.py "${ARGS[@]}") ;;
    lerobot-train) CMD=("${VAM_VENV:-${REPO_ROOT}/.venv}/bin/lerobot-train" "${ARGS[@]}") ;;
    *.py) CMD=("$PYTHON" "scripts/video_vam/${TRAINER}" "${ARGS[@]}") ;;
    *) printf 'Unknown trainer directive: %s\n' "$TRAINER" >&2; exit 65 ;;
esac

if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

if [[ -f scripts/video_vam/cosmos_cuda_env.sh && -d "${VAM_VENV:-${REPO_ROOT}/.venv}/lib/python3.12/site-packages/nvidia" ]]; then
    # shellcheck source=/dev/null
    source scripts/video_vam/cosmos_cuda_env.sh
else
    export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONUNBUFFERED=1

mkdir -p outputs/logs outputs/train
LOG_FILE="outputs/logs/${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ ${GPU_LOCK:-1} == 1 ]] && command -v flock >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1; then
    # shellcheck source=/dev/null
    source scripts/video_vam/gpu_lock.sh
    acquire_gpu_lock "$RUN_TAG"
fi

printf '=%.0s' {1..80}; printf '\n'
printf 'preset:  %s\nrun tag: %s\ncommit:  %s%s\nlog:     %s\nstart:   %s\n' \
    "$PRESET" "$RUN_TAG" "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
    "$(git diff --quiet 2>/dev/null || echo ' (dirty)')" "$LOG_FILE" "$(date)"
printf 'command: '; printf '%q ' "${CMD[@]}"; printf '\n'
printf '=%.0s' {1..80}; printf '\n'

set +e
"${CMD[@]}"
status=$?
set -e
printf 'finished %s with status %s at %s\n' "$PRESET" "$status" "$(date)"

if [[ $status -eq 0 && ${HF_AUTO_UPLOAD:-0} == 1 ]]; then
    UPLOAD_ARGS=(--run-dir "outputs/train/${RUN_TAG}")
    if [[ -n ${HF_REPO_ID:-} ]]; then
        UPLOAD_ARGS+=(--repo-id "$HF_REPO_ID")
    fi
    printf 'uploading outputs/train/%s to Hugging Face Hub...\n' "$RUN_TAG"
    "$PYTHON" scripts/video_vam/push_checkpoint_to_hf.py "${UPLOAD_ARGS[@]}"
fi
exit $status
