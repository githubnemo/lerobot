#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/video_vam/cosmos_cuda_env.sh
source "${_script_dir}/cosmos_cuda_env.sh"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required before GPU work" >&2
  exit 2
fi
_used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR == 1 {print $1}')"
if [[ "${_used_mib}" -gt 2048 ]]; then
  echo "ERROR: GPU is busy (${_used_mib} MiB already used); wait before LTX extraction" >&2
  exit 2
fi

exec "${VAM_VENV}/bin/python" "${_script_dir}/smoke_test_ltx_extractor.py" "$@"
