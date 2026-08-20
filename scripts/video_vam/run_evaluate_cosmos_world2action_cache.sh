#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/video_vam/cosmos_cuda_env.sh
source "${_script_dir}/cosmos_cuda_env.sh"

exec "${VAM_VENV}/bin/python" "${_script_dir}/evaluate_cosmos_world2action_cache.py" "$@"
