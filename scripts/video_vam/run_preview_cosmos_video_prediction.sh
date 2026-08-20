#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
# shellcheck source=scripts/video_vam/cosmos_cuda_env.sh
source "${repo_root}/scripts/video_vam/cosmos_cuda_env.sh"
exec "${VAM_VENV}/bin/python" "${repo_root}/scripts/video_vam/preview_cosmos_video_prediction.py" "$@"
