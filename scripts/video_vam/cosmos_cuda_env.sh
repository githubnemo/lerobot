#!/usr/bin/env bash
# Source this file before running VAM commands that construct Cosmos/TE modules.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    printf 'ERROR: source %s; do not execute it directly.\n' "${BASH_SOURCE[0]}" >&2
    exit 2
fi

_vam_cuda_env_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || {
    printf 'ERROR: cannot resolve the VAM script directory.\n' >&2
    return 1
}
VAM_REPO_ROOT="${VAM_REPO_ROOT:-$(cd -- "${_vam_cuda_env_script_dir}/../.." && pwd -P)}"
# Python CLIs import both the repository's scripts package and src layout.
export PYTHONPATH="${VAM_REPO_ROOT}:${VAM_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
VAM_VENV="${VAM_VENV:-${VAM_REPO_ROOT}/.venv}"
VAM_NVIDIA_BASE="${VAM_NVIDIA_BASE:-${VAM_VENV}/lib/python3.12/site-packages/nvidia}"

if [[ ! -x "${VAM_VENV}/bin/python" ]]; then
    printf 'ERROR: VAM virtualenv Python is missing: %s\n' "${VAM_VENV}/bin/python" >&2
    return 1
fi
if [[ ! -d "${VAM_NVIDIA_BASE}" ]]; then
    printf 'ERROR: NVIDIA wheel library root is missing: %s\n' "${VAM_NVIDIA_BASE}" >&2
    return 1
fi

_vam_cuda_library_relpaths=(
    cuda_runtime/lib
    cublas/lib
    cudnn/lib
    cusparse/lib
    cusolver/lib
    cufft/lib
    curand/lib
    cuda_nvrtc/lib
    nvjitlink/lib
    nvtx/lib
    nccl/lib
)
_vam_cuda_library_paths=()
for _vam_relpath in "${_vam_cuda_library_relpaths[@]}"; do
    _vam_library_path="${VAM_NVIDIA_BASE}/${_vam_relpath}"
    if [[ ! -d "${_vam_library_path}" ]]; then
        printf 'ERROR: required NVIDIA wheel library directory is missing: %s\n' "${_vam_library_path}" >&2
        return 1
    fi
    _vam_cuda_library_paths+=("${_vam_library_path}")
done

_vam_ld_library_path="$(IFS=:; printf '%s' "${_vam_cuda_library_paths[*]}")"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${_vam_ld_library_path}:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${_vam_ld_library_path}"
fi

for _vam_home in NVRTC_HOME CURAND_HOME CUDNN_HOME; do
    case "${_vam_home}" in
        NVRTC_HOME) _vam_component=cuda_nvrtc ;;
        CURAND_HOME) _vam_component=curand ;;
        CUDNN_HOME) _vam_component=cudnn ;;
    esac
    if [[ ! -d "${VAM_NVIDIA_BASE}/${_vam_component}" ]]; then
        printf 'ERROR: required NVIDIA wheel component is missing: %s\n' "${VAM_NVIDIA_BASE}/${_vam_component}" >&2
        return 1
    fi
    export "${_vam_home}=${VAM_NVIDIA_BASE}/${_vam_component}"
done
export NVTE_CUDA_INCLUDE_DIR="${VAM_NVIDIA_BASE}/cuda_runtime/include"
if [[ ! -d "${NVTE_CUDA_INCLUDE_DIR}" ]]; then
    printf 'ERROR: Transformer Engine CUDA include directory is missing: %s\n' "${NVTE_CUDA_INCLUDE_DIR}" >&2
    return 1
fi

unset _vam_cuda_env_script_dir _vam_cuda_library_relpaths _vam_cuda_library_paths _vam_relpath _vam_library_path _vam_ld_library_path _vam_home _vam_component
