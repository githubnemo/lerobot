#!/usr/bin/env bash
# Idempotent bootstrap for a RunPod RTX 5090 (Blackwell sm_120) pod backed by a persistent /workspace Network Volume.
#
# Run from Mac via SSH:
#   ssh root@<IP> -p <PORT> 'bash -s' < scripts/video_vam/runpod/bootstrap_pod.sh
#
# Optional env overrides:
#   REPO_URL         git clone URL (default: https://github.com/githubnemo/lerobot.git)
#   BRANCH           git branch (default: feat-video-vam)
#   DOWNLOAD_MODELS  1 to download Cosmos3-Edge, Cosmos3-Nano, FLUX3-Action (default: 1)
#   RECALIBRATE_FLUX3  1 to build sort-cubes recalibrated FLUX 3 Action base (default: 1)
set -Eeuo pipefail

export WORKSPACE="${WORKSPACE:-/workspace}"
export REPO_DIR="${REPO_DIR:-${WORKSPACE}/lerobot}"
export REPO_URL="${REPO_URL:-https://github.com/githubnemo/lerobot.git}"
export BRANCH="${BRANCH:-feat-video-vam}"
export VAM_CACHE_DIR="${VAM_CACHE_DIR:-${WORKSPACE}/.cache/video-vam}"
export HF_HOME="${HF_HOME:-${WORKSPACE}/.cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE}/.cache/uv}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

mkdir -p "${WORKSPACE}" "${VAM_CACHE_DIR}" "${HF_HOME}" "${HF_LEROBOT_HOME}" "${UV_CACHE_DIR}" "${HOME}/.cache"

# Symlink ~/.cache/{video-vam,huggingface} onto the persistent Network Volume
ln -sfn "${VAM_CACHE_DIR}" "${HOME}/.cache/video-vam"
ln -sfn "${HF_HOME}" "${HOME}/.cache/huggingface"

# Persist env vars in ~/.env.ml and ~/.bashrc (chmod 600)
umask 077
cat > "${HOME}/.env.ml" <<EOF
export VAM_CACHE_DIR="${VAM_CACHE_DIR}"
export HF_HOME="${HF_HOME}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME}"
export UV_CACHE_DIR="${UV_CACHE_DIR}"
export HF_AUTO_UPLOAD="${HF_AUTO_UPLOAD:-1}"
export HF_UPLOAD_OWNER="${HF_UPLOAD_OWNER:-Orellius}"
${HF_TOKEN:+export HF_TOKEN="${HF_TOKEN}"}
${WANDB_API_KEY:+export WANDB_API_KEY="${WANDB_API_KEY}"}
EOF
chmod 600 "${HOME}/.env.ml"
grep -qF '.env.ml' "${HOME}/.bashrc" 2>/dev/null || echo '[ -f "$HOME/.env.ml" ] && source "$HOME/.env.ml"' >> "${HOME}/.bashrc"

echo "=== [1/6] System packages (tmux, ffmpeg, git-lfs, uv) ==="
if ! command -v tmux >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq tmux ffmpeg git-lfs curl ca-certificates >/dev/null
fi
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

echo "=== [2/6] Clone / update repo (${REPO_DIR} @ ${BRANCH}) ==="
if [[ ! -d "${REPO_DIR}/.git" ]]; then
    git -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.clean= -c filter.lfs.required=false \
        clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${REPO_DIR}"
else
    cd "${REPO_DIR}"
    git fetch origin "${BRANCH}"
    git checkout "${BRANCH}"
    git -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.clean= -c filter.lfs.required=false \
        pull --ff-only origin "${BRANCH}"
fi
cd "${REPO_DIR}"

echo "=== [3/6] Python venv & Blackwell (CUDA 12.8) PyTorch setup ==="
if [[ ! -d "${REPO_DIR}/.venv" ]]; then
    uv venv --python 3.12 "${REPO_DIR}/.venv"
fi
uv sync --locked --extra peft || uv pip install -e ".[peft]"
uv pip install --quiet hf_transfer wandb huggingface_hub safetensors torchcodec

# Verify CUDA works on the actual GPU (RTX 5090 requires sm_120 / cu128 PyTorch)
if ! "${REPO_DIR}/.venv/bin/python" -c "import torch; assert torch.cuda.is_available(); x = torch.randn(4, 4, device='cuda'); _ = x @ x; print('CUDA OK:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), 'torch', torch.__version__)" ; then
    echo "Installing PyTorch cu128 wheel for Blackwell (sm_120)..."
    uv pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
    "${REPO_DIR}/.venv/bin/python" -c "import torch; assert torch.cuda.is_available(); x = torch.randn(4, 4, device='cuda'); _ = x @ x; print('CUDA OK:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), 'torch', torch.__version__)"
fi

echo "=== [4/6] Hugging Face & W&B auth ==="
if [[ -n "${HF_TOKEN:-}" ]]; then
    "${REPO_DIR}/.venv/bin/python" -c "from huggingface_hub import HfApi; import os; print('HF user:', HfApi(token=os.environ['HF_TOKEN']).whoami()['name'])"
else
    echo "WARNING: HF_TOKEN not set; gated models and auto-upload will fail." >&2
fi

if [[ "${DOWNLOAD_MODELS:-1}" == "1" ]]; then
    echo "=== [5/6] Downloading backbones to ${VAM_CACHE_DIR} ==="
    "${REPO_DIR}/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

cache_dir = Path(os.environ["VAM_CACHE_DIR"])
token = os.environ.get("HF_TOKEN")

targets = [
    ("nvidia/Cosmos3-Edge", cache_dir / "cosmos3-edge"),
    ("nvidia/Cosmos3-Nano", cache_dir / "cosmos3-nano"),
    ("black-forest-labs/flux-3-action-so101", cache_dir / "flux3-action-so101"),
]
for repo_id, dst in targets:
    marker = dst / ".download_complete"
    if marker.is_file():
        print(f"[skip] {repo_id} already complete at {dst}")
        continue
    print(f"[download] {repo_id} -> {dst} ...")
    snapshot_download(repo_id=repo_id, local_dir=str(dst), token=token)
    marker.write_text("ok\n")
    print(f"[done] {repo_id}")
PY
fi

if [[ "${RECALIBRATE_FLUX3:-1}" == "1" ]]; then
    echo "=== [6/6] Recalibrating FLUX 3 Action base for sort-cubes ==="
    RECALIBRATED_DIR="${REPO_DIR}/outputs/bases/flux3-so101-sortcubes-recal"
    if [[ ! -d "${RECALIBRATED_DIR}" ]]; then
        mkdir -p "${REPO_DIR}/outputs/bases"
        "${REPO_DIR}/.venv/bin/python" scripts/video_vam/export_flux3_recalibrated_base.py \
            --policy-path "${VAM_CACHE_DIR}/flux3-action-so101" \
            --dataset-repo-id Orellius/so101_sort_cubes_no_top \
            --dataset-revision a7c156de0e84a81e19457e6d783e4316e9f901e1 \
            --train-episodes 0:68 \
            --output "${RECALIBRATED_DIR}"
    else
        echo "[skip] Recalibrated FLUX 3 base already exists at ${RECALIBRATED_DIR}"
    fi
fi

echo "=== BOOTSTRAP COMPLETE ==="
echo "Repo:  ${REPO_DIR}"
echo "Cache: ${VAM_CACHE_DIR}"
echo "Launch first experiment inside tmux:"
echo "  tmux new -s exp001 'cd ${REPO_DIR} && bash scripts/video_vam/experiments/exp001_c3_edge_joint_lora_bs64.sh'"
