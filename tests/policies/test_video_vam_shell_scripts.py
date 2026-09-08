import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "scripts/video_vam/cosmos_cuda_env.sh"
LAUNCHER = REPO_ROOT / "scripts/video_vam/run_smoke_test_cosmos_extractor.sh"
CACHE_LAUNCHER = REPO_ROOT / "scripts/video_vam/run_build_cosmos_feature_cache.sh"
TRAIN_LAUNCHER = REPO_ROOT / "scripts/video_vam/run_train_cosmos_world2action_overfit.sh"
RPC_LAUNCHER = REPO_ROOT / "scripts/video_vam/run_rpc_server.sh"
EVAL_LAUNCHER = REPO_ROOT / "scripts/video_vam/run_evaluate_cosmos_world2action_cache.sh"


def test_video_vam_shell_scripts_have_valid_bash_syntax():
    for script in (HELPER, LAUNCHER, CACHE_LAUNCHER, TRAIN_LAUNCHER, EVAL_LAUNCHER, RPC_LAUNCHER):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_cosmos_launcher_sources_environment_and_forwards_arguments():
    helper_text = HELPER.read_text()
    launcher_text = LAUNCHER.read_text()
    assert "cuda_runtime/lib" in helper_text
    assert 'export LD_LIBRARY_PATH="${_vam_ld_library_path}:${LD_LIBRARY_PATH}"' in helper_text
    assert 'export NVTE_CUDA_INCLUDE_DIR="${VAM_NVIDIA_BASE}/cuda_runtime/include"' in helper_text
    assert 'source "${_script_dir}/cosmos_cuda_env.sh"' in launcher_text
    assert 'smoke_test_cosmos_extractor.py" "$@"' in launcher_text


def test_all_vam_launchers_make_repo_imports_available_from_arbitrary_cwd():
    with tempfile.TemporaryDirectory() as cwd:
        for launcher in (LAUNCHER, CACHE_LAUNCHER, TRAIN_LAUNCHER, EVAL_LAUNCHER):
            result = subprocess.run(
                [str(launcher), "--help"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            )
            assert "usage:" in result.stdout.lower()


def test_diagnostic_launchers_source_environment_and_forward_arguments():
    for launcher, script_name in (
        (CACHE_LAUNCHER, "build_cosmos_feature_cache.py"),
        (TRAIN_LAUNCHER, "train_cosmos_world2action_overfit.py"),
        (EVAL_LAUNCHER, "evaluate_cosmos_world2action_cache.py"),
    ):
        launcher_text = launcher.read_text()
        assert 'source "${_script_dir}/cosmos_cuda_env.sh"' in launcher_text
        assert f'{script_name}" "$@"' in launcher_text


def test_rpc_launcher_forwards_args_without_exclusive_gpu_lock():
    text = RPC_LAUNCHER.read_text()
    assert 'scripts/video_vam/rpc_server.py "$@"' in text
    assert "acquire_gpu_lock" not in text
    assert "source scripts/video_vam/gpu_lock.sh" not in text
    assert "flock" not in text
