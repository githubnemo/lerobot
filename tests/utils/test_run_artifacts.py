"""Storage tests use temporary directories and tiny tensors only."""

import json
import shutil

import pytest
import torch
from safetensors.torch import load_file

from lerobot.common.run_artifacts import (
    atomic_write,
    atomic_write_json,
    default_run_dir,
    save_weights_artifact,
    sha256_file,
)


def test_atomic_failure_preserves_previous_and_cleans_temporary(tmp_path):
    target = tmp_path / "weights"
    target.write_text("old")

    def fail(path):
        path.write_text("partial")
        raise RuntimeError("serialization failed")

    with pytest.raises(RuntimeError):
        atomic_write(target, fail)
    assert target.read_text() == "old"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_replace_does_not_follow_target_symlink(tmp_path):
    external = tmp_path / "external"
    external.write_text("user data")
    target = tmp_path / "artifact"
    target.symlink_to(external)
    atomic_write_json(target, {"ok": True})
    assert external.read_text() == "user data"
    assert not target.is_symlink()


def test_best_last_are_bounded_and_portable(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "legacy.safetensors").write_text("untouched")
    manifest = {"checkpoints": {"best": None, "last": None}}
    save_weights_artifact(run, "best", {"w": torch.tensor([1.0])}, 1, manifest, metric=1.0)
    for step in range(2, 5):
        save_weights_artifact(run, "last", {"w": torch.tensor([float(step)])}, step, manifest)
    moved = tmp_path / "moved"
    shutil.move(run, moved)
    data = json.loads((moved / "run_manifest.json").read_text())
    for role in ("best", "last"):
        artifact = data["checkpoints"][role]
        assert sha256_file(moved / artifact["path"]) == artifact["sha256"]
        assert artifact["resumable"] is False
    assert load_file(str(moved / "best.safetensors"))["w"].item() == 1
    assert load_file(str(moved / "last.safetensors"))["w"].item() == 4
    assert (moved / "legacy.safetensors").read_text() == "untouched"
    assert len(list(moved.glob("*.safetensors"))) == 3


def test_json_rejects_nan_without_replacing(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"value": 1})
    with pytest.raises(ValueError):
        atomic_write_json(path, {"value": float("nan")})
    assert json.loads(path.read_text()) == {"value": 1}


def test_default_directory_unique_and_repo_relative(tmp_path):
    first = default_run_dir(tmp_path, "smolexpert")
    second = default_run_dir(tmp_path, "smolexpert")
    assert first != second
    assert first.parent == tmp_path / "outputs/train"
    assert not first.exists()


def test_wrapper_setup_only_is_non_destructive(tmp_path):
    import os
    import subprocess
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts/video_vam/run_train_smolvla_scale100_1hr.sh"
    # Execute only directory setup, never logging/GPU locks/the trainer.
    setup = script.read_text().split("exec >", 1)[0]
    run = tmp_path / "fresh run"
    env = dict(os.environ, RUN_DIR=str(run), OUTPUT_DIR=str(run / "training"))
    first = subprocess.run(["bash", "-c", setup], env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert run.is_dir()
    assert not (run / "training").exists()
    sentinel = run / "user-checkpoint"
    sentinel.write_text("preserve")
    second = subprocess.run(["bash", "-c", setup], env=env, capture_output=True, text=True)
    assert second.returncode != 0
    assert "Refusing to overwrite" in second.stderr
    assert sentinel.read_text() == "preserve"
    assert "--save_freq=0" in script.read_text()
    assert "rm -rf" not in script.read_text()
