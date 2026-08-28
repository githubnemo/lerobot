import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional

from lerobot.policies.vam.context_transform import (
    CONTEXT_GRID_FLATTEN_ORDER,
    CONTEXT_TRANSFORMS,
    apply_context_transform,
    context_transform_metadata,
)
from lerobot.policies.vam.cosmos_cache_dataset import load_cache_manifest, write_cache_manifest
from scripts.video_vam.build_cosmos_feature_cache import (
    apply_context_transform as builder_context_transform,
)
from scripts.video_vam.train_cosmos_world2action import (
    apply_context_transform as trainer_context_transform,
    parse_args as train_parse_args,
    resolve_context_transform,
    should_save_checkpoint,
    validate_args as train_validate_args,
)


def _manifest(tmp_path, provenance):
    tensor = tmp_path / "episode-0000-frame-000004.safetensors"
    sidecar = tensor.with_suffix(".json")
    tensor.write_bytes(b"tensor")
    sidecar.write_text("{}")
    entry = {
        "sample_id": "episode-0000-frame-000004",
        "episode_index": 0,
        "frame_index": 4,
        "window_indices": [0, 1, 2, 3, 4],
        "noise_seed": 7,
        "safetensors": tensor.name,
        "sidecar": sidecar.name,
        "safetensors_sha256": hashlib.sha256(tensor.read_bytes()).hexdigest(),
        "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "bytes": tensor.stat().st_size + sidecar.stat().st_size,
    }
    payload = {
        "schema_version": 1,
        "cache_schema_version": 2,
        "dataset": {"repo_id": "dataset", "revision": "revision"},
        "subset": {
            "episodes": [0],
            "frame_start": None,
            "frame_end": None,
            "max_samples": 1,
            "stride": 1,
        },
        "provenance": provenance,
        "global_seed": 0,
        "entries": [entry],
        "total_bytes": entry["bytes"],
        "runtime": {},
    }
    path = tmp_path / "manifest.json"
    write_cache_manifest(payload, path)
    return path


def test_cache_and_training_use_identical_pool4_transform():
    raw = torch.arange(16 * 30 * 40 * 2, dtype=torch.float32).reshape(1, 19200, 2).to(torch.bfloat16)
    expected_grid = raw.view(1, 16, 30, 40, 2)
    expected_flat = expected_grid.permute(0, 1, 4, 2, 3).reshape(16, 2, 30, 40)
    expected = (
        functional.adaptive_avg_pool2d(expected_flat, (8, 10))
        .reshape(1, 16, 2, 8, 10)
        .permute(0, 1, 3, 4, 2)
        .reshape(1, 1280, 2)
    )
    assert builder_context_transform is trainer_context_transform is apply_context_transform
    assert torch.equal(builder_context_transform(raw, "pool4"), expected)
    assert torch.equal(trainer_context_transform(raw, "pool4"), expected)


def test_observed_only_pool2_reduces_2400_tokens_to_600():
    raw = torch.arange(2 * 30 * 40 * 2, dtype=torch.float32).reshape(1, 2400, 2).to(torch.bfloat16)
    expected_grid = raw.view(1, 2, 30, 40, 2)
    expected_flat = expected_grid.permute(0, 1, 4, 2, 3).reshape(2, 2, 30, 40)
    expected = (
        functional.adaptive_avg_pool2d(expected_flat, (15, 20))
        .reshape(1, 2, 2, 15, 20)
        .permute(0, 1, 3, 4, 2)
        .reshape(1, 600, 2)
    )

    reduced = apply_context_transform(raw, "pool2")

    assert reduced.shape == (1, 600, 2)
    assert torch.equal(reduced, expected)
    metadata = context_transform_metadata("pool2", temporal_frames=2)
    assert metadata["input_grid"] == {
        "temporal": 2,
        "height": 30,
        "width": 40,
        "flatten_order": CONTEXT_GRID_FLATTEN_ORDER,
    }
    assert metadata["output_grid"] == {
        "temporal": 2,
        "height": 15,
        "width": 20,
        "flatten_order": CONTEXT_GRID_FLATTEN_ORDER,
    }
    assert metadata["output_tokens"] == 600


def test_observed_only_context_has_no_future_frame_transform():
    raw = torch.zeros(1, 2400, 2)
    assert apply_context_transform(raw, "cond_frames").shape == (1, 2400, 2)
    with pytest.raises(ValueError, match="future frames"):
        apply_context_transform(raw, "gen_frames_pool2")


def test_context_metadata_declares_pool4_grid_and_tokens():
    metadata = context_transform_metadata("pool4")
    assert metadata["output_tokens"] == 1280
    assert metadata["output_grid"] == {
        "temporal": 16,
        "height": 8,
        "width": 10,
        "flatten_order": CONTEXT_GRID_FLATTEN_ORDER,
    }
    assert "pool4" in CONTEXT_TRANSFORMS


def test_manifest_accepts_declared_transform_and_rejects_inconsistent_tokens(tmp_path):
    provenance = {
        "builder": "test",
        "context_transform": "pool4",
        "context_tokens": 1280,
        "context_grid": {
            "temporal": 16,
            "height": 8,
            "width": 10,
            "flatten_order": "T,H,W",
        },
        # A decoy nested field guards the manifest-reader path used by queue checks.
        "output": {"context_transform": "none"},
    }
    path = _manifest(tmp_path, provenance)
    assert load_cache_manifest(path).context_transform == "pool4"
    assert load_cache_manifest(path).context_tokens == 1280
    payload = json.loads(path.read_text())
    payload["provenance"]["context_tokens"] = 4800
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="inconsistent"):
        load_cache_manifest(path)


def test_manifest_accepts_observed_only_pool2_grid(tmp_path):
    provenance = {
        "builder": "test",
        "context_transform": "pool2",
        "context_tokens": 600,
        "context_grid": {
            "temporal": 2,
            "height": 15,
            "width": 20,
            "flatten_order": "T,H,W",
        },
    }
    manifest = load_cache_manifest(_manifest(tmp_path, provenance))
    assert manifest.context_transform == "pool2"
    assert manifest.context_tokens == 600


def test_manifest_without_transform_remains_legacy_none(tmp_path):
    manifest = load_cache_manifest(_manifest(tmp_path, {"builder": "legacy"}))
    assert manifest.context_transform == "none"
    assert manifest.context_tokens is None


def test_trainer_rejects_double_transform_for_pre_reduced_cache():
    manifest = SimpleNamespace(context_transform="pool4")
    with pytest.raises(ValueError, match="twice"):
        resolve_context_transform("pool4", manifest)
    assert resolve_context_transform("none", manifest) == "none"


def test_metrics_only_mode_rejects_resume_and_skips_checkpoint_decisions():
    args = train_parse_args(
        [
            "--manifest",
            "manifest.json",
            "--split",
            "split.json",
            "--output-dir",
            "run",
            "--no-save-checkpoints",
        ]
    )
    train_validate_args(args)
    assert not should_save_checkpoint(args, step=1, should_validate=True, stop=False)
    with pytest.raises(ValueError, match="no-save-checkpoints"):
        train_validate_args(
            train_parse_args(
                [
                    "--manifest",
                    "manifest.json",
                    "--split",
                    "split.json",
                    "--output-dir",
                    "run",
                    "--no-save-checkpoints",
                    "--resume",
                ]
            )
        )
