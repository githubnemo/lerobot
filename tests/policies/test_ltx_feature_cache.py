from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lerobot.policies.vam.ltx_action import pool2_ltx_context
from lerobot.policies.vam.ltx_feature_cache import (
    LTXFeatureCacheError,
    LTXFeatureCacheItem,
    load_ltx_cache_manifest,
    save_ltx_feature_cache,
    sha256_file,
    verify_ltx_feature_cache,
    write_ltx_cache_manifest,
)


def provenance(context_transform: str = "pool2") -> dict:
    context_shape = [1, 640, 4096] if context_transform == "pool2" else [1, 2400, 4096]
    return {
        "schema_version": 1,
        "artifact": "ltx25_frozen_feature_cache",
        "dataset": {
            "repo_id": "hubnemo/cube_out_of_box_dataset",
            "revision": "r" * 40,
            "episode_index": 32,
            "frame_index": 100,
            "window_indices": [96, 97, 98, 99, 100],
        },
        "split": {"name": "validation"},
        "prompt": {"embedding_sha256": "a" * 64, "fallback_allowed": False},
        "backbone": {"checkpoint_sha256": "b" * 64, "video_vae_sha256": "c" * 64},
        "extraction": {"noise_seed": 7, "sigma": 1.0, "hidden_layer": 34},
        "temporal_contract": {
            "observed_rgb_frames": 5,
            "vae_input_frames": 9,
            "clean_latent_frames": 2,
            "sigma1_noise_latent_frames": 6,
            "future_pixels_used": False,
        },
        "output": {
            "context_transform": context_transform,
            "context_shape": context_shape,
            "context_dtype": "bfloat16",
        },
        "tensors": {"action_padding_semantics": "true means padded"},
    }


def cache_item(context_transform: str = "pool2") -> LTXFeatureCacheItem:
    tokens = 640 if context_transform == "pool2" else 2400
    return LTXFeatureCacheItem(
        context=torch.zeros(1, tokens, 4096, dtype=torch.bfloat16),
        state=torch.arange(6, dtype=torch.float32).view(1, 1, 6),
        target_action=torch.zeros(1, 30, 6, dtype=torch.float32),
        action_is_pad=torch.zeros(1, 30, dtype=torch.bool),
        provenance=provenance(context_transform),
    )


def test_pool2_ltx_context_shape_order_and_constant_preservation() -> None:
    grid = torch.arange(8, dtype=torch.float32).view(1, 8, 1, 1, 1).expand(1, 8, 15, 20, 4096)
    pooled = pool2_ltx_context(grid)
    assert pooled.shape == (1, 640, 4096)
    assert torch.equal(pooled[:, :80], torch.zeros_like(pooled[:, :80]))
    assert torch.equal(pooled[:, -80:], torch.full_like(pooled[:, -80:], 7))


def test_ltx_cache_round_trip_and_manifest(tmp_path: Path) -> None:
    tensor_path = tmp_path / "episode-0032-frame-000100.safetensors"
    save_ltx_feature_cache(cache_item(), tensor_path)
    loaded = verify_ltx_feature_cache(tensor_path)
    assert loaded.context.shape == (1, 640, 4096)
    sidecar = tensor_path.with_suffix(".json")
    entry = {
        "sample_id": "episode-0032-frame-000100",
        "episode_index": 32,
        "frame_index": 100,
        "window_indices": [96, 97, 98, 99, 100],
        "noise_seed": 7,
        "safetensors": tensor_path.name,
        "sidecar": sidecar.name,
        "safetensors_sha256": sha256_file(tensor_path),
        "sidecar_sha256": sha256_file(sidecar),
        "bytes": tensor_path.stat().st_size + sidecar.stat().st_size,
    }
    manifest = {
        "schema_version": 1,
        "cache_schema_version": 1,
        "artifact": "ltx25_frozen_feature_manifest",
        "dataset": {"repo_id": "hubnemo/cube_out_of_box_dataset", "revision": "r" * 40},
        "subset": {"split": "validation", "episodes": [32], "stride": 20},
        "provenance": {
            "backbone": "LTX-2.5-22B-distilled",
            "hidden_layer": 34,
            "high_noise_sigma": 1.0,
            "context_transform": "pool2",
            "context_tokens": 640,
            "context_channels": 4096,
            "context_grid": [8, 8, 10],
            "context_dtype": "bfloat16",
        },
        "global_seed": 0,
        "entries": [entry],
        "total_bytes": entry["bytes"],
        "runtime": {},
    }
    manifest_path = tmp_path / "manifest.json"
    write_ltx_cache_manifest(manifest, manifest_path)
    parsed = load_ltx_cache_manifest(manifest_path)
    assert parsed.entries[0].sample_id == entry["sample_id"]


def test_unpooled_ltx_cache_round_trip(tmp_path: Path) -> None:
    tensor_path = tmp_path / "episode-0032-frame-000100.safetensors"
    save_ltx_feature_cache(cache_item("none"), tensor_path)
    loaded = verify_ltx_feature_cache(tensor_path)
    assert loaded.context.shape == (1, 2400, 4096)
    assert loaded.provenance["output"]["context_transform"] == "none"


def test_ltx_cache_rejects_tampering(tmp_path: Path) -> None:
    tensor_path = tmp_path / "episode-0032-frame-000100.safetensors"
    save_ltx_feature_cache(cache_item(), tensor_path)
    payload = json.loads(tensor_path.with_suffix(".json").read_text())
    payload["output"]["context_transform"] = "frame_mean"
    tensor_path.with_suffix(".json").write_text(json.dumps(payload))
    with pytest.raises(LTXFeatureCacheError, match="unsupported LTX context transform"):
        verify_ltx_feature_cache(tensor_path)
