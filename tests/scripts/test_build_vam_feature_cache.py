"""Tests for unified Protocol 1.0 feature cache builder (build_vam_feature_cache.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lerobot.policies.vam.base import ProtocolViolationError
from scripts.video_vam.build_vam_feature_cache import (
    main as build_vam_feature_cache_main,
    parse_episode_range,
)


def test_parse_episode_range():
    assert parse_episode_range("0-3") == (0, 1, 2, 3)
    assert parse_episode_range("0,2,4") == (0, 2, 4)
    assert parse_episode_range("0-2, 5, 8-9") == (0, 1, 2, 5, 8, 9)


def test_build_vam_feature_cache_blocks_leakage(tmp_path: Path):
    out_dir = tmp_path / "leak_cache"
    with pytest.raises(ProtocolViolationError, match="DATA LEAKAGE DETECTED"):
        build_vam_feature_cache_main(
            [
                "--backbone",
                "cosmos7b",
                "--output-dir",
                str(out_dir),
                "--train-episodes",
                "0-5",
                "--val-episodes",
                "5-8",
                "--dry-run",
            ]
        )


def test_build_vam_feature_cache_blocks_regime_cross(tmp_path: Path):
    out_dir = tmp_path / "cross_cache"
    with pytest.raises(ProtocolViolationError, match="Training set contains held-out validation episodes"):
        build_vam_feature_cache_main(
            [
                "--backbone",
                "cosmos7b",
                "--output-dir",
                str(out_dir),
                "--train-episodes",
                "0-32",
                "--val-episodes",
                "33-39",
                "--dry-run",
            ]
        )


def test_build_vam_feature_cache_dry_run_all_backbones(tmp_path: Path):
    for backbone in ["cosmos7b", "cosmos14b", "flux2_klein"]:
        out_dir = tmp_path / f"{backbone}_cache"
        ret = build_vam_feature_cache_main(
            [
                "--backbone",
                backbone,
                "--output-dir",
                str(out_dir),
                "--train-episodes",
                "0-1",
                "--val-episodes",
                "32-33",
                "--max-train-samples",
                "2",
                "--max-val-samples",
                "2",
                "--device",
                "cpu",
                "--dry-run",
                "--overwrite",
            ]
        )
        assert ret == 0
        train_manifest = out_dir / "train" / "manifest.json"
        val_manifest = out_dir / "val" / "manifest.json"
        assert train_manifest.is_file()
        assert val_manifest.is_file()

        with open(train_manifest) as f:
            train_data = json.load(f)
            assert len(train_data["entries"]) == 2
            assert train_data["entries"][0]["episode_index"] == 0
            assert "safetensors" in train_data["entries"][0]
            assert "safetensors_sha256" in train_data["entries"][0]

        with open(val_manifest) as f:
            val_data = json.load(f)
            assert len(val_data["entries"]) == 2
            assert val_data["entries"][0]["episode_index"] == 32


def test_build_vam_feature_cache_dual_eval_split(tmp_path: Path):
    out_dir = tmp_path / "dual_cache"
    ret = build_vam_feature_cache_main(
        [
            "--backbone",
            "cosmos14b",
            "--output-dir",
            str(out_dir),
            "--train-episodes",
            "0-1, 40-41",
            "--val-episodes",
            "32-33",
            "--eval2-episodes",
            "90-91",
            "--protocol",
            "scale100",
            "--max-train-samples",
            "2",
            "--max-val-samples",
            "2",
            "--device",
            "cpu",
            "--dry-run",
            "--overwrite",
        ]
    )
    assert ret == 0
    assert (out_dir / "train" / "manifest.json").is_file()
    assert (out_dir / "val" / "manifest.json").is_file()
    assert (out_dir / "eval1" / "manifest.json").is_file()
    assert (out_dir / "eval2" / "manifest.json").is_file()

    with open(out_dir / "eval2" / "manifest.json") as f:
        eval2_data = json.load(f)
        assert len(eval2_data["entries"]) == 2
        assert eval2_data["entries"][0]["episode_index"] == 90


def test_build_vam_feature_cache_with_lora_dry_run(tmp_path: Path):
    """Verify that build_vam_feature_cache accepts and applies --lora-weights in dry-run mode."""
    from lerobot.policies.vam.flux2_klein_extractor import build_flux2_klein_dummy_model
    from lerobot.policies.vam.flux2_klein_lora import (
        Flux2KleinLoRAConfig,
        inject_flux2_klein_lora,
        save_flux2_klein_lora,
    )
    from scripts.video_vam.build_vam_feature_cache import main as cache_main

    # Create synthetic LoRA weights for flux2_klein dummy
    dummy = build_flux2_klein_dummy_model(
        num_layers=2,
        num_single_layers=2,
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=16,
        joint_attention_dim=32,
        axes_dims_rope=(4, 4, 4, 4),
        device="cpu",
        dtype=torch.float32,
    )
    lora_cfg = Flux2KleinLoRAConfig(
        rank=16, alpha=16.0, target_double_blocks=(0, 1), target_single_blocks=(0, 1)
    )
    inject_flux2_klein_lora(dummy, lora_cfg)
    lora_file = tmp_path / "test_flux2_lora.safetensors"
    save_flux2_klein_lora(dummy, lora_file)

    out_dir = tmp_path / "lora_cache"
    ret = cache_main(
        [
            "--backbone",
            "flux2_klein",
            "--output-dir",
            str(out_dir),
            "--train-episodes",
            "0-1",
            "--val-episodes",
            "32-33",
            "--lora-weights",
            str(lora_file),
            "--device",
            "cpu",
            "--dry-run",
            "--overwrite",
        ]
    )
    assert ret == 0
    assert (out_dir / "train" / "manifest.json").is_file()
    assert (out_dir / "val" / "manifest.json").is_file()
