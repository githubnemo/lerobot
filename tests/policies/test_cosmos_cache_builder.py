import json
from pathlib import Path

import pytest

from scripts.video_vam.build_cosmos_feature_cache import (
    _lora_checkpoint_provenance,
    _validate_args,
    parse_args,
)


def test_builder_accepts_episode_zero_and_bounded_subset():
    args = parse_args(["--episodes", "0", "--max-samples", "4", "--frame-start", "4", "--frame-end", "8"])
    _validate_args(args)
    assert args.episodes == [0]
    assert args.max_samples == 4
    assert args.vae_input_mode == "observed_prefix"
    assert args.state_t == 16


def test_builder_accepts_observed_only_state_t2():
    args = parse_args(["--episodes", "0", "--state-t", "2"])
    _validate_args(args)
    assert args.state_t == 2


def test_builder_accepts_explicit_legacy_vae_mode():
    args = parse_args(["--episodes", "0", "--vae-input-mode", "legacy_padded_vae"])
    _validate_args(args)
    assert args.vae_input_mode == "legacy_padded_vae"


def test_builder_rejects_ambiguous_resume_overwrite_and_ranges():
    args = parse_args(["--episodes", "0", "--resume", "--overwrite"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_args(args)
    args = parse_args(["--episodes", "0", "--frame-start", "8", "--frame-end", "8"])
    with pytest.raises(ValueError, match="greater"):
        _validate_args(args)


def test_builder_reads_strict_lora_provenance(tmp_path: Path):
    weights = tmp_path / "best_lora.safetensors"
    weights.write_bytes(b"adapter tensors")
    weights.with_suffix(".json").write_text(json.dumps({"lora": {"rank": 16, "alpha": 16.0}}) + "\n")

    provenance = _lora_checkpoint_provenance(weights)

    assert provenance["path"] == str(weights.resolve())
    assert len(provenance["sha256"]) == 64
    assert provenance["rank"] == 16
    assert provenance["alpha"] == 16.0
    assert provenance["adapters_applied"] is True
