import json
from pathlib import Path

from lerobot.policies.vam.cosmos_cache_dataset import CacheManifest
from scripts.video_vam.train_cosmos_video_lora import (
    _checkpoint_metadata,
    parse_args,
    validate_args,
)


def _manifest(path: Path) -> CacheManifest:
    return CacheManifest(
        path,
        {
            "dataset": {"repo_id": "hubnemo/cube_out_of_box_dataset", "revision": "rev"},
            "subset": {"stride": 3},
            "global_seed": 0,
            "provenance": {},
        },
        (),
    )


def test_video_lora_cli_rejects_cpu_before_model_construction():
    args = parse_args(["--device", "cpu"])
    try:
        validate_args(args)
    except ValueError as exc:
        assert "requires --device cuda" in str(exc)
    else:
        raise AssertionError("the video-only trainer must reject CPU execution")


def test_video_lora_source_has_no_action_decoder_path():
    source = Path("scripts/video_vam/train_cosmos_video_lora.py").read_text()
    assert "World2ActionDecoder" not in source
    assert "SmolExpertActionDecoder" not in source


def test_video_lora_checkpoint_metadata_contains_adapter_contract(tmp_path: Path):
    train_path = tmp_path / "train.json"
    val_path = tmp_path / "val.json"
    split_path = tmp_path / "split.json"
    checkpoint_path = tmp_path / "generic.pt"
    for path in (train_path, val_path, split_path):
        path.write_text("{}\n")
    checkpoint_path.write_bytes(b"generic checkpoint")
    args = parse_args(
        [
            "--backbone-checkpoint",
            str(checkpoint_path),
            "--train-episodes",
            "0",
            "1",
            "--lora-rank",
            "16",
            "--lora-alpha",
            "16",
            "--seed",
            "0",
        ]
    )
    metadata = _checkpoint_metadata(
        args,
        kind="best",
        step=12,
        best_step=12,
        best_metric=0.25,
        adapter_names=("blocks.0.self_attn.q_proj",),
        train_manifest=_manifest(train_path),
        val_manifest=_manifest(val_path),
        split_path=split_path,
        checkpoint_sha256="a" * 64,
        trainable_parameters=123,
        stop_reason="validation_plateau",
        final_step=12,
        train_episodes=(0, 1),
        val_episodes=(32, 33),
    )
    assert metadata["artifact"] == "cosmos_video_lora_training_checkpoint"
    assert metadata["lora"] == {
        "rank": 16,
        "alpha": 16.0,
        "trainable_parameters": 123,
        "adapter_targets": ["blocks.0.self_attn.q_proj"],
        "adapter_dtype": "float32",
    }
    assert metadata["best_step"] == 12
    assert metadata["best_val_video_loss"] == 0.25
    assert metadata["video_objective"]["future_input"].startswith("future latent slots are noised")
    assert json.loads(json.dumps(metadata))["dataset"]["val_episodes"] == [32, 33]
