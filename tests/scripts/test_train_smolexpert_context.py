import json
from pathlib import Path

import pytest
import torch
from torch import nn

from lerobot.policies.vam.cosmos_cache_dataset import CacheManifest
from lerobot.policies.vam.ltx_layer_mix import LTXLayerAttentionMix
from scripts.video_vam.train_smolexpert_on_cosmos import (
    CANONICAL_DATASET,
    CANONICAL_REVISION,
    LTX_PROMPT_ARTIFACT,
    context_spec_from_manifest,
    load_joint_checkpoint,
    prepare_model_context,
    save_checkpoint,
)


def _manifest(provenance: dict, *, artifact: str | None = None) -> CacheManifest:
    payload = {
        "dataset": {"repo_id": CANONICAL_DATASET, "revision": CANONICAL_REVISION},
        "provenance": provenance,
    }
    if artifact is not None:
        payload["artifact"] = artifact
    return CacheManifest(Path("/cache/manifest.json"), payload, ())


def test_cosmos_multidepth_manifest_sets_attention_layers_and_width() -> None:
    manifest = _manifest(
        {
            "backbone": "Cosmos-Predict2-2B",
            "hidden_layer": 20,
            "deepest_layer": 20,
            "state_t": 2,
            "high_noise_sigma": 80.0,
            "tapped_layers": [4, 8, 12, 16, 18, 20],
            "context_transform": "none",
            "context_tokens_per_layer": 2_400,
            "context_channels": 2_048,
            "context_grid": [2, 30, 40],
            "context_dtype": "bfloat16",
        },
        artifact="cosmos_multidepth_statet2_none_feature_manifest",
    )
    spec = context_spec_from_manifest(manifest)
    assert (spec.stored_tokens, spec.model_tokens, spec.channels) == (2_400, 2_400, 2_048)
    assert spec.tapped_layers == (4, 8, 12, 16, 18, 20)
    mixer = LTXLayerAttentionMix(
        spec.tapped_layers, hidden_width=spec.channels, attn_width=2_048, num_heads=8
    )
    mixed = prepare_model_context(
        torch.zeros(1, len(spec.tapped_layers), spec.stored_tokens, spec.channels, dtype=torch.bfloat16),
        spec,
        mixer,
    )
    assert mixed.shape == (1, spec.stored_tokens, spec.channels)


def test_cosmos_manifest_preserves_default_pool2_contract() -> None:
    manifest = _manifest(
        {
            "high_noise_sigma": 80.0,
            "context_storage": "detached bfloat16 [B, 19200, 2048]",
        }
    )
    spec = context_spec_from_manifest(manifest)
    assert (spec.stored_tokens, spec.model_tokens, spec.channels) == (19_200, 4_800, 2_048)
    assert (spec.stored_transform, spec.model_transform, spec.dtype) == (
        "none",
        "pool2",
        "bfloat16",
    )


@pytest.mark.parametrize(
    ("transform", "tokens"),
    [("pool2", 640), ("none", 2_400)],
)
def test_ltx_manifest_drives_context_shape(transform: str, tokens: int) -> None:
    manifest = _manifest(
        {
            "backbone": "LTX-2.5-22B-distilled",
            "hidden_layer": 34,
            "high_noise_sigma": 1.0,
            "context_transform": transform,
            "context_tokens": tokens,
            "context_channels": 4_096,
            "context_dtype": "bfloat16",
            "prompt": {
                "artifact_path": LTX_PROMPT_ARTIFACT,
                "embedding_shape": [1, 1024, 4096],
                "embedding_dtype": "bfloat16",
                "fallback_allowed": False,
            },
        },
        artifact="ltx25_frozen_feature_manifest",
    )
    spec = context_spec_from_manifest(manifest)
    assert (spec.stored_tokens, spec.model_tokens, spec.channels) == (tokens, tokens, 4_096)
    assert spec.model_transform == transform


def test_ltx_manifest_rejects_zero_prompt_provenance() -> None:
    manifest = _manifest(
        {
            "backbone": "LTX-2.5-22B-distilled",
            "hidden_layer": 34,
            "high_noise_sigma": 1.0,
            "context_transform": "pool2",
            "context_tokens": 640,
            "context_channels": 4_096,
            "context_dtype": "bfloat16",
            "prompt": {
                "artifact_path": "/tmp/zero.safetensors",
                "embedding_shape": [1, 1024, 4096],
                "embedding_dtype": "bfloat16",
                "fallback_allowed": True,
            },
        },
        artifact="ltx25_frozen_feature_manifest",
    )
    with pytest.raises(ValueError, match="real Gemma-4"):
        context_spec_from_manifest(manifest)


def test_joint_checkpoint_roundtrip_preserves_mixer_and_provenance(tmp_path: Path) -> None:
    decoder = nn.Linear(4, 3)
    mixer = LTXLayerAttentionMix((8, 14, 20), hidden_width=4, attn_width=4, num_heads=1, seed=7)
    expected_decoder = {name: value.detach().clone() for name, value in decoder.state_dict().items()}
    expected_mixer = {name: value.detach().clone() for name, value in mixer.state_dict().items()}
    metadata = {
        "train_manifest_sha256": "a" * 64,
        "val_manifest_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "attention_mixer": {"layers": [8, 14, 20], "attn_width": 4, "attn_heads": 1},
    }
    checkpoint = tmp_path / "joint.safetensors"
    save_checkpoint(decoder, checkpoint, metadata, mixer)
    with torch.no_grad():
        for parameter in (*decoder.parameters(), *mixer.parameters()):
            parameter.zero_()

    load_joint_checkpoint(decoder, mixer, checkpoint)

    for name, value in expected_decoder.items():
        torch.testing.assert_close(decoder.state_dict()[name], value)
    for name, value in expected_mixer.items():
        torch.testing.assert_close(mixer.state_dict()[name], value)
    assert json.loads(checkpoint.with_suffix(".json").read_text()) == metadata
