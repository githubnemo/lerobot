from pathlib import Path

import pytest

from lerobot.policies.vam.cosmos_cache_dataset import CacheManifest
from scripts.video_vam.train_smolexpert_on_cosmos import (
    CANONICAL_DATASET,
    CANONICAL_REVISION,
    LTX_PROMPT_ARTIFACT,
    context_spec_from_manifest,
)


def _manifest(provenance: dict, *, artifact: str | None = None) -> CacheManifest:
    payload = {
        "dataset": {"repo_id": CANONICAL_DATASET, "revision": CANONICAL_REVISION},
        "provenance": provenance,
    }
    if artifact is not None:
        payload["artifact"] = artifact
    return CacheManifest(Path("/cache/manifest.json"), payload, ())


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
