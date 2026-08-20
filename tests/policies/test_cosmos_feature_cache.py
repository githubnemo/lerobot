import json
from types import SimpleNamespace

import pytest
import torch

from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.cosmos_cache_dataset import (
    CosmosFeatureCacheDataset,
    CosmosFeatureCacheManifestError,
    load_cache_manifest,
    sha256_file,
    write_cache_manifest,
)
from lerobot.policies.vam.cosmos_feature_cache import (
    CosmosFeatureCacheArtifact,
    CosmosFeatureCacheValidationError,
    build_feature_cache_provenance,
    derive_window_seed,
    load_feature_cache,
    save_feature_cache,
)
from scripts.video_vam.smoke_test_cosmos_extractor import (
    prepare_sample,
    resolve_frame_index,
    run_timed_extraction,
)


def make_sample():
    config = CUBE_OUT_OF_BOX_CONTRACT
    camera = torch.zeros(config.sample_camera_shape, dtype=torch.uint8)
    for frame in range(config.history_length):
        camera[frame].fill_(frame)
    return {
        config.camera_key: camera,
        config.state_key: torch.zeros(config.sample_state_shape, dtype=torch.float32),
        config.action_key: torch.ones(config.sample_action_shape, dtype=torch.float32),
        f"{config.camera_key}_is_pad": torch.zeros(config.history_length, dtype=torch.bool),
        f"{config.state_key}_is_pad": torch.zeros(config.state_history_length, dtype=torch.bool),
        f"{config.action_key}_is_pad": torch.tensor([False] * 27 + [True] * 3),
        "task": config.task,
        "episode_index": torch.tensor(0),
        "timestamp": torch.tensor(0.4),
        "index": torch.tensor(4),
    }


def make_provenance(context, state, target_action, action_is_pad=None):
    config = CUBE_OUT_OF_BOX_CONTRACT
    if action_is_pad is None:
        action_is_pad = torch.zeros((1, 30), dtype=torch.bool)
    return build_feature_cache_provenance(
        dataset={
            "repo_id": config.repo_id,
            "revision": config.revision,
            "episode_index": 0,
            "frame_index": 4,
            "window_indices": [0, 1, 2, 3, 4],
            "window_offsets": [-4, -3, -2, -1, 0],
            "fps": 10,
        },
        source_shapes={
            "sample_camera": [5, 3, 480, 640],
            "sample_state": [1, 6],
            "sample_action": [30, 6],
            "rgb_history": [1, 3, 5, 480, 640],
            "prompt_embedding": [1, 512, 1024],
            "raw_hidden": [1, 16, 30, 40, 2048],
            "context": list(context.shape),
            "state": list(state.shape),
            "target_action": list(target_action.shape),
            "action_is_pad": list(action_is_pad.shape),
        },
        weights={
            "checkpoint_path": "/cache/generic.pt",
            "checkpoint_size_bytes": 1,
            "checkpoint_sha256": "a" * 64,
            "tokenizer_path": "/cache/tokenizer.pth",
            "tokenizer_size_bytes": 1,
            "tokenizer_sha256": "b" * 64,
            "checkpoint_kind": "generic_cosmos",
            "bridge_lora": None,
        },
        prompt_embedding={
            "artifact_path": "/cache/prompt.safetensors",
            "output_sha256": "c" * 64,
            "token_ids_sha256": "d" * 64,
            "shape": [1, 512, 1024],
            "dtype": "bfloat16",
        },
        extractor={
            "device": "cpu",
            "dtype": "bfloat16",
            "backend": "minimal_a2a",
            "high_noise_sigma": 10.0,
            "seed": 0,
            "noise_seed": 0,
            "hidden_layer": 20,
            "stop_after_step": 0,
            "input_shape": [1, 3, 5, 480, 640],
            "preprocess": "test pinned 480p",
            "conditioning": "test frame_replace",
            "official_resolution": "480",
            "official_positional_latent_max_h": 240,
            "official_positional_latent_max_w": 240,
            "bridge_lora": None,
            "extractor_input_keys": ["rgb_history", "prompt_embedding"],
            "excluded_from_extractor": ["state", "target_action"],
            "checkpoint_ignored_metadata_keys": [],
            "checkpoint_ignored_metadata_count": 0,
        },
        upstream_commits={
            "lerobot": "1" * 40,
            "mimic_video": "2" * 40,
            "vendor_manifest": "3" * 40,
        },
        runtime={
            "load_seconds": 1.0,
            "prompt_load_seconds": 0.1,
            "extraction_seconds": 2.0,
            "load_peak_allocated_bytes": 3,
            "load_peak_reserved_bytes": 4,
            "extraction_peak_allocated_bytes": 5,
            "extraction_peak_reserved_bytes": 6,
            "torch_version": "test",
            "cuda_version": "test",
            "gpu_name": "cpu",
            "python_version": "3.12",
        },
        context=context,
        state=state,
        target_action=target_action,
        action_is_pad=action_is_pad,
        raw_hidden_shape=(1, 16, 30, 40, 2048),
        raw_hidden_dtype=torch.bfloat16,
    )


def make_artifact():
    context = torch.arange(4 * 2048, dtype=torch.float32).reshape(1, 4, 2048).to(torch.bfloat16)
    state = torch.zeros((1, 1, 6), dtype=torch.float32)
    target_action = torch.ones((1, 30, 6), dtype=torch.float32)
    action_is_pad = torch.zeros((1, 30), dtype=torch.bool)
    return CosmosFeatureCacheArtifact(
        context,
        state,
        target_action,
        action_is_pad,
        make_provenance(context, state, target_action, action_is_pad),
    )


def test_fake_dataset_defaults_to_first_non_padded_window():
    dataset = SimpleNamespace(
        meta=SimpleNamespace(episodes=[{"dataset_from_index": 0, "dataset_to_index": 10}]),
        absolute_to_relative_idx={4: 0},
    )
    assert resolve_frame_index(dataset, episode_index=0) == 4
    assert resolve_frame_index(dataset, episode_index=0, window_start=2) == 6


def test_input_ordering_uses_causal_tchw_history_without_future_rgb_leakage():
    prepared = prepare_sample(make_sample(), frame_index=4)
    assert prepared.rgb_history.shape == (1, 3, 5, 480, 640)
    assert prepared.window_indices == (0, 1, 2, 3, 4)
    for frame in range(5):
        assert torch.all(prepared.rgb_history[0, :, frame] == frame)
    assert prepared.state.shape == (1, 1, 6)
    assert prepared.target_action.shape == (1, 30, 6)


def test_action_and_state_never_cross_extractor_boundary():
    prepared = prepare_sample(make_sample(), frame_index=4)
    prompt = torch.zeros((1, 512, 1024), dtype=torch.bfloat16)

    class FakeExtractor:
        def __init__(self):
            self.args = None

        def extract(self, images, prompt_embedding, *, noise_seed=None):
            self.args = (images, prompt_embedding, noise_seed)
            hidden = torch.zeros((1, 16, 30, 40, 2048), dtype=torch.bfloat16)
            return SimpleNamespace(
                hidden_grid=hidden,
                tokens=hidden.reshape(1, 19200, 2048),
                sigma=torch.tensor([10.0]),
                layer=20,
                provenance=SimpleNamespace(high_noise_sigma=10.0, stop_after_step=0),
            )

    extractor = FakeExtractor()
    run_timed_extraction(extractor, prepared, prompt, device=torch.device("cpu"))
    assert extractor.args is not None
    assert len(extractor.args) == 3
    assert extractor.args[0] is prepared.rgb_history
    assert extractor.args[1] is prompt
    assert extractor.args[2] is None


def test_roundtrip_provenance_and_label_roles(tmp_path):
    artifact = make_artifact()
    output = tmp_path / "episode-0.safetensors"
    save_feature_cache(artifact, output)
    loaded = load_feature_cache(output)
    assert torch.equal(loaded.context, artifact.context)
    assert torch.equal(loaded.state, artifact.state)
    assert torch.equal(loaded.target_action, artifact.target_action)
    assert torch.equal(loaded.action_is_pad, artifact.action_is_pad)
    payload = json.loads(output.with_suffix(".json").read_text())
    assert payload["dataset"]["revision"] == CUBE_OUT_OF_BOX_CONTRACT.revision
    assert payload["tensor_roles"]["context"] == "representation"
    assert payload["tensor_roles"]["state"] == "action_decoder_conditioning_not_extractor_input"
    assert payload["tensor_roles"]["target_action"] == "label_only_not_extractor_input"
    assert payload["output"]["context_shape"] == [1, 4, 2048]


def test_roundtrip_rejects_tampered_output_hash(tmp_path):
    artifact = make_artifact()
    output = tmp_path / "tampered.safetensors"
    save_feature_cache(artifact, output)
    sidecar = output.with_suffix(".json")
    payload = json.loads(sidecar.read_text())
    payload["output"]["context_sha256"] = "e" * 64
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(CosmosFeatureCacheValidationError, match="context_sha256"):
        load_feature_cache(output)


def test_overwrite_requires_explicit_opt_in(tmp_path):
    artifact = make_artifact()
    output = tmp_path / "episode-0.safetensors"
    save_feature_cache(artifact, output)
    with pytest.raises(FileExistsError, match="overwrite"):
        save_feature_cache(artifact, output)


def test_malformed_context_is_rejected_before_write(tmp_path):
    artifact = make_artifact()
    malformed = torch.zeros((1, 4, 2047), dtype=torch.bfloat16)
    with pytest.raises(CosmosFeatureCacheValidationError, match=r"\[B, N, 2048\]"):
        save_feature_cache(
            CosmosFeatureCacheArtifact(
                malformed, artifact.state, artifact.target_action, artifact.action_is_pad, artifact.provenance
            ),
            tmp_path / "bad.safetensors",
        )


def test_grad_requiring_context_is_detached_and_normal_cache_is_detached(tmp_path):
    artifact = make_artifact()
    context = artifact.context.float().requires_grad_().to(torch.bfloat16)
    output = tmp_path / "grad.safetensors"
    save_feature_cache(
        CosmosFeatureCacheArtifact(
            context, artifact.state, artifact.target_action, artifact.action_is_pad, artifact.provenance
        ),
        output,
    )
    assert load_feature_cache(output).context.requires_grad is False
    normal_output = tmp_path / "normal.safetensors"
    save_feature_cache(artifact, normal_output)
    assert load_feature_cache(normal_output).context.requires_grad is False


def test_window_seed_is_stable_and_window_specific():
    first = derive_window_seed(CUBE_OUT_OF_BOX_CONTRACT.revision, 0, 4, 17)
    assert first == derive_window_seed(CUBE_OUT_OF_BOX_CONTRACT.revision, 0, 4, 17)
    assert first != derive_window_seed(CUBE_OUT_OF_BOX_CONTRACT.revision, 0, 5, 17)
    assert first != derive_window_seed(CUBE_OUT_OF_BOX_CONTRACT.revision, 0, 4, 18)


def make_manifest(tmp_path, artifact):
    output = tmp_path / "episode-0000-frame-000004.safetensors"
    save_feature_cache(artifact, output)
    entry = {
        "sample_id": "episode-0000-frame-000004",
        "episode_index": 0,
        "frame_index": 4,
        "window_indices": [0, 1, 2, 3, 4],
        "noise_seed": 0,
        "safetensors": output.name,
        "sidecar": output.with_suffix(".json").name,
        "safetensors_sha256": sha256_file(output),
        "sidecar_sha256": sha256_file(output.with_suffix(".json")),
        "bytes": output.stat().st_size + output.with_suffix(".json").stat().st_size,
    }
    payload = {
        "schema_version": 1,
        "cache_schema_version": 2,
        "dataset": {
            "repo_id": CUBE_OUT_OF_BOX_CONTRACT.repo_id,
            "revision": CUBE_OUT_OF_BOX_CONTRACT.revision,
        },
        "subset": {"episodes": [0], "frame_start": None, "frame_end": None, "max_samples": 1},
        "provenance": {"builder": "test"},
        "global_seed": 0,
        "entries": [entry],
        "total_bytes": entry["bytes"],
        "runtime": {},
    }
    manifest = tmp_path / "manifest.json"
    write_cache_manifest(payload, manifest)
    return manifest, output


def test_manifest_loader_is_lazy_and_checks_artifact_hash_on_access(tmp_path):
    artifact = make_artifact()
    manifest, output = make_manifest(tmp_path, artifact)
    dataset = CosmosFeatureCacheDataset(manifest, verify_every_access=True)
    assert len(dataset) == 1
    assert dataset.manifest.entries[0].sample_id == "episode-0000-frame-000004"
    loaded = dataset[0]
    assert torch.equal(loaded.action_is_pad, artifact.action_is_pad)
    sidecar = output.with_suffix(".json")
    data = sidecar.read_bytes()
    sidecar.write_bytes(data.replace(b"0", b"1", 1))
    with pytest.raises(CosmosFeatureCacheManifestError, match="hash mismatch"):
        dataset[0]


def test_manifest_old_cache_schema_fails_clearly(tmp_path):
    artifact = make_artifact()
    manifest, _ = make_manifest(tmp_path, artifact)
    payload = json.loads(manifest.read_text())
    payload["cache_schema_version"] = 1
    manifest.write_text(json.dumps(payload))
    with pytest.raises(CosmosFeatureCacheManifestError, match="obsolete"):
        load_cache_manifest(manifest)
