import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.cosmos_cache_dataset import load_cache_manifest, write_cache_manifest
from lerobot.policies.vam.vam_split import (
    VAMSplitError,
    create_vam_split,
    get_train_entries,
    get_val_entries,
    load_vam_split,
)
from lerobot.policies.vam.world2action import ActionStateNormalizer, World2ActionConfig, World2ActionDecoder
from scripts.video_vam.build_cosmos_feature_cache import _selected_frames, parse_args
from scripts.video_vam.train_cosmos_world2action import (
    EarlyStopping,
    SafeWandbLogger,
    build_wandb_config,
    compute_training_normalizer,
    fixed_probe_loss,
    load_resume_checkpoint,
    save_training_checkpoint,
)


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, **kwargs):
        return torch.zeros_like(kwargs["xt_B_HA_A"]) + self.weight


def _entry(tmp_path, episode, frame):
    tensor = tmp_path / f"episode-{episode:04d}-frame-{frame:06d}.safetensors"
    sidecar = tensor.with_suffix(".json")
    tensor.write_bytes(f"tensor-{episode}-{frame}".encode())
    sidecar.write_text("{}")
    return {
        "sample_id": tensor.stem,
        "episode_index": episode,
        "frame_index": frame,
        "window_indices": list(range(frame - 4, frame + 1)),
        "noise_seed": episode + frame,
        "safetensors": tensor.name,
        "sidecar": sidecar.name,
        "safetensors_sha256": hashlib.sha256(tensor.read_bytes()).hexdigest(),
        "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "bytes": tensor.stat().st_size + sidecar.stat().st_size,
    }


def _manifest(tmp_path):
    entries = [_entry(tmp_path, 0, 4), _entry(tmp_path, 1, 4)]
    payload = {
        "schema_version": 1,
        "cache_schema_version": 2,
        "dataset": {
            "repo_id": CUBE_OUT_OF_BOX_CONTRACT.repo_id,
            "revision": CUBE_OUT_OF_BOX_CONTRACT.revision,
        },
        "subset": {"episodes": [0, 1], "frame_start": None, "frame_end": None, "max_samples": 2, "stride": 1},
        "provenance": {
            "builder": "test",
            "selection": {"stride": 1, "ordered_pairs": [[0, 4], [1, 4]]},
        },
        "global_seed": 0,
        "entries": entries,
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "runtime": {},
    }
    path = tmp_path / "manifest.json"
    write_cache_manifest(payload, path)
    return path


def test_stride_enumerates_episode_local_frames_and_filters_without_rephasing(monkeypatch):
    dataset = SimpleNamespace()
    rows = {
        0: {"dataset_from_index": 0, "dataset_to_index": 10},
        1: {"dataset_from_index": 10, "dataset_to_index": 20},
    }
    monkeypatch.setattr(
        "scripts.video_vam.build_cosmos_feature_cache._episode_row", lambda _, episode: rows[episode]
    )
    args = parse_args(["--episodes", "0", "1", "--stride", "3"])
    assert list(_selected_frames(dataset, args)) == [(0, 4), (0, 7), (1, 14), (1, 17)]
    args = parse_args(["--episodes", "0", "--stride", "3", "--frame-start", "6", "--frame-end", "10"])
    assert list(_selected_frames(dataset, args)) == [(0, 7)]


def test_split_round_trip_probe_and_mismatch_errors(tmp_path):
    manifest_path = _manifest(tmp_path)
    manifest = load_cache_manifest(manifest_path)
    assert manifest.vae_input_mode == "legacy_padded_vae"
    split_path = create_vam_split(
        manifest,
        tmp_path / "split.json",
        train_episodes=[0],
        val_episodes=[1],
        split_name="tiny",
        probe_seed=7,
    )
    split = load_vam_split(split_path, manifest)
    assert split.train_episodes == (0,)
    assert [entry.sample_id for entry in get_train_entries(manifest, split)] == ["episode-0000-frame-000004"]
    assert [entry.sample_id for entry in get_val_entries(manifest, split)] == ["episode-0001-frame-000004"]
    assert split.probe_for("episode-0001-frame-000004").epsilon_tensor().shape == (30, 6)
    payload = json.loads(split_path.read_text())
    payload["dataset"]["revision"] = "wrong"
    split_path.write_text(json.dumps(payload))
    with pytest.raises(VAMSplitError, match="dataset identity"):
        load_vam_split(split_path, manifest)


def test_split_fails_on_uncovered_manifest_episode(tmp_path):
    manifest_path = _manifest(tmp_path)
    manifest = load_cache_manifest(manifest_path)
    with pytest.raises(VAMSplitError, match="not covered"):
        create_vam_split(manifest, tmp_path / "split.json", train_episodes=[0], val_episodes=[])


def test_train_only_normalization_excludes_validation_extremes():
    train = [
        SimpleNamespace(
            state=torch.zeros(1, 1, 6),
            target_action=torch.zeros(1, 30, 6),
            action_is_pad=torch.zeros(1, 30, dtype=torch.bool),
        ),
        SimpleNamespace(
            state=torch.ones(1, 1, 6),
            target_action=torch.ones(1, 30, 6),
            action_is_pad=torch.zeros(1, 30, dtype=torch.bool),
        ),
    ]
    validation = SimpleNamespace(
        state=torch.full((1, 1, 6), 1000.0),
        target_action=torch.full((1, 30, 6), 1000.0),
        action_is_pad=torch.zeros(1, 30, dtype=torch.bool),
    )
    normalizer = compute_training_normalizer(train)
    assert torch.equal(normalizer.state_min, torch.zeros(6))
    assert torch.equal(normalizer.state_max, torch.ones(6))
    assert torch.equal(normalizer.action_min, torch.zeros(6))
    assert torch.equal(normalizer.action_max, torch.ones(6))
    assert not torch.equal(normalizer.action_max, validation.target_action.amax(dim=(0, 1)))


def test_fixed_probe_loss_is_deterministic():
    normalizer = ActionStateNormalizer.from_training_tensors(
        torch.stack([torch.zeros(6), torch.ones(6)]).reshape(2, 1, 6),
        torch.stack([torch.zeros(30, 6), torch.ones(30, 6)]),
    )
    decoder = World2ActionDecoder(
        World2ActionConfig(device="cpu"), denoiser=TinyDenoiser(), normalizer=normalizer
    )
    state = torch.zeros(2, 1, 6)
    action = torch.ones(2, 30, 6)
    context = torch.zeros(2, 2, 2048, dtype=torch.bfloat16)
    padding = torch.zeros(2, 30, dtype=torch.bool)
    tau = torch.tensor([0.25, 0.75])
    epsilon = torch.full((2, 30, 6), 0.5)
    first = fixed_probe_loss(decoder, state, action, context, padding, tau, epsilon)
    second = fixed_probe_loss(decoder, state, action, context, padding, tau, epsilon)
    assert torch.equal(first, second)


def test_early_stopping_triggers_after_patience():
    state = EarlyStopping(patience=2, min_delta=0.01)
    state, improved, stop = state.update(1.0)
    assert improved and not stop
    state, improved, stop = state.update(1.0)
    assert not improved and not stop
    state, improved, stop = state.update(1.0)
    assert not improved and stop


def test_resume_checkpoint_round_trip(tmp_path):
    decoder = World2ActionDecoder(World2ActionConfig(device="cpu"), denoiser=TinyDenoiser())
    optimizer = torch.optim.AdamW(decoder.denoiser.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    checkpoint = tmp_path / "last.safetensors"
    state_path = tmp_path / "last.state.pt"
    metadata = {
        "schema_version": 1,
        "artifact": "world2action_training_checkpoint",
        "checkpoint_kind": "last",
        "status": "validation_trained_non_rollout",
        "rollout_readiness": False,
        "robot_ready": False,
        "step": 3,
        "best_step": 2,
        "best_val_fixed_flow_loss": 1.0,
        "manifest": "manifest.json",
        "manifest_sha256": "a" * 64,
        "split": "split.json",
        "split_sha256": "b" * 64,
        "normalizer": "normalizer.safetensors",
        "normalizer_metadata": "normalizer.json",
        "backbone": {},
        "decoder_config": {},
        "parameter_count": {},
        "hyperparameters": {},
        "optimizer": {},
        "scheduler": {},
        "autocast": "cpu",
        "frozen_context_dtype": "bfloat16",
        "action_padding_semantics": "masked",
        "provenance": {},
        "resume_state": str(state_path),
    }
    save_training_checkpoint(decoder, optimizer, scheduler, path=checkpoint, metadata=metadata, rng_state={})
    decoder.denoiser.weight.data.fill_(9.0)
    restored_optimizer = torch.optim.AdamW(decoder.denoiser.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: 1.0)
    loaded, _ = load_resume_checkpoint(decoder, restored_optimizer, restored_scheduler, checkpoint)
    assert loaded["step"] == 3
    assert decoder.denoiser.weight.item() == pytest.approx(0.0)


def test_wandb_config_contains_comparison_identity(tmp_path):
    manifest_path = _manifest(tmp_path)
    manifest = load_cache_manifest(manifest_path)
    split_path = create_vam_split(manifest, tmp_path / "split.json", train_episodes=[0], val_episodes=[1])
    split = load_vam_split(split_path, manifest)
    args = SimpleNamespace(
        backbone_identity="backbone-a",
        backbone_checkpoint=None,
        batch_size=2,
        grad_accum_steps=3,
        seed=4,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=2,
    )
    config = build_wandb_config(
        args,
        manifest,
        split,
        {
            "identity": "backbone-a",
            "context_tokens": 19200,
            "context_channels": 2048,
            "context_adapter_used": False,
        },
        {"lr": 1e-4, "betas": [0.9, 0.99]},
        {"name": "lambdalinear", "cycle_lengths": 100},
    )
    assert config["backbone"]["context_tokens"] == 19200
    assert config["backbone"]["context_adapter_used"] is False
    assert config["manifest"]["cache_stride"] == 1
    assert config["split"]["train_episodes"] == [0]
    assert config["split"]["val_episodes"] == [1]
    assert config["split"]["validation_probe_seed"] == split.probe_seed
    assert config["training"]["batch_size"] == 2
    assert config["training"]["grad_accum_steps"] == 3
    assert config["dataset"]["repo_id"] == manifest.payload["dataset"]["repo_id"]


def test_wandb_metric_failure_disables_logger_without_raising(tmp_path, monkeypatch):
    class FailingRun:
        id = "test-run"
        url = "https://wandb.invalid/test-run"
        summary = {}

        def log(self, payload, step):
            raise RuntimeError("upload failed")

    class FakeWandb:
        @staticmethod
        def init(**kwargs):
            return FailingRun()

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
    logger = SafeWandbLogger(
        output_dir=tmp_path,
        project="test",
        run_name="test",
        tags=["test"],
        config={},
        resume=False,
        disabled=False,
    )
    logger.log_metrics({"train_loss": 1.0}, step=1)
    assert logger.enabled is False
