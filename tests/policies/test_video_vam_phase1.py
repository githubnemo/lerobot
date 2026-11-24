#!/usr/bin/env python
"""Unit tests for variable state_dim != action_dim VAM/EXPO support, stratified val_flow_loss,
SO-101 optional telemetry flags, and HF upload resolver.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from lerobot.policies.vam.expo_residual import ExpoResidualCritic
from lerobot.policies.vam.smol_expert import SmolVLANormalizer
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smol_normalizer_decoder_and_stratified_val_flow_loss() -> None:
    """Verify 18-D proprioception (pos+vel+current) + 6-D actions and deterministic val_flow_num_t=4."""
    train_mod = _load_script_module("train_smolexpert_test", "scripts/video_vam/train_smolexpert.py")

    state = torch.randn(4, 18)
    action = torch.randn(4, 30, 6)
    norm = SmolVLANormalizer.from_training_tensors(state, action)
    assert norm.state_mean.shape == (18,)
    assert norm.state_std.shape == (18,)
    assert norm.action_mean.shape == (6,)
    assert norm.action_std.shape == (6,)

    decoder = train_mod.build_synthetic_tiny_decoder(
        input_channels=32, device=torch.device("cpu"), normalizer=norm
    )
    assert decoder.state_dim == 18
    assert decoder.action_dim == 6

    ctx = torch.randn(2, 4, 32)
    s_in = torch.randn(2, 18)
    a_in = torch.randn(2, 30, 6)
    loss = decoder.flow_matching_loss(state=s_in, action=a_in, context=ctx)
    assert torch.isfinite(loss)
    sampled = decoder.sample_actions(s_in, ctx, num_steps=2, use_cuda_graph=False)
    assert sampled.shape == (2, 30, 6)

    # Verify evaluate_validation is deterministic and supports val_flow_num_t > 1
    class _Item:
        def __init__(self, idx: int) -> None:
            self.sample_id = f"ep0_f{idx}"
            self.state = torch.randn(18)
            self.action = torch.randn(30, 6)
            self.action_is_pad = torch.zeros(30, dtype=torch.bool)
            self.context = torch.randn(4, 32)

    class _FakeDataset:
        dataset_id = "test_ds"

        def __init__(self) -> None:
            torch.manual_seed(0)
            self._items = [_Item(0), _Item(1), _Item(2)]

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, i: int) -> _Item:
            return self._items[i]

    ds = _FakeDataset()
    r1 = train_mod.evaluate_validation(
        decoder, ds, device=torch.device("cpu"), batch_size=2, num_steps=2, seed=42, val_flow_num_t=4
    )
    r2 = train_mod.evaluate_validation(
        decoder, ds, device=torch.device("cpu"), batch_size=3, num_steps=2, seed=42, val_flow_num_t=4
    )
    assert r1["val_flow_loss"] == pytest.approx(r2["val_flow_loss"], rel=1e-5, abs=1e-6)
    assert r1["val_rmse"] == pytest.approx(r2["val_rmse"], rel=1e-5, abs=1e-6)


def test_expo_residual_critic_variable_state_dim() -> None:
    """Verify ExpoResidualCritic supports state_dim=18 with action_dim=6."""
    critic = ExpoResidualCritic(ctx_dim=64, state_dim=18, action_dim=6, chunk_len=30)
    ctx = torch.randn(2, 24, 64)
    state = torch.randn(2, 18)
    t_norm = torch.tensor([[0.1], [0.5]])
    base_actions = torch.randn(2, 30, 6)
    edited, feat = critic.forward_edit_with_feat(ctx, state, t_norm, base_actions)
    assert edited.shape == (2, 30, 6)
    q = critic.forward_q_from_feat(feat, state, t_norm, edited, edited)
    assert q.shape == (2,)


def test_so101_optional_servo_telemetry() -> None:
    """Verify SO101Follower keeps 6-D action_features while appending vel/load/current when enabled."""
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    def _make_mock_bus(*args, **kwargs):
        bus = MagicMock()
        bus.motors = dict.fromkeys(motor_names)
        bus.is_connected = True
        return bus

    with patch("lerobot.robots.so_follower.so_follower.SerialMotorsBus", side_effect=_make_mock_bus):
        robot_default = SO101Follower(SO101FollowerConfig(port="/dev/null"))
        assert len(robot_default.observation_features) == 6
        assert len(robot_default.action_features) == 6

        robot_full = SO101Follower(
            SO101FollowerConfig(
                port="/dev/null",
                record_velocity=True,
                record_load=True,
                record_current=True,
            )
        )

    obs_keys = list(robot_full.observation_features.keys())
    act_keys = list(robot_full.action_features.keys())
    assert len(act_keys) == 6
    assert len(obs_keys) == 24
    assert obs_keys[:6] == act_keys
    assert all(k.endswith(".vel") for k in obs_keys[6:12])
    assert all(k.endswith(".load") for k in obs_keys[12:18])
    assert all(k.endswith(".current") for k in obs_keys[18:24])

    robot_full.bus.sync_read.side_effect = [
        dict.fromkeys(motor_names, 10.0),
        dict.fromkeys(motor_names, 5),
        dict.fromkeys(motor_names, 120),
        dict.fromkeys(motor_names, 42),
    ]
    obs = robot_full.get_observation()
    assert len(obs) == 24
    assert obs["gripper.pos"] == 10.0
    assert obs["gripper.vel"] == 5.0
    assert obs["gripper.load"] == 120.0
    assert obs["gripper.current"] == 42.0


def test_push_checkpoint_to_hf_resolve_sources(tmp_path: Path) -> None:
    """Verify push_checkpoint_to_hf handles both VAM and lerobot-train layouts."""
    mod = _load_script_module("push_checkpoint_to_hf", "scripts/video_vam/push_checkpoint_to_hf.py")

    vam_dir = tmp_path / "exp001_vam"
    vam_dir.mkdir()
    (vam_dir / "best.safetensors").write_bytes(b"weights")
    (vam_dir / "last.safetensors").write_bytes(b"last")
    src, ignore = mod.resolve_upload_source(vam_dir, include_last=False)
    assert src == vam_dir
    assert "last.safetensors" in ignore

    lr_dir = tmp_path / "exp004_flux3"
    pm_dir = lr_dir / "checkpoints" / "002500" / "pretrained_model"
    pm_dir.mkdir(parents=True)
    (pm_dir / "config.json").write_text("{}")
    (pm_dir / "adapter_model.safetensors").write_bytes(b"lora")
    src2, _ = mod.resolve_upload_source(lr_dir)
    assert src2 == pm_dir.resolve()

    with pytest.raises(FileNotFoundError):
        mod.resolve_upload_source(tmp_path / "empty_missing")
