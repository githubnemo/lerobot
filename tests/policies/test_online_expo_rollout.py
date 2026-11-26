#!/usr/bin/env python
"""End-to-end simulated rollout test of run_mac_vam_rpc.py + rpc_server.py
both WITHOUT RL (standard RTC rollout) and WITH online Score-Centered EXPO RL.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from lerobot.policies.vam.smol_expert import SmolExpertActionDecoder, SmolVLANormalizer

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_mod(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_synthetic_vam_checkpoint(ckpt_dir: Path, *, backend: str = "cosmos3_edge") -> Path:
    """Create a valid VideoVAMPolicy checkpoint directory (normalizer + placeholder safetensors + config)."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base_state = torch.tensor([-0.3, -86.7, 88.2, 71.1, -53.4, 35.0])
    norm = SmolVLANormalizer.from_training_tensors(
        base_state.unsqueeze(0) + torch.randn(16, 6) * 5.0,
        base_state.view(1, 1, 6) + torch.randn(16, 30, 6) * 5.0,
    )
    save_file({"placeholder": torch.zeros(1)}, str(ckpt_dir / "best.safetensors"))
    save_file(
        {
            "state_mean": norm.state_mean,
            "state_std": norm.state_std,
            "action_mean": norm.action_mean,
            "action_std": norm.action_std,
        },
        str(ckpt_dir / "normalizer.safetensors"),
    )
    cfg = {
        "type": "video_vam",
        "backend": backend,
        "device": "cpu",
        "input_features": {
            "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.state": {"type": "STATE", "shape": [6]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        "camera_key": "observation.images.front",
        "cosmos3_layer": 24 if backend == "cosmos3_nano" else 20,
        "cosmos3_fps": 10.0,
        "cosmos3_prompt": "Grab the purple cube and put it into the cardboard box",
        "action_euler_steps": 2,
    }
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    return ckpt_dir


def test_sim_rollout_without_and_with_online_expo_rl(tmp_path: Path) -> None:
    """Verify run_mac_vam_rpc.py --no-rpc --sim works both without RL and with online EXPO RL."""
    ckpt_dir = _make_synthetic_vam_checkpoint(tmp_path / "c3_edge_sim_ckpt", backend="cosmos3_edge")
    expo_ckpt = tmp_path / "expo_online_sim.pt"
    debug_no_rl = tmp_path / "debug_no_rl"
    debug_with_rl = tmp_path / "debug_with_rl"

    run_mod = _load_mod("run_mac_vam_rpc_sim_test", "scripts/video_vam/run_mac_vam_rpc.py")
    train_mod = _load_mod("train_smolexpert_sim_test", "scripts/video_vam/train_smolexpert.py")

    def _fake_from_training_checkpoint(
        checkpoint_path, *, normalizer, input_channels=2048, device="cpu", num_steps=2, **kwargs
    ):
        del checkpoint_path, kwargs
        dec = train_mod.build_synthetic_tiny_decoder(
            input_channels=input_channels, normalizer=normalizer, device=device
        )
        dec.num_steps = num_steps
        return dec

    def _fake_build_extractor(self):
        return object(), torch.zeros(1, 1)

    def _fake_extract_context(self, images: torch.Tensor, feature_seed: int) -> torch.Tensor:
        del images, feature_seed
        ctx_dim = 4096 if self.config.backend == "cosmos3_nano" else 2048
        return torch.randn(1, 600, ctx_dim, device=self.config.device, dtype=torch.float32) * 0.1

    with (
        patch.object(SmolExpertActionDecoder, "from_training_checkpoint", _fake_from_training_checkpoint),
        patch(
            "lerobot.policies.vam.modeling_video_vam.VideoVAMPolicy._build_extractor", _fake_build_extractor
        ),
        patch(
            "lerobot.policies.vam.modeling_video_vam.VideoVAMPolicy._extract_context", _fake_extract_context
        ),
        patch.object(run_mod, "_ensure_cosmos3_backbone", lambda *a, **k: None),
    ):
        # 1. WITHOUT RL: standard in-process RTC rollout on simulated SO-101
        rc_no_rl = run_mod.main(
            [
                "--no-rpc",
                "--sim",
                "--policy",
                "video_vam",
                "--checkpoint",
                str(ckpt_dir),
                "--device",
                "cpu",
                "--no-compile",
                "--euler-steps",
                "2",
                "--fps",
                "10",
                "--camera-fps",
                "10",
                "--duration",
                "2.0",
                "--home-duration",
                "0.1",
                "--num-episodes",
                "1",
                "--debug-dir",
                str(debug_no_rl),
            ]
        )
        assert rc_no_rl == 0
        assert (debug_no_rl / "episode_001" / "chunks.jsonl").is_file()
        assert (debug_no_rl / "episode_001" / "ticks.jsonl").is_file()

        # 2. WITH ONLINE EXPO RL: 3 episodes, update critic after ep 2 & ep 3, save checkpoint
        rc_rl = run_mod.main(
            [
                "--no-rpc",
                "--sim",
                "--policy",
                "video_vam",
                "--checkpoint",
                str(ckpt_dir),
                "--device",
                "cpu",
                "--no-compile",
                "--euler-steps",
                "2",
                "--fps",
                "10",
                "--camera-fps",
                "10",
                "--duration",
                "2.0",
                "--home-duration",
                "0.1",
                "--online-expo-episodes",
                "3",
                "--online-update-interval",
                "2",
                "--online-update-steps",
                "15",
                "--expo-checkpoint",
                str(expo_ckpt),
                "--debug-dir",
                str(debug_with_rl),
            ]
        )
        assert rc_rl == 0
        assert expo_ckpt.is_file(), "Online EXPO RL should save updated ExpoResidualCritic checkpoint"
        saved = torch.load(expo_ckpt, map_location="cpu", weights_only=False)
        assert "state_dict" in saved and saved["config"]["ctx_dim"] == 2048
        assert saved["config"]["base_alpha"] == 1.0
        assert saved["config"]["max_delta_deg"] == 8.0
        # Verify residual weights actually moved from zero-init after online RL updates
        edit_w_keys = [k for k in saved["state_dict"] if k.startswith("edit_net.") and k.endswith(".weight")]
        assert edit_w_keys and float(saved["state_dict"][edit_w_keys[-1]].abs().max().item()) > 0.0

        # 3. Reload saved EXPO checkpoint in a subsequent rollout (eval with trained RL adapter!)
        rc_reload = run_mod.main(
            [
                "--no-rpc",
                "--sim",
                "--policy",
                "video_vam",
                "--checkpoint",
                str(ckpt_dir),
                "--device",
                "cpu",
                "--no-compile",
                "--euler-steps",
                "2",
                "--fps",
                "10",
                "--camera-fps",
                "10",
                "--duration",
                "1.5",
                "--home-duration",
                "0.1",
                "--num-episodes",
                "1",
                "--expo-checkpoint",
                str(expo_ckpt),
            ]
        )
        assert rc_reload == 0


def test_rpc_server_expo_auto_detects_cosmos3_nano_dim(tmp_path: Path) -> None:
    """Verify Cosmos 3 Nano (4096-dim) initializes ExpoResidualCritic with ctx_dim=4096 cleanly."""
    ckpt_dir = _make_synthetic_vam_checkpoint(tmp_path / "c3_nano_sim_ckpt", backend="cosmos3_nano")
    rpc_mod = _load_mod("rpc_server_nano_expo_test", "scripts/video_vam/rpc_server.py")
    train_mod = _load_mod("train_smolexpert_nano_test", "scripts/video_vam/train_smolexpert.py")

    def _fake_from_training_checkpoint(
        checkpoint_path, *, normalizer, input_channels=4096, device="cpu", num_steps=2, **kwargs
    ):
        del checkpoint_path, kwargs
        dec = train_mod.build_synthetic_tiny_decoder(
            input_channels=input_channels, normalizer=normalizer, device=device
        )
        dec.num_steps = num_steps
        return dec

    def _fake_build_extractor(self):
        return object(), torch.zeros(1, 1)

    with (
        patch.object(SmolExpertActionDecoder, "from_training_checkpoint", _fake_from_training_checkpoint),
        patch(
            "lerobot.policies.vam.modeling_video_vam.VideoVAMPolicy._build_extractor", _fake_build_extractor
        ),
    ):
        ns = argparse.Namespace(
            policy="video_vam",
            checkpoint=ckpt_dir,
            device="cpu",
            compile=False,
            rtc=True,
            execution_horizon=10,
            max_guidance_weight=2.0,
            joint_limits_min=[-180.0, -180.0, -180.0, -180.0, -180.0, 0.0],
            joint_limits_max=[180.0, 180.0, 180.0, 180.0, 180.0, 100.0],
            euler_steps=2,
            instance_id="test",
            jepa_world_model=None,
            jepa_planning_mode="none",
            jepa_num_candidates=8,
            expo_checkpoint=None,
            online_expo=True,
            shared_detector_checkpoint=None,
        )
        app = rpc_mod.RPCApplication(ns)
        assert app.expo_critic is not None
        assert app.expo_critic.ctx_dim == 4096
