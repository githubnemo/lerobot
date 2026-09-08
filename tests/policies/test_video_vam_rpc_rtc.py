from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rpc_server_parses_leftover_and_delay() -> None:
    server = _load("video_vam_rpc_server", "scripts/video_vam/rpc_server.py")
    leftover = [[float(i), 0.0, 0.0, 0.0, 0.0, 50.0] for i in range(10)]
    kwargs = server.rtc_kwargs_from_payload({"prev_chunk_left_over": leftover, "inference_delay": 3})
    assert kwargs["inference_delay"] == 3
    assert tuple(kwargs["prev_chunk_left_over"].shape) == (10, 6)
    assert kwargs["prev_chunk_left_over"][3, 0].item() == 3.0


def test_rpc_server_open_loop_omits_rtc_kwargs() -> None:
    server = _load("video_vam_rpc_server", "scripts/video_vam/rpc_server.py")
    assert server.rtc_kwargs_from_payload({"state": [0.0] * 6}) == {}


def test_rpc_server_rejects_delay_without_leftover() -> None:
    server = _load("video_vam_rpc_server", "scripts/video_vam/rpc_server.py")
    with pytest.raises(ValueError, match="inference_delay requires prev_chunk_left_over"):
        server.rtc_kwargs_from_payload({"inference_delay": 3})


def test_rpc_server_rejects_bad_leftover_shape() -> None:
    server = _load("video_vam_rpc_server", "scripts/video_vam/rpc_server.py")
    with pytest.raises(ValueError, match="prev_chunk_left_over"):
        server.rtc_kwargs_from_payload({"prev_chunk_left_over": [[1.0, 2.0]]})


def test_mac_leftover_payload_preserves_real_prefix() -> None:
    launcher = _load("run_mac_vam_rpc", "scripts/video_vam/run_mac_vam_rpc.py")
    assert launcher.leftover_payload(None) is None
    assert launcher.leftover_payload(torch.zeros(0, 6)) is None
    values = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    assert launcher.leftover_payload(values, 10) == values.tolist()
    assert len(launcher.leftover_payload(torch.ones(15, 6), 10)) == 15


def test_planned_inference_delay_ceils_control_steps() -> None:
    launcher = _load("run_mac_vam_rpc", "scripts/video_vam/run_mac_vam_rpc.py")
    assert launcher.planned_inference_delay(None, 10) == 0
    assert launcher.planned_inference_delay(0.0, 10) == 0
    assert launcher.planned_inference_delay(1.01, 10) == 11
    assert launcher.planned_inference_delay(1.5, 10) == 15


def test_should_restart_rpc_requires_exact_identity() -> None:
    launcher = _load("run_mac_vam_rpc", "scripts/video_vam/run_mac_vam_rpc.py")
    identity = {"policy": "smolvla", "checkpoint": "/resolved", "fingerprint": "abc"}
    health = {"ok": True, "identity": identity, "instance_id": "one"}
    assert not launcher.should_restart_rpc(health_payload=health, requested_identity=identity)
    for invalid in (None, {}, {"ok": True}, {**health, "identity": {**identity, "fingerprint": "new"}}):
        assert launcher.should_restart_rpc(health_payload=invalid, requested_identity=identity)


@pytest.mark.parametrize(
    "payload",
    [
        {"prev_chunk_left_over": [[float("nan")] * 6]},
        {"prev_chunk_left_over": [[0.0] * 6], "inference_delay": 2},
        {"inference_delay": -1},
        {"inference_delay": 1.5},
        {"inference_delay": True},
    ],
)
def test_rpc_rejects_invalid_rtc_inputs(payload):
    server = _load("video_vam_rpc_server", "scripts/video_vam/rpc_server.py")
    with pytest.raises(ValueError):
        server.rtc_kwargs_from_payload(payload)


def _fake_app(server, policy_name):
    from types import SimpleNamespace

    app = server.RPCApplication.__new__(server.RPCApplication)
    app.name = policy_name
    app.identity = {"protocol": 2}
    app.instance_id = "test"
    app.device = torch.device("cpu")
    app.horizon = 30
    app.execution_horizon = 10
    app.rtc_enabled = True
    app.original_space = "normalized" if policy_name == "smolvla" else "physical"
    app.lower = app.upper = None
    app.frame_index = 0
    app.history_key = "observation.images.front.history"
    app.history_index_key = "observation.history_frame_index"
    calls = []

    def predict(batch, **kwargs):
        calls.append((batch, kwargs))
        return torch.ones(1, 30, 6)

    policy = SimpleNamespace(
        config=SimpleNamespace(max_action_dim=32),
        reset=lambda: None,
        predict_action_chunk=predict,
        rtc_processor=object(),
    )
    app.policy = policy
    app.backend = (
        policy
        if policy_name == "video_vam"
        else SimpleNamespace(
            preprocessor=lambda obs: obs,
            postprocessor=lambda action: action.mul_(10),
            _device_and_dtype=lambda: (torch.device("cpu"), torch.float32),
        )
    )
    return app, calls


@pytest.mark.parametrize("policy", ["smolvla", "video_vam"])
def test_rpc_native_forwarding_and_original_action_space(monkeypatch, policy):
    server = _load("video_vam_rpc_server", "scripts/video_vam/rpc_server.py")
    app, calls = _fake_app(server, policy)
    frames = torch.zeros((480, 640, 3) if policy == "smolvla" else (5, 480, 640, 3), dtype=torch.uint8)
    monkeypatch.setattr(server, "_frame", lambda _: frames)
    payload = {
        "identity": app.identity,
        "instance_id": "test",
        "request_id": "r1",
        "state": [0.0] * 6,
        "images_front_u8_png_b64": "fake",
        "frame_index": 47,
        "prev_chunk_left_over": [[2.0] * 6] * 12,
        "inference_delay": 3,
        "original_space": app.original_space,
    }
    result = app.predict(payload)
    assert calls[0][1]["inference_delay"] == 3
    assert torch.equal(calls[0][1]["prev_chunk_left_over"], torch.full((12, 6), 2.0))
    assert result["original_chunk"][0] == [1.0] * 6
    assert result["action_chunk"][0] == ([10.0] * 6 if policy == "smolvla" else [1.0] * 6)
    if policy == "video_vam":
        assert calls[0][0][app.history_index_key].item() == 47
    assert result["request_id"] == "r1"
    for changes in (
        {"state": [float("inf")] * 6},
        {"instance_id": "old"},
        {"original_space": "wrong"},
        {"execution_horizon": 12},
    ):
        with pytest.raises(ValueError):
            app.predict({**payload, **changes})
    app.rtc_enabled = False
    with pytest.raises(ValueError, match="RTC-enabled"):
        app.predict(payload)


def test_checkpoint_identity_resolves_alias_and_detects_file_change(tmp_path):
    import argparse
    import json

    server = _load("video_vam_rpc_server", "scripts/video_vam/rpc_server.py")
    directory = tmp_path / "step 100"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"type": "smolvla"}))
    weights = directory / "model.safetensors"
    weights.write_bytes(b"one")
    alias = tmp_path / "last"
    alias.symlink_to(directory, target_is_directory=True)
    args = argparse.Namespace(
        checkpoint=alias,
        policy="smolvla",
        device="cpu",
        compile=False,
        rtc=True,
        execution_horizon=10,
        max_guidance_weight=10.0,
        joint_limits_min=None,
        joint_limits_max=None,
    )
    first = server.checkpoint_identity(args)
    assert first["checkpoint"] == str(directory)
    weights.write_bytes(b"two")
    assert server.checkpoint_identity(args)["fingerprint"] != first["fingerprint"]
    args.rtc = False
    assert server.checkpoint_identity(args)["config"] != first["config"]
    (directory / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        server.checkpoint_identity(args)
