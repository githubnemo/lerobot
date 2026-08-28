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
    kwargs = server.rtc_kwargs_from_payload({"prev_chunk_left_over": leftover, "inference_delay": 11})
    assert kwargs["inference_delay"] == 11
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


def test_mac_leftover_payload_pads_and_truncates() -> None:
    launcher = _load("run_mac_vam_rpc", "scripts/video_vam/run_mac_vam_rpc.py")
    assert launcher.leftover_payload(None, 10) is None
    assert launcher.leftover_payload(torch.zeros(0, 6), 10) is None

    padded = launcher.leftover_payload(torch.arange(18, dtype=torch.float32).reshape(3, 6), 5)
    assert padded is not None
    assert len(padded) == 5
    assert padded[0] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert padded[3] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    truncated = launcher.leftover_payload(torch.ones(15, 6), 10)
    assert truncated is not None
    assert len(truncated) == 10


def test_planned_inference_delay_ceils_control_steps() -> None:
    launcher = _load("run_mac_vam_rpc", "scripts/video_vam/run_mac_vam_rpc.py")
    assert launcher.planned_inference_delay(None, 10) == 0
    assert launcher.planned_inference_delay(0.0, 10) == 0
    assert launcher.planned_inference_delay(1.01, 10) == 11
    assert launcher.planned_inference_delay(1.5, 10) == 15


def test_should_restart_rpc_after_pull_or_missing_rtc() -> None:
    launcher = _load("run_mac_vam_rpc", "scripts/video_vam/run_mac_vam_rpc.py")
    restart = launcher.should_restart_rpc
    assert restart(updated=False, healthy=True, require_rtc=True, rtc_advertised=True) is False
    assert restart(updated=True, healthy=True, require_rtc=True, rtc_advertised=True) is True
    assert restart(updated=False, healthy=False, require_rtc=False, rtc_advertised=False) is True
    assert restart(updated=False, healthy=True, require_rtc=True, rtc_advertised=False) is True
    assert restart(updated=False, healthy=True, require_rtc=False, rtc_advertised=False) is False
