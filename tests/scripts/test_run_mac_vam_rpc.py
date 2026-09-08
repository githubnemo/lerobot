"""CPU-only fake hardware/transport tests; never start a model server."""

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launcher():
    spec = importlib.util.spec_from_file_location(
        "mac_rpc_test", ROOT / "scripts/video_vam/run_mac_vam_rpc.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clock(monkeypatch, launcher):
    value = SimpleNamespace(now=100.0)
    monkeypatch.setattr(launcher.time, "monotonic", lambda: value.now)
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: setattr(value, "now", value.now + seconds))
    return value


def health(policy="smolvla", horizon=50):
    return {
        "identity": {"protocol": 2},
        "instance_id": "one",
        "horizon": horizon,
        "original_space": "normalized" if policy == "smolvla" else "physical",
    }


def args(**changes):
    values = {
        "inference_type": "rtc",
        "execution_horizon": 10,
        "fps": 10,
        "duration": 4.0,
        "prefetch_steps": 12,
        "ready_timeout": 20.0,
        "task": "cube",
        "feature_seed": 0,
        "joint_limits_min": [-180.0] * 5 + [0.0],
        "joint_limits_max": [180.0] * 5 + [100.0],
    }
    return argparse.Namespace(**(values | changes))


class Robot:
    def __init__(self, module, clock):
        self.module, self.clock = module, clock
        self.is_connected = False
        self.is_calibrated = True
        self.calibration = {"existing": True}
        self.cameras = {}
        self.bus = SimpleNamespace(is_connected=False)
        self.sent = []
        self.disconnected = False

    def connect(self, *, calibrate):
        assert calibrate is False
        self.is_connected = True

    def get_observation(self):
        return dict.fromkeys(self.module.ACTION_NAMES, 0.0) | {
            "front": np.zeros((480, 640, 3), dtype=np.uint8)
        }

    def send_action(self, action):
        assert set(action) == set(self.module.ACTION_NAMES)
        self.sent.append((self.clock.now, dict(action)))
        return action

    def disconnect(self):
        self.disconnected = True
        self.is_connected = False


class Executor:
    def __init__(self, clock, delays):
        self.clock, self.delays = clock, iter(delays)
        self.snapshots = []
        self.shutdown_called = False
        self.active = None

    def submit(self, function, snapshot):
        assert self.active is None or self.active.done(), "duplicate in-flight inference"
        self.snapshots.append(snapshot)
        due = self.clock.now + next(self.delays, 0.2)
        future = SimpleNamespace(
            done=lambda: self.clock.now + 1e-9 >= due, result=lambda: function(snapshot), cancel=lambda: False
        )
        self.active = future
        return future

    def shutdown(self, **kwargs):
        self.shutdown_called = True


def rollout(monkeypatch, launcher, clock, *, policy="smolvla", delays=(0.3, 0.2), **options):
    robot = Robot(launcher, clock)
    executor = Executor(clock, delays)
    client = launcher.VideoVAMRPCClient(policy=policy, health=health(policy))
    monkeypatch.setattr(launcher, "_make_robot", lambda _: robot)
    monkeypatch.setattr(launcher, "ThreadPoolExecutor", lambda **_: executor)

    def predict(snapshot):
        original = torch.ones(50, 6)
        physical = original * 5
        physical[:, 5] = 50
        return {"original_chunk": original.tolist(), "action_chunk": physical.tolist()}

    monkeypatch.setattr(client, "request", predict)
    return robot, executor, client, args(**options)


@pytest.mark.parametrize("policy", ["smolvla", "video_vam"])
def test_single_flight_warmup_fresh_history_and_native_queue(monkeypatch, launcher, clock, policy):
    robot, executor, client, options = rollout(monkeypatch, launcher, clock, policy=policy)
    launcher._run_rollout(options, client)
    assert len(executor.snapshots) >= 3
    warmup, initial, rtc = executor.snapshots[:3]
    assert warmup["prev_chunk_left_over"] is initial["prev_chunk_left_over"] is None
    assert initial["frame_index"] > warmup["frame_index"]
    assert rtc["prev_chunk_left_over"][0] == [1.0] * 6  # original, NOT executed 5/50 targets
    assert len(warmup["frames"]) == (5 if policy == "video_vam" else 1)
    assert all(b[0] - a[0] >= 0.099 for a, b in zip(robot.sent, robot.sent[1:], strict=False))
    assert robot.disconnected and executor.shutdown_called


def test_sync_mode_never_supplies_rtc_prefix(monkeypatch, launcher, clock):
    robot, executor, client, options = rollout(
        monkeypatch, launcher, clock, inference_type="sync", duration=5.7
    )
    launcher._run_rollout(options, client)
    assert len(executor.snapshots) >= 3
    assert all(s["prev_chunk_left_over"] is None and s["inference_delay"] == 0 for s in executor.snapshots)
    assert robot.disconnected


def test_underrun_aborts_without_duplicate_fetch(monkeypatch, launcher, clock):
    robot, executor, client, options = rollout(
        monkeypatch, launcher, clock, delays=(0.3, 0.2, 5.0), duration=8.0
    )
    with pytest.raises(RuntimeError, match="queue exhausted"):
        launcher._run_rollout(options, client)
    assert len(executor.snapshots) == 3
    assert robot.disconnected and executor.shutdown_called


def test_clipped_command_stops(monkeypatch, launcher, clock):
    robot, executor, client, options = rollout(monkeypatch, launcher, clock)
    monkeypatch.setattr(robot, "send_action", lambda action: action | {"gripper.pos": 10.0})
    with pytest.raises(RuntimeError, match="clipped"):
        launcher._run_rollout(options, client)
    assert robot.disconnected


def test_interrupt_disconnects_before_waiting_for_worker(monkeypatch, launcher, clock):
    robot, executor, client, options = rollout(monkeypatch, launcher, clock)
    monkeypatch.setattr(robot, "get_observation", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(
        executor, "shutdown", lambda **_: pytest.fail("disconnected late") if not robot.disconnected else None
    )
    with pytest.raises(KeyboardInterrupt):
        launcher._run_rollout(options, client)
    assert robot.disconnected


def test_partial_connect_cleanup(launcher):
    closed = []
    robot = SimpleNamespace(
        is_connected=False,
        cameras={"front": SimpleNamespace(is_connected=True, disconnect=lambda: closed.append("camera"))},
        bus=SimpleNamespace(is_connected=True, disconnect=lambda **kwargs: closed.append(kwargs)),
    )
    launcher._disconnect_robot(robot)
    assert closed == ["camera", {"disable_torque": True}]


def test_camera_configuration_and_current_action_api(launcher):
    camera = launcher._parse_cameras(launcher.DEFAULT_CAMERAS)["front"]
    assert camera.width == 640 and camera.height == 480
    assert set(launcher._chunk_to_action([0.0] * 6)) == set(launcher.ACTION_NAMES)
    with pytest.raises(ValueError, match="finite"):
        launcher._chunk_to_action([float("nan")] * 6)
    for bad in ("{}", str(dict.fromkeys(launcher.MOTOR_NAMES, -1))):
        with pytest.raises(ValueError):
            launcher._parse_max_relative_target(bad)


def test_snapshot_is_detached_and_stale_history_rejected(launcher, clock):
    client = launcher.VideoVAMRPCClient(policy="video_vam", health=health("video_vam"))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        client.remember(frame)
        clock.now += 0.1
    snapshot = client.snapshot([0.0] * 6, task="cube", feature_seed=0)
    frame[:] = 255
    client.remember(frame)
    assert snapshot["frames"][-1].getpixel((0, 0)) == (0, 0, 0)
    clock.now += 1
    with pytest.raises(RuntimeError, match="stale"):
        client.snapshot([0.0] * 6, task="cube", feature_seed=0)


def test_no_catchup_bursts(launcher):
    assert launcher._next_tick(1.0, 1.4, 0.1) == 1.5
    assert launcher._next_tick(1.0, 1.05, 0.1) == 1.1


def test_transport_has_timeouts_quotes_and_rejects_wrong_instance(monkeypatch, launcher, clock):
    client = launcher.VideoVAMRPCClient(policy="smolvla", health=health(), timeout=7)
    client.remember(np.zeros((480, 640, 3), dtype=np.uint8))
    snapshot = client.snapshot([0.0] * 6, task="cube; 'quoted'", feature_seed=0)

    def run(command, **kwargs):
        assert command[1:9] == launcher.SSH_OPTIONS
        assert kwargs["timeout"] == 19
        assert "--max-time 7" in command[-1]
        payload = json.loads(kwargs["input"])
        return subprocess.CompletedProcess(
            command, 0, json.dumps(payload | {"instance_id": "old"}).encode(), b""
        )

    monkeypatch.setattr(launcher.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="instance_id mismatch"):
        client.request(snapshot)

    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired("ssh", kw["timeout"])

    monkeypatch.setattr(launcher.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        client.request(snapshot)


def test_ssh_quotes_whole_remote_argv(monkeypatch, launcher):
    captured = []
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda command, **kw: captured.append(command) or subprocess.CompletedProcess(command, 0, b"", b""),
    )
    argv = ["bash", "/path with spaces/wrapper", "--checkpoint", "/checkpoint; touch /tmp/never"]
    launcher._ssh(argv)
    assert shlex.split(captured[0][-1]) == argv


@pytest.mark.parametrize(
    "existing,owner,operation,token,listening,error",
    [
        (True, "foreign", "start", "new", False, "unowned"),
        (False, "", "start", "new", True, "occupied"),
        (True, "lerobot-video-vam-rpc-v2", "stop", "wrong", False, None),
        (False, "", "start", "new", False, None),
        (True, "lerobot-video-vam-rpc-v2", "start", "new", False, None),
        (True, "lerobot-video-vam-rpc-v2", "start", "new", True, None),
        (True, "lerobot-video-vam-rpc-v2", "stop", "old", False, None),
    ],
)
def test_managed_lifecycle_never_kills_foreign_processes(
    monkeypatch, launcher, tmp_path, existing, owner, operation, token, listening, error
):
    import builtins
    import socket

    calls = []
    state = {"exists": existing, "listening": listening}

    def tmux(command, **kwargs):
        calls.append(command)
        op = command[1]
        output = ""
        code = 0
        if op == "list-sessions":
            output = "video-vam-rpc-8766\t" + chr(36) + "12\n" if state["exists"] else ""
        elif op == "show-options":
            assert command[command.index("-t") + 1] == chr(36) + "12"
            output = owner if command[-1] == "@vam_owner" else "old"
        elif op == "kill-session":
            state["exists"] = False
            state["listening"] = False
        elif op == "new-session":
            state["exists"] = True
            output = chr(36) + "12\n"
            assert command[-1] == "exec sleep 30"
        elif op == "set-option":
            assert command[command.index("-t") + 1] == chr(36) + "12"
        elif op == "respawn-window":
            assert command[command.index("-t") + 1] == chr(36) + "12:"
            assert "'/checkpoint; literal'" in command[-1]
        return subprocess.CompletedProcess(command, code, output, "")

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def settimeout(self, value):
            pass

        def connect_ex(self, address):
            return 0 if state["listening"] else 1

    import io
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b'{"instance_id":"old"}'))
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda *a, **kw: real_open(tmp_path / "lock", "a"))
    monkeypatch.setattr(socket, "socket", Socket)
    monkeypatch.setattr(subprocess, "run", tmux)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manager",
            operation,
            "8766",
            launcher.MANAGED_OWNER,
            token,
            "/repo",
            "wrapper",
            json.dumps(["--checkpoint", "/checkpoint; literal"]),
        ],
    )
    if error:
        with pytest.raises(RuntimeError, match=error):
            exec(launcher.REMOTE_MANAGER, {})
    else:
        exec(launcher.REMOTE_MANAGER, {})
    should_kill = existing and owner == launcher.MANAGED_OWNER and (operation == "start" or token == "old")
    assert any(call[1] == "kill-session" for call in calls) == should_kill


def test_exact_readiness_mismatch_fails(monkeypatch, launcher, clock):
    monkeypatch.setattr(launcher, "_healthz_payload", lambda port: {"instance_id": "other"})
    with pytest.raises(RuntimeError, match="Different server"):
        launcher._wait_ready(argparse.Namespace(ready_timeout=1, port=8766), {"protocol": 2}, "ours")


@pytest.mark.parametrize("keep", [False, True])
def test_startup_failure_cleans_only_its_instance(monkeypatch, launcher, keep):
    operations = []
    monkeypatch.setattr(launcher, "_describe_server", lambda args: {"checkpoint": "/resolved"})
    monkeypatch.setattr(launcher, "_healthz_payload", lambda port: None)
    monkeypatch.setattr(launcher, "_manage", lambda args, op, token: operations.append((op, token)))
    monkeypatch.setattr(
        launcher, "_wait_ready", lambda *args: (_ for _ in ()).throw(RuntimeError("startup failed"))
    )
    with pytest.raises(RuntimeError, match="startup failed"):
        launcher.main(["--dry-run"] + (["--keep-server"] if keep else []))
    assert [op for op, _ in operations] == ["start", "stop"]
    assert operations[0][1] == operations[1][1]


def test_reuses_exact_server_without_stopping_it(monkeypatch, launcher):
    identity = {"checkpoint": "/resolved"}
    state = health() | {
        "ok": True,
        "identity": identity,
        "action_dim": 6,
        "fps": 10,
        "history_size": 1,
        "rtc": True,
        "action_units": ["degrees"] * 5 + ["range_0_100"],
    }
    monkeypatch.setattr(launcher, "_describe_server", lambda args: identity)
    monkeypatch.setattr(launcher, "_healthz_payload", lambda port: state)
    monkeypatch.setattr(launcher, "_manage", lambda *args: pytest.fail("must not restart/stop reused server"))
    monkeypatch.setattr(launcher, "_run_dry", lambda *args: None)
    assert launcher.main(["--dry-run"]) == 0


def test_runtime_deadline_miss_aborts(monkeypatch, launcher, clock):
    robot, executor, client, options = rollout(monkeypatch, launcher, clock)
    original = robot.get_observation

    def slow_observation():
        if robot.sent:
            clock.now += 0.3
        return original()

    monkeypatch.setattr(robot, "get_observation", slow_observation)
    with pytest.raises(RuntimeError, match="deadline missed"):
        launcher._run_rollout(options, client)
    assert len(robot.sent) == 1 and robot.disconnected


@pytest.mark.parametrize("key", ["action_chunk", "original_chunk"])
def test_rejects_nonfinite_chunks(launcher, key):
    result = {"action_chunk": [[0.0] * 6] * 50, "original_chunk": [[0.0] * 6] * 50}
    result[key] = [[float("inf")] * 6] * 50
    with pytest.raises(RuntimeError, match="finite"):
        launcher._chunks_from_result(result, 50)


@pytest.mark.parametrize("fail_child", [False, True])
def test_real_tmux_management_and_early_exit_logs(monkeypatch, launcher, tmp_path, capsys, fail_child):
    """Use an isolated tmux socket and dummy child: no GPU, HTTP server, or robot."""
    import shutil
    import socket
    import time
    import uuid

    executable = shutil.which("tmux")
    if executable is None:
        pytest.skip("tmux is not installed")
    socket_path = str(tmp_path / "tmux.sock")
    original_run = subprocess.run

    def isolated_tmux(command, **kwargs):
        assert command[0] == "tmux"
        return original_run([executable, "-S", socket_path, "-f", "/dev/null", *command[1:]], **kwargs)

    wrapper = tmp_path / "dummy wrapper.sh"
    marker = "intentional-startup-failure"
    wrapper.write_text(
        "#!/bin/bash\necho " + marker + " >&2\n" + ("exit 37\n" if fail_child else "exec sleep 20\n")
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    token = uuid.uuid4().hex
    log = Path(f"/tmp/video-vam-rpc-{port}-{token}.log")
    lock = Path(f"/tmp/video-vam-rpc-{port}.lifecycle.lock")
    monkeypatch.setattr(subprocess, "run", isolated_tmux)

    def manage(operation):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "manager",
                operation,
                str(port),
                launcher.MANAGED_OWNER,
                token,
                str(tmp_path),
                wrapper.name,
                "[]",
            ],
        )
        exec(launcher.REMOTE_MANAGER, {})
        return json.loads(capsys.readouterr().out)

    try:
        if fail_child:
            try:
                manage("start")
            except RuntimeError as exc:
                assert marker in str(exc)
                assert str(log) in str(exc)
            else:
                deadline = time.monotonic() + 3
                while True:
                    status = manage("status")
                    if not status["alive"]:
                        break
                    assert time.monotonic() < deadline
                    time.sleep(0.05)
                assert marker in status["log"]
        else:
            assert manage("start")["instance_id"] == token
            assert manage("status")["alive"] is True
        manage("stop")
        assert manage("status")["alive"] is False
    finally:
        original_run([executable, "-S", socket_path, "kill-server"], capture_output=True, timeout=5)
        log.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


@pytest.mark.parametrize("policy,horizon", [("smolvla", 50), ("video_vam", 30)])
def test_dry_run_exercises_real_original_prefix_and_refreshes_history(
    monkeypatch, launcher, clock, capsys, policy, horizon
):
    client = launcher.VideoVAMRPCClient(policy=policy, health=health(policy, horizon))
    original = torch.arange(horizon * 6, dtype=torch.float32).reshape(horizon, 6) / 100
    physical = original + 10
    physical[:, 5] = 50
    if policy == "video_vam":
        original = physical.clone()
    snapshots = []

    def request(snapshot):
        snapshots.append(snapshot)
        clock.now += 2.0 if len(snapshots) == 1 else 0.3
        return {"original_chunk": original.tolist(), "action_chunk": physical.tolist()}

    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(launcher, "_make_robot", lambda _: pytest.fail("dry-run accessed hardware"))
    launcher._run_dry(args(), client)
    assert len(snapshots) == 2
    first, second = snapshots
    assert "prev_chunk_left_over" not in first and "inference_delay" not in first
    assert second["prev_chunk_left_over"] == original[1:].tolist()
    assert len(second["prev_chunk_left_over"]) == horizon - 1
    assert second["inference_delay"] == 1
    assert second["original_space"] == client.health["original_space"]
    assert second["frame_index"] == first["frame_index"] + client.frames.maxlen
    assert first["request_id"] != second["request_id"]
    assert "execution_horizon" not in second  # retain immutable server configuration
    if policy == "smolvla":
        assert second["prev_chunk_left_over"] != physical[1:].tolist()
    output = capsys.readouterr().out
    assert "Initial no-prefix request: 2.000s" in output
    assert "RTC prefix request: 0.300s" in output
    assert "no camera or hardware was tested" in output


def test_sync_dry_run_sends_no_rtc_conditioning(monkeypatch, launcher, clock, capsys):
    client = launcher.VideoVAMRPCClient(policy="smolvla", health=health())
    snapshots = []

    def request(snapshot):
        snapshots.append(snapshot)
        return {"original_chunk": [[1.0] * 6] * 50, "action_chunk": [[10.0] * 6] * 50}

    monkeypatch.setattr(client, "request", request)
    launcher._run_dry(args(inference_type="sync"), client)
    assert len(snapshots) == 1
    assert not {"prev_chunk_left_over", "inference_delay", "execution_horizon"} & snapshots[0].keys()
    assert "RTC prefix request" not in capsys.readouterr().out


@pytest.mark.parametrize("failed_request", [1, 2])
def test_dry_run_validates_both_responses(monkeypatch, launcher, clock, failed_request):
    client = launcher.VideoVAMRPCClient(policy="smolvla", health=health())
    calls = []

    def request(snapshot):
        calls.append(snapshot)
        bad = float("nan") if len(calls) == failed_request else 1.0
        return {"original_chunk": [[bad] * 6] * 50, "action_chunk": [[10.0] * 6] * 50}

    monkeypatch.setattr(client, "request", request)
    with pytest.raises(RuntimeError, match="original_chunk must be finite"):
        launcher._run_dry(args(), client)
    assert len(calls) == failed_request
