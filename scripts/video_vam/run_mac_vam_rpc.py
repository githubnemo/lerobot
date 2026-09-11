#!/usr/bin/env python3
"""Mac-side Video-VAM & SmolVLA Universal Rollout Launcher.

Communicates with the model server on abakus via SSH HTTP RPC and streams
action chunks directly to the SO-101 / SO-100 robot arm at 10 Hz.

Features:
1. Automated Server Management: If the model server is not already running on abakus
   (or running a different policy/checkpoint), automatically starts it in tmux.
2. Real-Time Chunking (RTC): Enabled by default with execution delay compensation and
   native denoiser prefix guidance (the repository RTC approximation).
3. Multiple Policies:
   - --policy smolvla (converged 29.2k checkpoint, or new Scale-100 checkpoint)
   - --policy video_vam (Cosmos 2B backbone + SmolExpert action decoder)
4. RPC-only Dry-Run Verification:
   Pass --dry-run to test RPC inference only, without opening cameras or connecting motors.

Usage Examples:

    # 1. Test model inference over RPC without moving motors:
    python scripts/video_vam/run_mac_vam_rpc.py --policy smolvla --dry-run

    # 2. Live hardware rollout with SO-101 arm (RTC enabled by default).
    # Also supply --joint-limits-min and --joint-limits-max: six calibrated bounds each.
    # Existing calibration is required. Disconnect releases torque; support the arm.
    python scripts/video_vam/run_mac_vam_rpc.py \
        --policy smolvla \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodemXXXX \
        --robot.id=so101 \
        --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
        --duration=30 --task="take cube out of box"

    # 3. Rollout with newly trained Scale-100 model:
    python scripts/video_vam/run_mac_vam_rpc.py --policy smolvla --checkpoint latest_scale100 --dry-run
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import math
import shlex
import shutil
import subprocess
import time
import uuid
import warnings
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from io import BytesIO
from typing import Any

import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker

SSH_HOST = "abakus"


def _ssh_executable() -> str:
    path = shutil.which("ssh")
    if path is None:
        raise RuntimeError("ssh not found on PATH")
    return path


DEFAULT_PORT = 8766
REPO = "/home/anton/lerobot-video-vam"
SERVER_WRAPPER = "scripts/video_vam/run_rpc_server.sh"
IMAGE_SIZE = (640, 480)

# Checkpoints
DEFAULT_SMOLVLA_CHECKPOINT = "/home/anton/lerobot-video-vam/outputs/train/cube_out_of_box_il_smolvla_train_only_stats_0_31_20260826_1hr/checkpoints/029200/pretrained_model"
SCALE100_SMOLVLA_CHECKPOINT = "/home/anton/lerobot-video-vam/outputs/train/cube_out_of_box_scale100_smolvla_1hr/checkpoints/last/pretrained_model"
DEFAULT_COSMOS_CHECKPOINT = (
    "/home/anton/.cache/video-vam/runs/cosmos2b-videolora-smolexpert-20260828/smolexpert"
)

ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
MOTOR_NAMES = tuple(name.removesuffix(".pos") for name in ACTION_NAMES)
DEFAULT_CAMERAS = "{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
DEFAULT_MAX_RELATIVE_TARGET = {
    "shoulder_pan": 30.0,
    "shoulder_lift": 30.0,
    "elbow_flex": 30.0,
    "wrist_flex": 30.0,
    "wrist_roll": 30.0,
    "gripper": 50.0,
}


SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ControlMaster=no",
    "-o",
    "ControlPath=none",
]
MANAGED_OWNER = "lerobot-video-vam-rpc-v2"


def _ssh(args: list[str], *, check: bool = True, timeout: float = 15.0) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            [_ssh_executable(), *SSH_OPTIONS, SSH_HOST, shlex.join(args)],
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"SSH operation timed out after {timeout}s") from exc
    if check and completed.returncode:
        raise RuntimeError(completed.stderr.decode(errors="replace") or "SSH failed")
    return completed


def _healthz_payload(port: int) -> dict[str, Any] | None:
    result = _ssh(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "2",
            "--max-time",
            "3",
            f"http://127.0.0.1:{port}/healthz",
        ],
        check=False,
    )
    if result.returncode:
        return None
    try:
        value = json.loads(result.stdout)
    except (ValueError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _server_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--policy",
        args.policy,
        "--checkpoint",
        args.resolved_checkpoint,
        "--port",
        str(args.port),
        "--device",
        args.device,
        "--execution-horizon",
        str(args.execution_horizon),
        "--max-guidance-weight",
        str(args.max_guidance_weight),
        "--compile" if args.compile else "--no-compile",
        "--rtc" if args.inference_type == "rtc" else "--no-rtc",
    ]
    if args.joint_limits_min is not None:
        values += ["--joint-limits-min", *map(str, args.joint_limits_min)]
        values += ["--joint-limits-max", *map(str, args.joint_limits_max)]
    return values


def _describe_server(args: argparse.Namespace) -> dict[str, Any]:
    result = _ssh(
        ["bash", f"{REPO}/{SERVER_WRAPPER}", *_server_args(args), "--describe"], timeout=args.ready_timeout
    )
    value = json.loads(result.stdout)
    required = {"protocol", "policy", "checkpoint", "fingerprint", "server_code", "config"}
    if not isinstance(value, dict) or not required <= value.keys() or value["protocol"] != 2:
        raise RuntimeError("Remote server did not provide a complete protocol-v2 identity")
    if not value["fingerprint"] or not value["checkpoint"] or value["policy"] != args.policy:
        raise RuntimeError("Incomplete checkpoint identity")
    return value


def should_restart_rpc(*, health_payload: dict[str, Any] | None, requested_identity: dict[str, Any]) -> bool:
    return not (
        health_payload
        and health_payload.get("ok") is True
        and health_payload.get("identity") == requested_identity
        and health_payload.get("instance_id")
    )


# This lock serializes only tmux lifecycle operations for one port, never GPU work.
REMOTE_MANAGER = r"""
import fcntl, json, pathlib, shlex, socket, subprocess, sys, time
op, port, owner, token, repo, wrapper, encoded_args = sys.argv[1:]
port = int(port)
session = 'video-vam-rpc-' + str(port)
log = '/tmp/video-vam-rpc-' + str(port) + '-' + token + '.log'
def tmux(*args):
    return subprocess.run(['tmux', *args], capture_output=True, text=True, timeout=5)
def session_id():
    result = tmux('list-sessions', '-F', '#{session_name}\t#{session_id}')
    for line in result.stdout.splitlines():
        name, separator, identifier = line.partition('\t')
        if separator and name == session:
            return identifier
    return None
def exists():
    return session_id() is not None
def option(name):
    target = session_id()
    if target is None:
        return ''
    return tmux('show-options', '-qv', '-t', target, name).stdout.strip()
def startup_log():
    path = pathlib.Path(log)
    if not path.is_file():
        return '(startup log not created)'
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 4000))
        return stream.read().decode(errors='replace')
def startup_error(message):
    return RuntimeError(message + '\nStartup log: ' + log + '\n' + startup_log())
def listening():
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(('127.0.0.1', port)) == 0
with open('/tmp/video-vam-rpc-' + str(port) + '.lifecycle.lock', 'a') as lock:
    lock_deadline = time.monotonic() + 3
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= lock_deadline:
                raise RuntimeError('Lifecycle operation already in progress')
            time.sleep(0.1)
    if op == 'stop':
        if exists() and option('@vam_owner') == owner and option('@vam_instance') == token:
            result = tmux('kill-session', '-t', session_id())
            if result.returncode:
                raise RuntimeError(result.stderr)
            deadline = time.monotonic() + 10
            while listening() and time.monotonic() < deadline:
                time.sleep(0.1)
            if listening():
                raise RuntimeError('Managed session stopped but port is still occupied; no other process was killed')
        print(json.dumps({'stopped': True}))
    elif op == 'status':
        alive = exists() and option('@vam_owner') == owner and option('@vam_instance') == token
        text = startup_log()
        print(json.dumps({'alive': alive, 'log': text}))
    elif op == 'start':
        if exists():
            if option('@vam_owner') != owner:
                raise RuntimeError('Refusing to replace an unowned tmux session')
            # Never kill a listener just because it occupies our requested port.
            if listening():
                import urllib.request
                with urllib.request.urlopen('http://127.0.0.1:' + str(port) + '/healthz', timeout=3) as response:
                    health = json.load(response)
                if health.get('instance_id') != option('@vam_instance'):
                    raise RuntimeError('Listener does not belong to the managed instance')
            result = tmux('kill-session', '-t', session_id())
            if result.returncode:
                raise RuntimeError(result.stderr)
            deadline = time.monotonic() + 10
            while listening() and time.monotonic() < deadline:
                time.sleep(0.1)
        if listening():
            raise RuntimeError('Port occupied by an unmanaged or still-stopping process')
        command = shlex.join(['bash', str(pathlib.Path(repo) / wrapper), *json.loads(encoded_args), '--instance-id', token])
        command = 'exec ' + command + ' >' + shlex.quote(log) + ' 2>&1'
        # Set ownership before launching a child that could exit immediately.
        result = tmux('new-session', '-d', '-P', '-F', '#{session_id}', '-s', session, 'exec sleep 30')
        if result.returncode:
            raise startup_error(result.stderr)
        target = result.stdout.strip()
        try:
            if not target or target != session_id():
                raise startup_error('Could not identify the reserved tmux session')
            # Option commands do not consistently accept exact-name targets across tmux versions.
            for key, value in (('@vam_owner', owner), ('@vam_instance', token)):
                result = tmux('set-option', '-t', target, key, value)
                if result.returncode:
                    raise startup_error(result.stderr)
            result = tmux('respawn-window', '-k', '-t', target + ':', command)
            if result.returncode:
                raise startup_error(result.stderr)
            if session_id() != target:
                raise startup_error('Managed server exited during startup')
        except BaseException:
            if target:
                tmux('kill-session', '-t', target)
            raise
        print(json.dumps({'instance_id': token, 'log': log}))
    else:
        raise ValueError('unknown operation')
"""


def _manage(args: argparse.Namespace, operation: str, token: str) -> dict[str, Any]:
    result = _ssh(
        [
            "python3",
            "-c",
            REMOTE_MANAGER,
            operation,
            str(args.port),
            MANAGED_OWNER,
            token,
            REPO,
            SERVER_WRAPPER,
            json.dumps(_server_args(args)),
        ],
        timeout=75,
    )
    return json.loads(result.stdout)


def _wait_ready(args: argparse.Namespace, identity: dict[str, Any], token: str) -> dict[str, Any]:
    deadline = time.monotonic() + args.ready_timeout
    while time.monotonic() < deadline:
        health = _healthz_payload(args.port)
        if health is not None:
            if health.get("instance_id") != token:
                raise RuntimeError("Different server instance appeared during startup")
            if health is None or should_restart_rpc(health_payload=health, requested_identity=identity):
                raise RuntimeError("Loaded server identity differs from requested checkpoint/config")
            return health
        status = _manage(args, "status", token)
        if not status["alive"]:
            raise RuntimeError(f"Managed server exited during startup: {status['log']}")
        time.sleep(0.5)
    raise RuntimeError(
        f"Server not ready within {args.ready_timeout}s; see /tmp/video-vam-rpc-{args.port}-{token}.log on {SSH_HOST}"
    )


def _finite_state(value: Any) -> list[float]:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tuple(tensor.shape) != (6,) or not torch.isfinite(tensor).all().item():
        raise ValueError("state/action step must contain six finite values")
    return tensor.tolist()


class VideoVAMRPCClient:
    def __init__(
        self,
        *,
        policy: str,
        port: int = DEFAULT_PORT,
        ssh_host: str = SSH_HOST,
        timeout: float = 120.0,
        health: dict[str, Any] | None = None,
    ) -> None:
        if policy not in {"smolvla", "video_vam"}:
            raise ValueError("policy must be smolvla or video_vam")
        self.policy, self.port, self.ssh_host = policy, port, ssh_host
        self.timeout, self.health = timeout, health or {}
        self.frames: deque[Any] = deque(maxlen=5 if policy == "video_vam" else 1)
        self.frame_times: deque[float] = deque(maxlen=self.frames.maxlen)
        self.frame_index = -1

    @staticmethod
    def _image(frame: Any) -> Any:
        import numpy as np
        from PIL import Image

        if isinstance(frame, Image.Image):
            image = frame.convert("RGB")
        elif isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.shape == (480, 640, 3):
            image = Image.fromarray(frame, "RGB")
        else:
            raise TypeError("front camera must supply an RGB uint8 [480,640,3] array or PIL image")
        if image.size != IMAGE_SIZE:
            raise ValueError("front camera must be 640x480")
        return image.copy()

    @staticmethod
    def _png_b64(frame: Any) -> str:
        stream = BytesIO()
        frame.save(stream, format="PNG")
        return base64.b64encode(stream.getvalue()).decode("ascii")

    def remember(self, frame: Any, *, timestamp: float | None = None) -> None:
        self.frames.append(self._image(frame))
        self.frame_times.append(time.monotonic() if timestamp is None else timestamp)
        self.frame_index += 1

    def history_spacing_ms(self) -> list[float]:
        times = list(self.frame_times)
        return [(b - a) * 1000 for a, b in zip(times, times[1:], strict=False)]

    def snapshot(
        self,
        state: list[float],
        *,
        task: str,
        feature_seed: int | None,
        leftover: torch.Tensor | None = None,
        inference_delay: int = 0,
    ) -> dict[str, Any]:
        if len(self.frames) != self.frames.maxlen:
            raise RuntimeError("observation history is not ready")
        now = time.monotonic()
        if now - self.frame_times[-1] > 0.175 or any(
            not 0.05 <= gap / 1000 <= 0.175 for gap in self.history_spacing_ms()
        ):
            raise RuntimeError("observation history is stale or not sampled at 10 Hz")
        return {
            "frames": list(self.frames),
            "state": _finite_state(state),
            "task": task,
            "feature_seed": feature_seed,
            "frame_index": self.frame_index,
            "identity": self.health["identity"],
            "instance_id": self.health["instance_id"],
            "request_id": uuid.uuid4().hex,
            "original_space": self.health["original_space"],
            "prev_chunk_left_over": leftover_payload(leftover),
            "inference_delay": inference_delay,
        }

    def request(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        payload = dict(snapshot)
        frames = [self._png_b64(frame) for frame in payload.pop("frames")]
        payload["images_front_u8_png_b64"] = frames if self.policy == "video_vam" else frames[-1]
        command = shlex.join(
            [
                "curl",
                "--fail-with-body",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "3",
                "--max-time",
                str(self.timeout),
                "-X",
                "POST",
                f"http://127.0.0.1:{self.port}/predict",
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                "@-",
            ]
        )
        try:
            completed = subprocess.run(
                [_ssh_executable(), *SSH_OPTIONS, self.ssh_host, command],
                input=json.dumps(payload, allow_nan=False).encode(),
                capture_output=True,
                check=False,
                timeout=self.timeout + 12,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("RPC transport timed out; no request will be retried") from exc
        if completed.returncode:
            raise RuntimeError(
                f"RPC failed: {completed.stderr.decode(errors='replace')} {completed.stdout.decode(errors='replace')[:1000]}"
            )
        result = json.loads(completed.stdout)
        if not isinstance(result, dict) or "error" in result:
            raise RuntimeError(f"RPC server error: {result}")
        for key in ("identity", "instance_id", "request_id", "original_space"):
            if result.get(key) != payload[key]:
                raise RuntimeError(f"RPC response {key} mismatch")
        _chunks_from_result(result, self.health["horizon"])
        return result


def planned_inference_delay(latency_s: float | None, fps: int) -> int:
    return 0 if not latency_s else math.ceil(latency_s * fps)


def leftover_payload(tensor: torch.Tensor | None, horizon: int | None = None) -> list[list[float]] | None:
    if tensor is None or tensor.numel() == 0:
        return None
    if tensor.ndim != 2 or tensor.shape[1] != 6 or not torch.isfinite(tensor).all().item():
        raise ValueError("leftover must be finite [T,6]")
    # Do not invent physical zero targets or truncate the available guidance prefix.
    return tensor.detach().cpu().float().tolist()


def _parse_max_relative_target(raw: str) -> dict[str, float]:
    parsed = ast.literal_eval(raw)
    if not isinstance(parsed, dict) or set(parsed) != set(MOTOR_NAMES):
        raise ValueError("max_relative_target must contain exactly the six motor names")
    result = {name: float(parsed[name]) for name in MOTOR_NAMES}
    if any(not math.isfinite(value) or value <= 0 for value in result.values()):
        raise ValueError("relative target limits must be finite and positive")
    return result


def _parse_cameras(raw: str) -> dict[str, Any]:
    import yaml  # type: ignore[import-untyped]

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    parsed = yaml.safe_load(raw)
    if not isinstance(parsed, dict) or set(parsed) != {"front"}:
        raise ValueError("RPC requires exactly a front camera")
    config = dict(parsed["front"])
    if config.pop("type", None) != "opencv":
        raise ValueError("RPC launcher currently supports an OpenCV front camera")
    camera = OpenCVCameraConfig(**config)
    if (camera.width, camera.height) != IMAGE_SIZE or str(camera.color_mode.value).lower() != "rgb":
        raise ValueError("front camera must be RGB 640x480")
    if camera.fps is None or not math.isfinite(camera.fps) or camera.fps < 10:
        raise ValueError("front camera must deliver at least 10 FPS")
    return {"front": camera}


def _chunk_to_action(step: list[float]) -> dict[str, float]:
    return dict(zip(ACTION_NAMES, _finite_state(step), strict=True))


def _chunks_from_result(result: dict[str, Any], horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    chunks = []
    for key in ("original_chunk", "action_chunk"):
        value = result.get(key)
        if not isinstance(value, list):
            raise RuntimeError(f"missing {key}")
        tensor = torch.tensor(value, dtype=torch.float32)
        if tuple(tensor.shape) != (horizon, 6) or not torch.isfinite(tensor).all().item():
            raise RuntimeError(f"{key} must be finite [{horizon},6]")
        chunks.append(tensor)
    return chunks[0], chunks[1]


def _sleep_until(target: float) -> None:
    remaining = target - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def _next_tick(previous: float, now: float, period: float) -> float:
    scheduled = previous + period
    return now + period if scheduled <= now else scheduled


def _bool_arg(value: str) -> bool:
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def _run_dry(args: argparse.Namespace, client: VideoVAMRPCClient) -> None:
    from PIL import Image

    image = Image.new("RGB", IMAGE_SIZE, (128, 128, 128))

    def fresh_snapshot() -> dict[str, Any]:
        # A blocking prediction may age the entire history; replace it at control cadence.
        for _ in range(client.frames.maxlen or 1):
            client.remember(image)
            time.sleep(0.1)
        snapshot = client.snapshot([0.0] * 6, task=args.task, feature_seed=args.feature_seed)
        snapshot.pop("prev_chunk_left_over")
        snapshot.pop("inference_delay")
        return snapshot

    snapshot = fresh_snapshot()
    started = time.monotonic()
    result = client.request(snapshot)
    elapsed = time.monotonic() - started
    original, actions = _chunks_from_result(result, client.health["horizon"])
    print(f"[DRY-RUN] Initial no-prefix request: {elapsed:.3f}s end-to-end, {len(actions)} finite actions.")

    if args.inference_type == "rtc":
        if len(original) < 2:
            raise RuntimeError("RTC dry-run requires at least two original actions")
        # Simulate one consumed action, not a hardware command. Preserve the actual
        # remaining prefix in the server's original space; never pad physical zeros.
        leftover = original[1:].clone()
        snapshot = fresh_snapshot()
        snapshot["prev_chunk_left_over"] = leftover_payload(leftover)
        snapshot["inference_delay"] = 1
        started = time.monotonic()
        result = client.request(snapshot)
        elapsed = time.monotonic() - started
        _, actions = _chunks_from_result(result, client.health["horizon"])
        print(
            f"[DRY-RUN] RTC prefix request: {elapsed:.3f}s end-to-end, {len(actions)} finite actions; "
            f"{len(leftover)} real {client.health['original_space']} prefix steps, inference_delay=1."
        )
    print("[DRY-RUN] RPC-only validation complete. Synthetic images/state; no camera or hardware was tested.")


def _make_robot(args: argparse.Namespace) -> Any:
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    if args.robot_type not in {"so101_follower", "so100_follower"}:
        raise ValueError("Only SO-100/101 followers are supported")
    config = SOFollowerRobotConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=args.robot_cameras,
        use_degrees=True,
        max_relative_target=args.robot_max_relative_target,
        disable_torque_on_disconnect=True,
    )
    return make_robot_from_config(config)


def _disconnect_robot(robot: Any) -> None:
    # is_connected requires all cameras too; partial connect failures need per-device cleanup.
    errors = []
    if robot.is_connected:
        try:
            robot.disconnect()
            return
        except Exception as exc:
            errors.append(exc)
    for camera in robot.cameras.values():
        if camera.is_connected:
            try:
                camera.disconnect()
            except Exception as exc:
                errors.append(exc)
    if robot.bus.is_connected:
        try:
            robot.bus.disconnect(disable_torque=True)
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise RuntimeError(f"Partial robot disconnect failed: {errors}")


def _run_rollout(args: argparse.Namespace, client: VideoVAMRPCClient) -> None:
    robot = _make_robot(args)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vam-rpc")
    future: Future[dict[str, Any]] | None = None
    rtc = args.inference_type == "rtc"
    queue = ActionQueue(RTCConfig(enabled=rtc, execution_horizon=args.execution_horizon))
    latencies = LatencyTracker(maxlen=20)
    horizon = client.health["horizon"]
    period = 1.0 / args.fps
    phase = "warmup"
    started = 0.0
    deadline = time.monotonic() + args.ready_timeout
    running = False
    lower = torch.tensor(args.joint_limits_min)
    upper = torch.tensor(args.joint_limits_max)
    try:
        # Do not open an interactive calibration workflow from an autonomous launcher.
        if not robot.calibration:
            raise RuntimeError("Existing robot calibration is required; calibrate separately")
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            raise RuntimeError("Robot calibration mismatch; calibrate separately")
        next_tick = time.monotonic()
        previous_observation = None
        while time.monotonic() < deadline:
            obs = robot.get_observation()
            observed_at = time.monotonic()
            if (
                running
                and previous_observation is not None
                and observed_at - previous_observation > 1.75 * period
            ):
                raise RuntimeError("Control deadline missed; refusing to execute stale actions")
            previous_observation = observed_at
            state = _finite_state([obs[name] for name in ACTION_NAMES])
            client.remember(obs["front"])
            if future is not None and future.done():
                result = future.result()
                original, actions = _chunks_from_result(result, horizon)
                latency = time.monotonic() - started
                future = None
                if phase == "warmup":
                    # Compilation latency is not a runtime estimate; request a fresh observation next.
                    phase = "initial"
                else:
                    if ((actions < lower) | (actions > upper)).any().item():
                        raise RuntimeError("Action chunk exceeds absolute joint limits")
                    delay = planned_inference_delay(latency, args.fps) if running and rtc else 0
                    if delay >= horizon:
                        raise RuntimeError("Inference exceeded the action horizon")
                    queue.merge(original, actions, delay)
                    latencies.add(latency)
                    if not running:
                        running = True
                        phase = "running"
                        deadline = time.monotonic() + args.duration
                        next_tick = time.monotonic()
            if len(client.frames) == client.frames.maxlen and future is None:
                threshold = max(args.prefetch_steps, planned_inference_delay(latencies.max(), args.fps) + 2)
                if rtc and running and threshold >= horizon:
                    raise RuntimeError("Measured RPC latency cannot sustain this policy horizon")
                need_request = not running or (queue.qsize() <= threshold if rtc else queue.empty())
                if need_request:
                    previous = queue.get_left_over() if rtc and running else None
                    delay = planned_inference_delay(latencies.max(), args.fps) if previous is not None else 0
                    if previous is not None and delay >= len(previous):
                        raise RuntimeError("Insufficient action prefix to cover inference")
                    snapshot = client.snapshot(
                        state,
                        task=args.task,
                        feature_seed=args.feature_seed,
                        leftover=previous,
                        inference_delay=delay,
                    )
                    started = time.monotonic()
                    future = executor.submit(client.request, snapshot)
            step = queue.get() if running else None
            if step is not None:
                action = _chunk_to_action(step.tolist())
                sent = robot.send_action(action)
                actual = _finite_state([sent[name] for name in ACTION_NAMES])
                if max(abs(a - b) for a, b in zip(actual, step.tolist(), strict=True)) > 1e-3:
                    raise RuntimeError("Robot safety limiter clipped a command; stopping rollout")
            elif running and rtc:
                raise RuntimeError(
                    "RTC action queue exhausted; stopping instead of retrying stale observations"
                )
            next_tick = _next_tick(next_tick, time.monotonic(), period)
            _sleep_until(min(next_tick, deadline))
        if not running:
            raise RuntimeError("Warmup/initial inference exceeded startup deadline")
    finally:
        if future is not None:
            future.cancel()
        try:
            _disconnect_robot(robot)
        finally:
            # Running subprocesses have hard deadlines. Hardware disconnect never waits on RPC.
            executor.shutdown(wait=True, cancel_futures=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy", choices=("smolvla", "video_vam"), default="smolvla", help="Model policy")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Custom checkpoint path or 'latest_scale100'"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"RPC port on abakus (default {DEFAULT_PORT})"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument("--rpc-timeout", type=float, default=120.0)
    parser.add_argument(
        "--joint-limits-min", type=float, nargs=6, default=[-180.0, -180.0, -180.0, -180.0, -180.0, 0.0]
    )
    parser.add_argument(
        "--joint-limits-max", type=float, nargs=6, default=[180.0, 180.0, 180.0, 180.0, 180.0, 100.0]
    )
    parser.add_argument("--max-guidance-weight", type=float, default=10.0)
    parser.add_argument("--keep-server", action="store_true", help="leave server running in tmux after exit")
    parser.add_argument("--stop-server", action="store_true", help="stop server even if already running")
    parser.add_argument(
        "--dry-run", action="store_true", help="test RPC connection without connecting motors"
    )
    parser.add_argument("--robot.type", dest="robot_type", default="so101_follower")
    parser.add_argument(
        "--robot.port", dest="robot_port", default=None, help="Serial port (/dev/tty.usbmodem*)"
    )
    parser.add_argument("--robot.id", dest="robot_id", default="so101")
    parser.add_argument("--robot.use_degrees", dest="robot_use_degrees", type=_bool_arg, default=True)
    parser.add_argument("--robot.cameras", dest="robot_cameras_raw", default=DEFAULT_CAMERAS)
    parser.add_argument("--robot.max_relative_target", dest="robot_max_relative_target_raw", default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--task", default="take cube out of box")
    parser.add_argument("--strategy.type", dest="strategy_type", default="base")
    parser.add_argument(
        "--inference.type",
        dest="inference_type",
        choices=("sync", "rtc"),
        default="rtc",
        help="Inference mode: 'rtc' (Real-Time Chunking, default) or 'sync'",
    )
    parser.add_argument(
        "--inference.rtc.execution_horizon",
        dest="execution_horizon",
        type=int,
        default=10,
        help="RTC leftover prefix length (default 10).",
    )
    parser.add_argument(
        "--prefetch-steps",
        type=int,
        default=12,
        help="Action-queue threshold to trigger background prefetch (default 12)",
    )
    parser.add_argument("--feature-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.fps != 10:
        parser.error("These checkpoints require 10 Hz observation/action cadence")
    if not 1 <= args.port <= 65535 or args.execution_horizon < 1 or args.prefetch_steps < 1:
        parser.error("invalid port, execution horizon, or prefetch threshold")
    if any(
        not math.isfinite(x) or x <= 0
        for x in (args.duration, args.ready_timeout, args.rpc_timeout, args.max_guidance_weight)
    ):
        parser.error("duration/timeouts/guidance must be finite and positive")
    if args.feature_seed < 0:
        parser.error("feature seed must be nonnegative")
    if args.strategy_type != "base":
        parser.error("only strategy.type=base is supported")
    if (args.joint_limits_min is None) != (args.joint_limits_max is None):
        parser.error("joint limits must be supplied together")
    if args.joint_limits_min is not None and (
        any(not math.isfinite(x) for x in args.joint_limits_min + args.joint_limits_max)
        or any(a >= b for a, b in zip(args.joint_limits_min, args.joint_limits_max, strict=True))
    ):
        parser.error("joint limits must be finite and ordered")
    if not args.dry_run and (not args.robot_use_degrees or args.joint_limits_min is None):
        parser.error("hardware requires degree units and explicit six-dimensional absolute joint limits")

    # Universal Policy & Checkpoint Aliases
    aliases = {
        "smolvla_v1": (
            "/home/anton/lerobot-video-vam/outputs/train/cube_out_of_box_il_smolvla_train_only_stats_0_31_20260826_1hr/checkpoints/029200/pretrained_model",
            "smolvla",
        ),
        "smolvla_v2": (
            "/home/anton/lerobot-video-vam/outputs/train/cube_out_of_box_scale100_smolvla_1hr/checkpoints/025000/pretrained_model",
            "smolvla",
        ),
        "cosmos2b_t16": (
            "/home/anton/lerobot-video-vam/outputs/train/cube-out-of-box-cosmos-pool2-smolexpert",
            "video_vam",
        ),
        "cosmos2b_t2_undistilled": (
            "/home/anton/lerobot-video-vam/outputs/train/cosmos2b-t2-undistilled-smolexpert",
            "video_vam",
        ),
        "cosmos2b_t2_distilled": (
            "/home/anton/lerobot-video-vam/outputs/train/cosmos2b-t2-distilled-smolexpert",
            "video_vam",
        ),
        "cosmos3_base": (
            "/home/anton/lerobot-video-vam/outputs/train/cosmos3-edge-undseq-smolexpert",
            "video_vam",
        ),
        "cosmos3_lora": (
            "/home/anton/lerobot-video-vam/outputs/train/v2-cosmos3-edge-lora-smolexpert",
            "video_vam",
        ),
    }

    if args.checkpoint in aliases:
        args.resolved_checkpoint, args.policy = aliases[args.checkpoint]
    elif args.checkpoint in {"latest_scale100", "scale100"}:
        args.resolved_checkpoint, args.policy = aliases["smolvla_v2"]
    elif args.checkpoint is not None:
        args.resolved_checkpoint = args.checkpoint
    else:
        args.resolved_checkpoint = (
            DEFAULT_SMOLVLA_CHECKPOINT if args.policy == "smolvla" else DEFAULT_COSMOS_CHECKPOINT
        )

    if not args.dry_run:
        if not args.robot_port:
            raise SystemExit(
                "Hardware rollout requires --robot.port (e.g. --robot.port=/dev/tty.usbmodemXXXX)"
            )
        if not args.robot_id:
            raise SystemExit("Hardware rollout requires --robot.id")
        if args.robot_max_relative_target_raw:
            args.robot_max_relative_target = _parse_max_relative_target(args.robot_max_relative_target_raw)
        else:
            args.robot_max_relative_target = dict(DEFAULT_MAX_RELATIVE_TARGET)
        args.robot_cameras = _parse_cameras(args.robot_cameras_raw)

    token = None
    owned = False
    succeeded = False
    try:
        identity = _describe_server(args)
        # Pin a mutable `last` symlink before starting a process.
        args.resolved_checkpoint = identity["checkpoint"]
        health = _healthz_payload(args.port)
        if health is None or should_restart_rpc(health_payload=health, requested_identity=identity):
            token = uuid.uuid4().hex
            owned = True
            _manage(args, "start", token)
            health = _wait_ready(args, identity, token)
        else:
            token = health["instance_id"]
        if (
            health.get("action_dim") != 6
            or health.get("fps") != 10
            or health.get("original_space") != ("normalized" if args.policy == "smolvla" else "physical")
            or health.get("rtc") is not (args.inference_type == "rtc")
            or health.get("history_size") != (5 if args.policy == "video_vam" else 1)
            or health.get("action_units") != ["degrees"] * 5 + ["range_0_100"]
            or not isinstance(health.get("horizon"), int)
            or health["horizon"] <= args.execution_horizon
        ):
            raise RuntimeError("Server advertises incompatible or missing action metadata")
        client = VideoVAMRPCClient(
            policy=args.policy, port=args.port, timeout=args.rpc_timeout, health=health
        )
        if args.dry_run:
            _run_dry(args, client)
        else:
            _run_rollout(args, client)
        succeeded = True
    finally:
        if token and (args.stop_server or (owned and (not args.keep_server or not succeeded))):
            try:
                _manage(args, "stop", token)
            except Exception as exc:
                warnings.warn(f"Could not confirm managed-server cleanup: {exc}", stacklevel=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
