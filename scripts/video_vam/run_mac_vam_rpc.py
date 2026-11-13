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
import contextlib
import http.client
import importlib.util
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
import warnings
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker

SSH_HOST = os.environ.get("VAM_SSH_HOST", "abakus")


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
DEFAULT_PORT_PATH = "/dev/tty.usbmodem5A460820701"
DEFAULT_ROBOT_ID = "shabby"
DEFAULT_CAMERAS = "{front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, warmup_s: 5}}"
DEFAULT_MAX_RELATIVE_TARGET = {
    "shoulder_pan": 45.0,
    "shoulder_lift": 45.0,
    "elbow_flex": 45.0,
    "wrist_flex": 45.0,
    "wrist_roll": 45.0,
    "gripper": 50.0,
}


SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ControlMaster=auto",
    "-o",
    "ControlPersist=10m",
    "-o",
    "ControlPath=/tmp/ssh_mux_%h_%p_%r",
]
MANAGED_OWNER = "lerobot-video-vam-rpc-v2"


def _is_local_host(host: str | None = None) -> bool:
    h = SSH_HOST if host is None else host
    return h in {"local", "localhost", "127.0.0.1"}


def _ssh(args: list[str], *, check: bool = True, timeout: float = 15.0) -> subprocess.CompletedProcess[bytes]:
    try:
        cmd = (
            ["bash", "-c", shlex.join(args)]
            if _is_local_host()
            else [_ssh_executable(), *SSH_OPTIONS, SSH_HOST, shlex.join(args)]
        )
        completed = subprocess.run(
            cmd,
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
    if getattr(args, "expo_checkpoint", None) is not None:
        values += ["--expo-checkpoint", str(args.expo_checkpoint)]
    if getattr(args, "shared_detector_checkpoint", None) is not None:
        values += ["--shared-detector-checkpoint", str(args.shared_detector_checkpoint)]
    if getattr(args, "euler_steps", None) is not None:
        values += ["--euler-steps", str(args.euler_steps)]
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
        ssh_host: str | None = None,
        timeout: float = 120.0,
        health: dict[str, Any] | None = None,
        fps: int = 10,
        in_process_app: Any | None = None,
    ) -> None:
        if policy not in {"smolvla", "video_vam"}:
            raise ValueError("policy must be smolvla or video_vam")
        self.policy, self.port, self.ssh_host = policy, port, (SSH_HOST if ssh_host is None else ssh_host)
        self.timeout, self.health = timeout, health or {}
        self.fps = fps
        self.in_process_app = in_process_app
        self.frames: deque[Any] = deque(maxlen=5 if policy == "video_vam" else 1)
        self.frame_times: deque[float] = deque(maxlen=self.frames.maxlen)
        self.frame_index = -1
        self._http_conn: Any = None

    @staticmethod
    def _image(frame: Any) -> Any:
        import numpy as np
        from PIL import Image

        if isinstance(frame, Image.Image):
            image = frame.convert("RGB")
        elif isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3:
            image = Image.fromarray(frame, "RGB")
        else:
            raise TypeError("front camera must supply an RGB uint8 [H,W,3] array or PIL image")
        if image.size != IMAGE_SIZE:
            image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        return image.copy()

    @staticmethod
    def _png_b64(frame: Any) -> str:
        import cv2
        import numpy as np

        arr = frame if isinstance(frame, np.ndarray) else np.array(frame)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 and arr.shape[2] == 3 else arr
        success, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if not success:
            raise RuntimeError("Failed to encode frame to JPEG")
        return base64.b64encode(enc).decode("ascii")

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
        period = 1.0 / max(1, self.fps)
        max_gap = max(0.175, 3.5 * period)
        min_gap = min(0.015, 0.3 * period)
        if now - self.frame_times[-1] > max_gap or any(
            not min_gap <= gap / 1000 <= max_gap for gap in self.history_spacing_ms()
        ):
            raise RuntimeError(f"observation history is stale or not sampled at {self.fps} Hz")
        max_h = self.health.get("horizon")
        trimmed_leftover = (
            leftover[:max_h] if (leftover is not None and isinstance(max_h, int) and max_h > 0) else leftover
        )
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
            "prev_chunk_left_over": leftover_payload(trimmed_leftover),
            "inference_delay": inference_delay,
        }

    def _post_local_http(self, path: str, body: bytes) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
        for attempt in range(2):
            had_conn = self._http_conn is not None
            conn = self._http_conn
            if conn is None:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
                self._http_conn = conn
            try:
                conn.request("POST", path, body=body, headers=headers)
                resp = conn.getresponse()
                raw = resp.read()
            except ConnectionRefusedError:
                with contextlib.suppress(Exception):
                    conn.close()
                self._http_conn = None
                if had_conn and attempt == 0:
                    continue
                raise
            except (http.client.HTTPException, BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                with contextlib.suppress(Exception):
                    conn.close()
                self._http_conn = None
                if had_conn and attempt == 0:
                    continue
                raise
            if resp.status != 200:
                text = raw.decode(errors="replace")[:1000]
                raise RuntimeError(f"HTTP {resp.status}: {text}")
            return json.loads(raw)
        raise RuntimeError("HTTP request failed after retry")

    def request(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        payload = dict(snapshot)
        result: dict[str, Any] | None = None
        if self.in_process_app is not None:
            pil_frames = payload.pop("frames")
            tensors = [
                torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).reshape(
                    img.height, img.width, 3
                )
                for img in pil_frames
            ]
            payload["images_front_u8_raw"] = (
                torch.stack(tensors, dim=0) if self.policy == "video_vam" else tensors[-1]
            )
            with self.in_process_app.predict_lock:
                result = self.in_process_app.predict(payload)
        else:
            frames = [self._png_b64(frame) for frame in payload.pop("frames")]
            payload["images_front_u8_png_b64"] = frames if self.policy == "video_vam" else frames[-1]
            body = json.dumps(payload, allow_nan=False).encode()
            try:
                result = self._post_local_http("/predict", body)
            except (
                http.client.HTTPException,
                BrokenPipeError,
                ConnectionResetError,
                TimeoutError,
                OSError,
            ) as transport_exc:
                if _is_local_host(self.ssh_host):
                    raise
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
                    cmd = [_ssh_executable(), *SSH_OPTIONS, self.ssh_host, command]
                    completed = subprocess.run(
                        cmd,
                        input=body,
                        capture_output=True,
                        check=False,
                        timeout=self.timeout + 12,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError("RPC transport timed out; no request will be retried") from exc
                if completed.returncode:
                    raise RuntimeError(
                        f"RPC failed: {completed.stderr.decode(errors='replace')} {completed.stdout.decode(errors='replace')[:1000]}"
                    ) from transport_exc
                result = json.loads(completed.stdout)
        if not isinstance(result, dict) or "error" in result:
            raise RuntimeError(f"RPC server error: {result}")
        for key in ("identity", "instance_id", "request_id", "original_space"):
            if result.get(key) != payload[key]:
                raise RuntimeError(f"RPC response {key} mismatch")
        _chunks_from_result(result, self.health["horizon"])
        return result

    def complete_episode(
        self,
        *,
        spawn_group: int = 0,
        update_critic: bool = False,
        update_steps: int = 600,
        save_checkpoint: str | None = None,
        manual_reward: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "identity": self.health["identity"],
            "instance_id": self.health["instance_id"],
            "spawn_group": int(spawn_group),
            "update_critic": bool(update_critic),
            "update_steps": int(update_steps),
            "save_checkpoint": save_checkpoint,
        }
        if manual_reward is not None:
            payload["manual_reward"] = float(manual_reward)
        if self.in_process_app is not None:
            with self.in_process_app.predict_lock:
                result = self.in_process_app.complete_episode(payload)
            if not isinstance(result, dict) or "error" in result:
                raise RuntimeError(f"RPC episode_complete error: {result}")
            return result
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
                f"http://127.0.0.1:{self.port}/episode_complete",
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                "@-",
            ]
        )
        cmd = (
            ["bash", "-c", command]
            if _is_local_host(self.ssh_host)
            else [_ssh_executable(), *SSH_OPTIONS, self.ssh_host, command]
        )
        completed = subprocess.run(
            cmd,
            input=json.dumps(payload, allow_nan=False).encode(),
            capture_output=True,
            check=False,
            timeout=self.timeout + 12,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Episode completion RPC failed: {completed.stderr.decode(errors='replace')} {completed.stdout.decode(errors='replace')[:1000]}"
            )
        result = json.loads(completed.stdout)
        if not isinstance(result, dict) or "error" in result:
            raise RuntimeError(f"RPC episode_complete error: {result}")
        return result

    def override_episode(self, *, ep_index: int, reward: float) -> dict[str, Any]:
        payload = {
            "ep_index": int(ep_index),
            "reward": float(reward),
        }
        if self.in_process_app is not None:
            with self.in_process_app.predict_lock:
                return self.in_process_app.override_episode(payload)
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
                f"http://127.0.0.1:{self.port}/override_episode",
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                "@-",
            ]
        )
        cmd = (
            ["bash", "-c", command]
            if _is_local_host(self.ssh_host)
            else [_ssh_executable(), *SSH_OPTIONS, self.ssh_host, command]
        )
        completed = subprocess.run(
            cmd,
            input=json.dumps(payload, allow_nan=False).encode(),
            capture_output=True,
            check=False,
            timeout=self.timeout + 12,
        )
        if completed.returncode:
            raise RuntimeError(f"Override RPC failed: {completed.stderr.decode(errors='replace')}")
        return json.loads(completed.stdout)


def _poll_key() -> str | None:
    """Non-blocking single-character poll from stdin if available."""
    import select
    import sys

    try:
        if not sys.stdin.isatty():
            return None
        rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
        if rlist:
            ch = sys.stdin.read(1)
            return ch.lower() if ch else None
    except Exception:
        return None
    return None


def planned_inference_delay(latency_s: float | None, fps: int) -> int:
    return 0 if not latency_s else math.ceil(latency_s * fps)


def leftover_payload(tensor: torch.Tensor | None, horizon: int | None = None) -> list[list[float]] | None:
    if tensor is None or tensor.numel() == 0:
        return None
    if tensor.ndim != 2 or tensor.shape[1] != 6 or not torch.isfinite(tensor).all().item():
        raise ValueError("leftover must be finite [T,6]")
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
            time.sleep(1.0 / max(1, client.fps))
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
        use_degrees=args.robot_use_degrees,
        max_relative_target=args.robot_max_relative_target,
        disable_torque_on_disconnect=False,
    )
    return make_robot_from_config(config)


def _soft_release_torque(robot: Any) -> None:
    """Softly releases motor torques one by one from end-effector to base."""
    import contextlib

    if not hasattr(robot, "bus") or not robot.bus.is_connected:
        return
    release_order = ["gripper", "wrist_roll", "wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan"]
    for motor in release_order:
        with contextlib.suppress(Exception):
            robot.bus.disable_torque(motor)
            time.sleep(0.15)


def _smooth_move_to(robot: Any, target_state: list[float], duration_s: float = 3.5, fps: int = 20) -> None:
    """Smoothly moves the arm from its current position to target_state using cosine ease-in-out."""
    try:
        obs = robot.get_observation()
        start_pos = [obs[name] for name in ACTION_NAMES]
        steps = max(1, int(duration_s * fps))
        period = 1.0 / fps
        for i in range(1, steps + 1):
            t0 = time.monotonic()
            alpha = (1.0 - math.cos(math.pi * (i / steps))) / 2.0
            interp = [s + alpha * (t - s) for s, t in zip(start_pos, target_state, strict=True)]
            action = _chunk_to_action(interp)
            robot.send_action(action)
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except Exception as exc:
        warnings.warn(f"Failed to complete smooth transition: {exc}", stacklevel=2)


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
            robot.bus.disconnect(disable_torque=False)
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise RuntimeError(f"Partial robot disconnect failed: {errors}")


def _run_single_episode(
    robot: Any,
    args: argparse.Namespace,
    client: VideoVAMRPCClient,
    executor: ThreadPoolExecutor,
) -> float | None:
    future: Future[dict[str, Any]] | None = None
    manual_reward: float | None = None
    rtc = args.inference_type == "rtc"
    queue = ActionQueue(RTCConfig(enabled=rtc, execution_horizon=args.execution_horizon))
    latencies = LatencyTracker(maxlen=20)
    horizon = client.health["horizon"]
    camera_fps = getattr(args, "camera_fps", None) or args.fps
    action_stride = max(1, camera_fps // args.fps)
    period = 1.0 / camera_fps
    phase = "warmup"
    started = 0.0
    deadline = time.monotonic() + args.ready_timeout
    running = False
    lower = torch.tensor(args.joint_limits_min)
    upper = torch.tensor(args.joint_limits_max)
    last_action = None
    starved_ticks = 0
    ticks_sent = 0
    ticks_since_merge = 0
    snapshot_ticks = 0
    obs_ticks = 0
    max_trackable_latency = min(1.5, max(0.35, (horizon - 4) / max(1, args.fps)))

    next_tick = time.monotonic()
    previous_observation = None
    try:
        while time.monotonic() < deadline:
            obs = robot.get_observation()
            observed_at = time.monotonic()
            if (
                running
                and previous_observation is not None
                and observed_at - previous_observation > 3.5 * period
            ):
                print(
                    f"[WARN] Observation loop jitter ({observed_at - previous_observation:.3f}s); continuing."
                )
            previous_observation = observed_at
            state = _finite_state([obs[name] for name in ACTION_NAMES])
            client.remember(obs["front"])
            if future is not None and future.done():
                result = future.result()
                original, actions = _chunks_from_result(result, horizon)
                latency = time.monotonic() - started
                det_p = float(result.get("reward_prob", 0.0))
                pre_p = float(result.get("pre_grasp_prob", 0.0))
                edit_norm = float(result.get("edit_norm", 0.0))
                chunk_idx = int(result.get("chunk_idx", 0))
                det_str = (
                    f" | chunk={chunk_idx} grip={state[5]:.1f}° pre_p={pre_p:.2f} det_p={det_p:.3f} Δa={edit_norm:.2f}°"
                    if chunk_idx > 0
                    else ""
                )
                server_s = result.get("latency_s")
                if client.in_process_app is not None:
                    print(
                        f"[LOCAL] Chunk received: {latency:.3f}s (phase: {phase}, queue: {queue.qsize()}{det_str})"
                    )
                elif (
                    isinstance(server_s, (int, float))
                    and not isinstance(server_s, bool)
                    and math.isfinite(server_s)
                ):
                    print(
                        f"[RPC] Chunk received: {latency:.3f}s (gpu: {float(server_s):.3f}s, phase: {phase}, queue: {queue.qsize()}{det_str})"
                    )
                else:
                    print(
                        f"[RPC] Chunk received: {latency:.3f}s (phase: {phase}, queue: {queue.qsize()}{det_str})"
                    )
                if running and result.get("reward_detected"):
                    print(
                        f"🎯 [LIVE REWARD DETECTED] MultiFrameSuccessDetector p={det_p:.3f} "
                        f"— completing drop (1.0s) and ending episode!"
                    )
                    deadline = min(deadline, time.monotonic() + 1.0)
                future = None
                if phase == "warmup":
                    # Warm up RTC guided branch as well before starting episode execution.
                    phase = "warmup_rtc" if rtc else "initial"
                elif phase == "warmup_rtc":
                    phase = "initial"
                else:
                    actions = torch.clamp(actions, min=lower, max=upper)
                    exact_delay = ticks_sent - snapshot_ticks if running and rtc else 0
                    delay = min(max(0, exact_delay), horizon - 1)
                    if running and rtc:
                        prev_remaining = queue.get_processed_left_over()
                        prev_orig_remaining = queue.get_left_over()
                        post_delay = actions[delay:].clone()
                        if prev_remaining is not None and len(prev_remaining) > 0 and len(post_delay) > 0:
                            blend_len = min(len(prev_remaining), len(post_delay), 6)
                            alpha = torch.linspace(
                                0.0, 1.0, blend_len + 2, dtype=actions.dtype, device=actions.device
                            )[1:-1].unsqueeze(-1)
                            post_delay[:blend_len] = (
                                (1.0 - alpha) * prev_remaining[:blend_len].to(actions.device, actions.dtype)
                                + alpha * post_delay[:blend_len]
                            )
                            actions = actions.clone()
                            actions[delay:] = post_delay
                            if prev_orig_remaining is not None and len(prev_orig_remaining) >= blend_len:
                                orig_post = original[delay:].clone()
                                orig_post[:blend_len] = (
                                    (1.0 - alpha)
                                    * prev_orig_remaining[:blend_len].to(original.device, original.dtype)
                                    + alpha * orig_post[:blend_len]
                                )
                                original = original.clone()
                                original[delay:] = orig_post
                        elif last_action is not None and len(post_delay) > 0:
                            held = torch.tensor(
                                [last_action[name] for name in ACTION_NAMES],
                                dtype=actions.dtype,
                                device=actions.device,
                            )
                            blend_len = min(len(post_delay), 6)
                            alpha = torch.linspace(
                                0.0, 1.0, blend_len + 2, dtype=actions.dtype, device=actions.device
                            )[1:-1].unsqueeze(-1)
                            post_delay[:blend_len] = (1.0 - alpha) * held.unsqueeze(0) + alpha * post_delay[:blend_len]
                            actions = actions.clone()
                            actions[delay:] = post_delay
                            if client.health.get("original_space") == "physical" and len(original) > delay:
                                orig_post = original[delay:].clone()
                                orig_blend_len = min(len(orig_post), blend_len)
                                orig_alpha = alpha[:orig_blend_len].to(
                                    device=original.device, dtype=original.dtype
                                )
                                orig_held = held.to(device=original.device, dtype=original.dtype)
                                orig_post[:orig_blend_len] = (
                                    (1.0 - orig_alpha) * orig_held.unsqueeze(0)
                                    + orig_alpha * orig_post[:orig_blend_len]
                                )
                                original = original.clone()
                                original[delay:] = orig_post
                    elif running and not rtc and last_action is not None and len(actions) > 0:
                        held = torch.tensor(
                            [last_action[name] for name in ACTION_NAMES],
                            dtype=actions.dtype,
                            device=actions.device,
                        )
                        blend_len = min(len(actions), 6)
                        alpha = torch.linspace(
                            0.0, 1.0, blend_len + 2, dtype=actions.dtype, device=actions.device
                        )[1:-1].unsqueeze(-1)
                        actions = actions.clone()
                        actions[:blend_len] = (1.0 - alpha) * held.unsqueeze(0) + alpha * actions[:blend_len]
                    queue.merge(original, actions, delay)
                    ticks_since_merge = 0
                    if latency < max_trackable_latency:
                        latencies.add(latency)
                    elif running:
                        print(f"[WARN] Ignoring outlier RPC latency spike ({latency:.3f}s) in delay tracker")
                    if not running:
                        running = True
                        phase = "running"
                        deadline = time.monotonic() + args.duration
                        next_tick = time.monotonic()
            if len(client.frames) == client.frames.maxlen and future is None:
                est_delay = planned_inference_delay(latencies.percentile(0.85), args.fps)
                threshold = min(horizon - 5, max(args.prefetch_steps, est_delay + 2))
                qsize = queue.qsize()
                min_exec_before_prefetch = 1 if qsize <= est_delay + 4 else min(4, max(1, qsize // 4))
                ready_after_merge = qsize == 0 or ticks_since_merge >= min_exec_before_prefetch
                need_request = not running or ((qsize <= threshold and ready_after_merge) if rtc else queue.empty())
                if need_request:
                    if phase == "warmup_rtc":
                        previous = torch.tensor([state] * max(1, horizon - 5), dtype=torch.float32)
                        delay = min(4, len(previous) - 1)
                    else:
                        previous = queue.get_left_over() if rtc and running else None
                        if previous is not None and 0 < len(previous) < est_delay + 4:
                            target_len = min(horizon - 4, est_delay + 4)
                            if target_len > len(previous):
                                pad = previous[-1:].expand(target_len - len(previous), -1)
                                previous = torch.cat([previous, pad], dim=0)
                        delay = est_delay if previous is not None else 0
                        if previous is not None and delay >= len(previous):
                            delay = max(0, len(previous) - 1)
                    snapshot = client.snapshot(
                        state,
                        task=args.task,
                        feature_seed=args.feature_seed,
                        leftover=previous,
                        inference_delay=delay,
                    )
                    started = time.monotonic()
                    snapshot_ticks = ticks_sent
                    future = executor.submit(client.request, snapshot)
            is_action_tick = (obs_ticks % action_stride) == 0
            obs_ticks += 1
            key = _poll_key()
            if running and key in ("s", "f") and manual_reward is None:
                if key == "s":
                    manual_reward = 1.0
                    print(
                        "\n⌨️ [HUMAN KEYPRESS 's'] Marked SUCCESS (1.0) — completing drop (0.8s) and ending episode!"
                    )
                    deadline = min(deadline, time.monotonic() + 0.8)
                else:
                    manual_reward = 0.0
                    print("\n⌨️ [HUMAN KEYPRESS 'f'] Marked FAIL (0.0) — ending episode immediately!")
                    deadline = min(deadline, time.monotonic())
            if is_action_tick:
                step = queue.get() if running else None
                if step is not None:
                    starved_ticks = 0
                    ticks_sent += 1
                    ticks_since_merge += 1
                    action = _chunk_to_action(step.tolist())
                    sent = robot.send_action(action)
                    last_action = sent
                    actual = _finite_state([sent[name] for name in ACTION_NAMES])
                    if max(abs(a - b) for a, b in zip(actual, step.tolist(), strict=True)) > 1e-3:
                        print("[SAFETY] Relative target limiter smoothed step delta; continuing.")
                        clamp_delta = torch.tensor(actual, dtype=torch.float32) - step.to(dtype=torch.float32)
                        with queue.lock:
                            if queue.queue is not None and queue.last_index < len(queue.queue):
                                rem = len(queue.queue) - queue.last_index
                                k = min(rem, 8)
                                decay = torch.linspace(
                                    1.0, 0.0, k + 1, dtype=queue.queue.dtype, device=queue.queue.device
                                )[:-1].unsqueeze(-1)
                                adj = decay * clamp_delta.to(
                                    device=queue.queue.device, dtype=queue.queue.dtype
                                )
                                reanchored = queue.queue[queue.last_index : queue.last_index + k] + adj
                                reanchored = torch.clamp(
                                    reanchored,
                                    min=lower.to(device=reanchored.device, dtype=reanchored.dtype),
                                    max=upper.to(device=reanchored.device, dtype=reanchored.dtype),
                                )
                                reanchored[:, 5] = torch.clamp(reanchored[:, 5], min=0.0, max=100.0)
                                queue.queue[queue.last_index : queue.last_index + k] = reanchored
                                if (
                                    client.health.get("original_space") == "physical"
                                    and queue.original_queue is not None
                                    and len(queue.original_queue) == len(queue.queue)
                                ):
                                    orig_reanchored = queue.original_queue[
                                        queue.last_index : queue.last_index + k
                                    ] + adj.to(
                                        device=queue.original_queue.device,
                                        dtype=queue.original_queue.dtype,
                                    )
                                    orig_reanchored = torch.clamp(
                                        orig_reanchored,
                                        min=lower.to(
                                            device=orig_reanchored.device, dtype=orig_reanchored.dtype
                                        ),
                                        max=upper.to(
                                            device=orig_reanchored.device, dtype=orig_reanchored.dtype
                                        ),
                                    )
                                    orig_reanchored[:, 5] = torch.clamp(
                                        orig_reanchored[:, 5], min=0.0, max=100.0
                                    )
                                    queue.original_queue[queue.last_index : queue.last_index + k] = (
                                        orig_reanchored
                                    )
                elif running and rtc:
                    if future is not None and last_action is not None:
                        starved_ticks += 1
                        if starved_ticks == 1 or starved_ticks % 15 == 0:
                            print(f"[RTC] Waiting for in-flight RPC chunk (holding pose, tick {starved_ticks})...")
                        robot.send_action(last_action)
                    else:
                        raise RuntimeError(
                            "RTC action queue exhausted; stopping instead of retrying stale observations"
                        )
                elif running and not rtc and future is not None and last_action is not None:
                    robot.send_action(last_action)
            next_tick = _next_tick(next_tick, time.monotonic(), period)
            _sleep_until(min(next_tick, deadline))
        if not running:
            raise RuntimeError("Warmup/initial inference exceeded startup deadline")
        return manual_reward
    finally:
        if future is not None:
            import contextlib

            if not future.cancel():
                with contextlib.suppress(Exception):
                    future.result(timeout=5.0)


def _run_rollout(args: argparse.Namespace, client: VideoVAMRPCClient) -> None:
    robot = _make_robot(args)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vam-rpc")
    home_pose: list[float] | None = None
    try:
        # Do not open an interactive calibration workflow from an autonomous launcher.
        if not robot.calibration:
            raise RuntimeError("Existing robot calibration is required; calibrate separately")
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            raise RuntimeError("Robot calibration mismatch; calibrate separately")

        # Capture initial pose or move to explicit dataset home pose for resets
        init_obs = robot.get_observation()
        initial_pose = [init_obs[name] for name in ACTION_NAMES]
        if getattr(args, "home_pose", None) is not None:
            home_pose = [float(x) for x in args.home_pose]
            print(
                f"[BASE POSE] Moving from {[f'{x:.1f}°' for x in initial_pose]} "
                f"to target home pose: {[f'{x:.1f}°' for x in home_pose]}"
            )
            _smooth_move_to(robot, home_pose, duration_s=getattr(args, "home_duration", 3.0))
        else:
            home_pose = initial_pose
            print(f"[BASE POSE] Recorded initial pose as home: {[f'{x:.1f}°' for x in home_pose]}")

        episodes = 0
        online_ep = getattr(args, "online_expo_episodes", None)
        num_ep = online_ep if (online_ep is not None and online_ep > 0) else getattr(args, "num_episodes", 1)
        target_episodes = num_ep if num_ep > 0 else float("inf")
        online_enabled = bool(
            (online_ep is not None and online_ep > 0)
            or getattr(args, "shared_detector_checkpoint", None) is not None
        )
        update_interval = max(1, int(getattr(args, "online_update_interval", 5)))
        import contextlib
        import sys
        import termios
        import tty

        old_term = None
        if sys.stdin.isatty():
            with contextlib.suppress(Exception):
                old_term = termios.tcgetattr(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())
        try:
            while episodes < target_episodes:
                episodes += 1
                print(f"\n{'=' * 65}")
                print(
                    f"🎬 Starting Episode {episodes} (Duration: {args.duration:.0f}s) "
                    f"[Press 's'=Success, 'f'=Fail anytime]"
                )
                print(f"{'=' * 65}")

                live_manual_reward = _run_single_episode(robot, args, client, executor)

                if online_enabled:
                    should_update = (episodes % update_interval == 0) or (episodes == target_episodes)
                    ep_stats = client.complete_episode(
                        spawn_group=getattr(args, "spawn_group", 0),
                        update_critic=should_update,
                        update_steps=getattr(args, "online_update_steps", 600),
                        save_checkpoint=getattr(args, "expo_checkpoint", None),
                        manual_reward=live_manual_reward,
                    )
                    rew_icon = "✅ SUCCESS (1.0)" if ep_stats.get("reward", 0.0) >= 0.5 else "❌ FAIL (0.0)"
                    src_tag = "HUMAN-KEY" if live_manual_reward is not None else "DETECTOR"
                    print(
                        f"[EXPO-ONLINE] Ep {episodes} ({src_tag}): {rew_icon} "
                        f"(p={ep_stats.get('reward_prob', 0.0):.3f}, "
                        f"group={ep_stats.get('spawn_group', 0)}, "
                        f"R_mean={ep_stats.get('group_mean_reward', 0.0):.2f}, "
                        f"A_dr_grpo={ep_stats.get('advantage', 0.0):+.2f})"
                    )
                    if ep_stats.get("critic_updated"):
                        det_loss_val = ep_stats.get("detector_loss")
                        det_msg = (
                            f" | Detector fine-tuned (loss={det_loss_val:.4f})"
                            if det_loss_val is not None
                            else ""
                        )
                        print(
                            f"  🔥 Updated ExpoResidualCritic on {ep_stats.get('episode_count', episodes)} "
                            f"episodes (loss={ep_stats.get('update_loss', 0.0):.4f}){det_msg}"
                        )

                if episodes < target_episodes:
                    print(f"\n[RESET] Episode {episodes} complete. Smoothly returning to base position...")
                    _smooth_move_to(robot, home_pose, duration_s=getattr(args, "home_duration", 2.0))
                    print(
                        f"[RESET] Base position reached. Reset the scene ({args.reset_time:.0f}s) "
                        f"[Press 's'=Override Success, 'f'=Override Fail]..."
                    )
                    reset_time = getattr(args, "reset_time", 6.0)
                    countdown_start = time.monotonic()
                    while time.monotonic() - countdown_start < reset_time:
                        key = _poll_key()
                        if online_enabled and key in ("s", "f"):
                            ov_rew = 1.0 if key == "s" else 0.0
                            ov_res = client.override_episode(ep_index=episodes - 1, reward=ov_rew)
                            ov_icon = "✅ SUCCESS (1.0)" if ov_rew >= 0.5 else "❌ FAIL (0.0)"
                            print(
                                f"\n  ⌨️ [HUMAN OVERRIDE '{key}'] Ep {episodes} -> {ov_icon} "
                                f"(Critic loss={ov_res.get('loss')}, Detector loss={ov_res.get('detector_loss')})"
                            )
                        remaining = reset_time - (time.monotonic() - countdown_start)
                        print(
                            f"  ⏳ Reset countdown: {int(remaining) + 1}s (press 's'/'f' to override)...",
                            end="\r",
                            flush=True,
                        )
                        time.sleep(0.2)
                    print("\n  ✅ Reset countdown complete! Starting next episode.")
                    client.frames.clear()
                    client.frame_times.clear()
        finally:
            if old_term is not None:
                with contextlib.suppress(Exception):
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_term)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Received Ctrl+C. Stopping episode loop...")
    finally:
        if home_pose is not None and robot.is_connected:
            print("[SHUTDOWN] Returning smoothly to base position...")
            _smooth_move_to(robot, home_pose, duration_s=getattr(args, "home_duration", 2.0))
        print("[SHUTDOWN] Softly releasing motor torques sequentially...")
        _soft_release_torque(robot)
        try:
            _disconnect_robot(robot)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def _resolve_local_checkpoint(raw_checkpoint: str) -> str:
    ckpt_path = Path(raw_checkpoint)
    if not ckpt_path.exists():
        local_repo = Path(__file__).resolve().parents[2]
        if not ckpt_path.is_absolute():
            candidate = local_repo / raw_checkpoint
            if candidate.exists():
                return str(candidate)
        elif raw_checkpoint.startswith("/home/anton/lerobot-video-vam/"):
            candidate = local_repo / raw_checkpoint.removeprefix("/home/anton/lerobot-video-vam/")
            if candidate.exists():
                return str(candidate)
    return raw_checkpoint


def _ensure_cosmos3_backbone(resolved_checkpoint: str, policy: str) -> None:
    if policy != "video_vam":
        return
    ckpt_cfg_path = Path(resolved_checkpoint) / "config.json"
    if ckpt_cfg_path.is_file():
        ckpt_cfg = json.loads(ckpt_cfg_path.read_text())
        if ckpt_cfg.get("backend") == "cosmos3_edge":
            from lerobot.policies.vam.modeling_video_vam import _resolve_cosmos3_checkpoint_path

            c3_path = _resolve_cosmos3_checkpoint_path(
                ckpt_cfg.get("cosmos3_checkpoint", "/home/anton/.cache/video-vam/cosmos3-edge")
            )
            if not (c3_path / "transformer").is_dir():
                fallback_c3 = Path.home() / ".cache/video-vam/cosmos3-edge"
                if not (fallback_c3 / "transformer").is_dir():
                    from huggingface_hub import snapshot_download

                    print(
                        "[SETUP] Downloading nvidia/Cosmos3-Edge backbone to ~/.cache/video-vam/cosmos3-edge..."
                    )
                    fallback_c3.mkdir(parents=True, exist_ok=True)
                    snapshot_download(
                        "nvidia/Cosmos3-Edge",
                        local_dir=str(fallback_c3),
                        allow_patterns=[
                            "transformer/*",
                            "vae/*",
                            "text_tokenizer/*",
                            "scheduler/*",
                            "*.json",
                            "*.jinja",
                        ],
                    )


def _validate_health(health: dict[str, Any], args: argparse.Namespace) -> None:
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


def _load_in_process_app(args: argparse.Namespace) -> Any:
    rpc_path = Path(__file__).with_name("rpc_server.py")
    spec = importlib.util.spec_from_file_location("video_vam_rpc_server_inproc", rpc_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load rpc_server.py from {rpc_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("video_vam_rpc_server_inproc", module)
    spec.loader.exec_module(module)

    args.resolved_checkpoint = _resolve_local_checkpoint(args.resolved_checkpoint)
    server_ns = argparse.Namespace(
        policy=args.policy,
        checkpoint=Path(args.resolved_checkpoint),
        device=args.device,
        compile=args.compile,
        rtc=(args.inference_type == "rtc"),
        execution_horizon=args.execution_horizon,
        max_guidance_weight=args.max_guidance_weight,
        joint_limits_min=args.joint_limits_min,
        joint_limits_max=args.joint_limits_max,
        euler_steps=args.euler_steps,
        instance_id="in-process",
        jepa_world_model=None,
        jepa_planning_mode="none",
        jepa_num_candidates=8,
        expo_checkpoint=getattr(args, "expo_checkpoint", None),
        shared_detector_checkpoint=getattr(args, "shared_detector_checkpoint", None),
    )
    return module.RPCApplication(server_ns)


def main(argv: list[str] | None = None) -> int:
    global SSH_HOST, REPO
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--policy", choices=("smolvla", "video_vam"), default="video_vam", help="Model policy"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="cosmos3_lora",
        help="Model alias ('cosmos3_lora', 'cosmos2b_t2_distilled', 'smolvla_v2') or path (default 'cosmos3_lora')",
    )
    parser.add_argument(
        "--expo-checkpoint",
        type=str,
        default=None,
        help="Optional path to an ExpoResidualCritic (.pt) checkpoint on the inference host.",
    )
    parser.add_argument(
        "--shared-detector-checkpoint",
        type=str,
        default=None,
        help="Optional path to a SharedFeatureSequenceDetector (.pt) checkpoint for automated online reward detection.",
    )
    parser.add_argument(
        "--online-expo-episodes",
        type=int,
        default=None,
        help="Number of physical episodes to run in an automated online Score-Centered EXPO post-training session.",
    )
    parser.add_argument(
        "--online-update-interval",
        type=int,
        default=5,
        help="Update ExpoResidualCritic weights every K episodes during online post-training (default 5).",
    )
    parser.add_argument(
        "--online-update-steps",
        type=int,
        default=600,
        help="Gradient steps per online ExpoResidualCritic update (default 600).",
    )
    parser.add_argument(
        "--spawn-group",
        type=int,
        default=0,
        help="Integer spawn group ID for Dr. GRPO per-group advantage centering A_{g,i} = R_{g,i} - mean(R_g).",
    )
    parser.add_argument(
        "--remote-host",
        default=SSH_HOST,
        help="SSH host alias for the GPU server (default: runpod-vam or $VAM_SSH_HOST)",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"RPC port on remote host (default {DEFAULT_PORT})"
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
    parser.add_argument(
        "--home-pose",
        type=float,
        nargs=6,
        default=None,
        help="Optional 6-DoF home pose (deg/0-100) to move to before Episode 1 and on every reset.",
    )
    parser.add_argument("--max-guidance-weight", type=float, default=2.0)
    parser.add_argument(
        "--keep-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="leave server running in tmux after exit for fast subsequent runs (default True)",
    )
    parser.add_argument("--stop-server", action="store_true", help="stop server even if already running")
    parser.add_argument(
        "--dry-run", action="store_true", help="test RPC connection without connecting motors"
    )
    parser.add_argument("--robot.type", dest="robot_type", default="so101_follower")
    parser.add_argument(
        "--robot.port", dest="robot_port", default=None, help=f"Serial port (default: {DEFAULT_PORT_PATH})"
    )
    parser.add_argument(
        "--robot.id",
        dest="robot_id",
        default=DEFAULT_ROBOT_ID,
        help=f"Robot ID (default: {DEFAULT_ROBOT_ID})",
    )
    parser.add_argument("--robot.use_degrees", dest="robot_use_degrees", type=_bool_arg, default=True)
    parser.add_argument("--robot.cameras", dest="robot_cameras_raw", default=DEFAULT_CAMERAS)
    parser.add_argument("--robot.max_relative_target", dest="robot_max_relative_target_raw", default=None)
    parser.add_argument("--fps", type=int, default=10, help="Action execution rate in Hz (10 or 30)")
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=None,
        help="Camera observation history sampling rate in Hz (defaults to --fps; e.g. 30 with --fps 10)",
    )
    parser.add_argument(
        "--duration", type=float, default=45.0, help="Rollout duration per episode (default 45s)"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=-1,
        help="Number of episodes (-1 for continuous loop until Ctrl+C, default -1)",
    )
    parser.add_argument(
        "--reset-time",
        type=float,
        default=6.0,
        help="Pause in seconds between episodes to reset the scene (default 6.0)",
    )
    parser.add_argument(
        "--home-duration",
        type=float,
        default=3.5,
        help="Duration in seconds for smooth transition back to base position (default 3.5)",
    )
    parser.add_argument("--task", default="take cube out of box")
    parser.add_argument("--strategy.type", dest="strategy_type", default="base")
    parser.add_argument(
        "--no-rpc",
        dest="no_rpc",
        action="store_true",
        default=False,
        help="Load and run policy directly in-process on local device without starting or querying an RPC server (overrides --remote-host).",
    )
    parser.add_argument(
        "--rpc",
        dest="no_rpc",
        action="store_false",
        help="Use RPC server for policy inference (default).",
    )
    parser.add_argument(
        "--inference.type",
        dest="inference_type",
        choices=("sync", "rtc"),
        default="rtc",
        help="Inference mode: 'rtc' (Real-Time Chunking, default) or 'sync'",
    )
    parser.add_argument(
        "--rtc",
        dest="inference_type",
        action="store_const",
        const="rtc",
        help="Enable Real-Time Chunking (alias for --inference.type=rtc)",
    )
    parser.add_argument(
        "--no-rtc",
        dest="inference_type",
        action="store_const",
        const="sync",
        help="Disable Real-Time Chunking (alias for --inference.type=sync)",
    )
    parser.add_argument(
        "--inference.rtc.execution_horizon",
        "--execution-horizon",
        dest="execution_horizon",
        type=int,
        default=10,
        help="RTC leftover prefix length and blending window (default 10).",
    )
    parser.add_argument(
        "--prefetch-steps",
        type=int,
        default=10,
        help="Action-queue threshold to trigger background prefetch (default 10)",
    )
    parser.add_argument("--feature-seed", type=int, default=0)
    parser.add_argument(
        "--euler-steps",
        type=int,
        default=None,
        help="Optional flow-matching Euler step override for video_vam server.",
    )
    args = parser.parse_args(argv)
    SSH_HOST = args.remote_host
    if _is_local_host(SSH_HOST):
        REPO = str(Path(__file__).resolve().parents[2])
    if args.fps not in (10, 30):
        parser.error("--fps must be 10 or 30 Hz")
    if args.camera_fps is not None and (
        args.camera_fps not in (10, 30) or args.camera_fps < args.fps or args.camera_fps % args.fps != 0
    ):
        parser.error("--camera-fps must be 10 or 30 Hz and an integer multiple of --fps")
    if not 1 <= args.port <= 65535 or args.execution_horizon < 1 or args.prefetch_steps < 1:
        parser.error("invalid port, execution horizon, or prefetch threshold")
    if any(
        not math.isfinite(x) or x <= 0
        for x in (args.duration, args.ready_timeout, args.rpc_timeout, args.max_guidance_weight)
    ):
        parser.error("duration/timeouts/guidance must be finite and positive")
    if args.feature_seed < 0:
        parser.error("feature seed must be nonnegative")
    if args.euler_steps is not None and args.euler_steps < 1:
        parser.error("--euler-steps must be >= 1")
    if args.strategy_type != "base":
        parser.error("only strategy.type=base is supported")
    if (args.joint_limits_min is None) != (args.joint_limits_max is None):
        parser.error("joint limits must be supplied together")
    if args.joint_limits_min is not None and (
        any(not math.isfinite(x) for x in args.joint_limits_min + args.joint_limits_max)
        or any(a >= b for a, b in zip(args.joint_limits_min, args.joint_limits_max, strict=True))
    ):
        parser.error("joint limits must be finite and ordered")
    if not args.dry_run and args.joint_limits_min is None:
        parser.error("hardware requires explicit six-dimensional absolute joint limits")

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
        "cosmos3_aug_v1": (
            "/home/anton/lerobot-video-vam/outputs/train/v1-cosmos3-edge-lora-online-aug-smolexpert",
            "video_vam",
        ),
        "cosmos3_aug_v2": (
            "/home/anton/lerobot-video-vam/outputs/train/v2-cosmos3-edge-lora-online-aug-smolexpert",
            "video_vam",
        ),
        "cosmos2b_aug_v2": (
            "/home/anton/lerobot-video-vam/outputs/train/v2-cosmos2b-t2-online-aug-smolexpert",
            "video_vam",
        ),
        "flux2_aug_v2": (
            "/home/anton/lerobot-video-vam/outputs/train/v2-flux2-klein-online-aug-smolexpert",
            "video_vam",
        ),
        "c3_sort_cubes_fast30m": (
            "/home/anton/lerobot-video-vam/outputs/train/c3_sort_cubes_fast30m",
            "video_vam",
        ),
        "c2b_t2_sort_cubes_fast30m": (
            "/home/anton/lerobot-video-vam/outputs/train/c2b_t2_sort_cubes_fast30m",
            "video_vam",
        ),
        "c3_sort_cubes_bs64": (
            "/home/anton/lerobot-video-vam/outputs/train/sort-cubes-cosmos3-edge-optimized-bs64",
            "video_vam",
        ),
        "c3peft_sort_cubes": (
            "/home/anton/lerobot-video-vam/outputs/train/c3peft_sort_cubes",
            "video_vam",
        ),
        "smolvla_sort_cubes_200k": (
            "/home/anton/lerobot-video-vam/outputs/train/sort-cubes-smolvla-200k-compiled/checkpoints/190000/pretrained_model",
            "smolvla",
        ),
        "smolvla_sort_cubes_100k": (
            "/home/anton/lerobot-video-vam/outputs/train/sort-cubes-smolvla-100k-exact/checkpoints/last/pretrained_model",
            "smolvla",
        ),
    }

    if args.checkpoint in aliases:
        args.resolved_checkpoint, args.policy = aliases[args.checkpoint]
    elif args.checkpoint in {"latest_scale100", "scale100"}:
        args.resolved_checkpoint, args.policy = aliases["smolvla_v2"]
    elif args.checkpoint is not None:
        args.resolved_checkpoint = args.checkpoint
    else:
        args.resolved_checkpoint, args.policy = aliases["cosmos3_lora"]

    # Cosmos 2B LoRA is now merged prior to torch.compile, so compile is supported

    if not args.dry_run:
        if not args.robot_port:
            import glob

            if Path(DEFAULT_PORT_PATH).exists():
                args.robot_port = DEFAULT_PORT_PATH
            else:
                detected = sorted(glob.glob("/dev/tty.usbmodem*") + glob.glob("/dev/ttyACM*"))
                if detected:
                    args.robot_port = detected[0]
                else:
                    raise SystemExit(
                        "Hardware rollout requires a connected follower robot (/dev/tty.usbmodem*)"
                    )
        if not args.robot_id:
            args.robot_id = DEFAULT_ROBOT_ID
        if args.robot_max_relative_target_raw:
            args.robot_max_relative_target = _parse_max_relative_target(args.robot_max_relative_target_raw)
        else:
            args.robot_max_relative_target = dict(DEFAULT_MAX_RELATIVE_TARGET)
        args.robot_cameras = _parse_cameras(args.robot_cameras_raw)

    if args.no_rpc:
        args.resolved_checkpoint = _resolve_local_checkpoint(args.resolved_checkpoint)
        _ensure_cosmos3_backbone(args.resolved_checkpoint, args.policy)
        app = _load_in_process_app(args)
        try:
            health = app.health()
            _validate_health(health, args)
            client = VideoVAMRPCClient(
                policy=args.policy,
                port=args.port,
                ssh_host="local",
                timeout=args.rpc_timeout,
                health=health,
                fps=args.camera_fps or args.fps,
                in_process_app=app,
            )
            if args.dry_run:
                _run_dry(args, client)
            else:
                _run_rollout(args, client)
        finally:
            closer = getattr(app, "close", None)
            if closer is not None:
                closer()
        return 0

    token = None
    owned = False
    succeeded = False
    try:
        identity = _describe_server(args)
        # Pin a mutable `last` symlink before starting a process.
        args.resolved_checkpoint = identity["checkpoint"]
        if _is_local_host(SSH_HOST):
            _ensure_cosmos3_backbone(args.resolved_checkpoint, args.policy)
        health = _healthz_payload(args.port)
        if health is None or should_restart_rpc(health_payload=health, requested_identity=identity):
            token = uuid.uuid4().hex
            owned = True
            _manage(args, "start", token)
            health = _wait_ready(args, identity, token)
        else:
            token = health["instance_id"]
        if not _is_local_host(SSH_HOST):
            import urllib.request

            def _local_forward_alive() -> bool:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/healthz", timeout=1.5) as r:
                        return r.status == 200
                except Exception:
                    return False

            if not _local_forward_alive():
                subprocess.run(
                    [
                        _ssh_executable(),
                        *SSH_OPTIONS,
                        "-O",
                        "forward",
                        "-L",
                        f"{args.port}:127.0.0.1:{args.port}",
                        SSH_HOST,
                    ],
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            if not _local_forward_alive():
                subprocess.run(
                    ["pkill", "-f", f"ssh.*-L {args.port}:127.0.0.1:{args.port}"],
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                subprocess.run(
                    [
                        _ssh_executable(),
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "ConnectTimeout=10",
                        "-o",
                        "ControlMaster=no",
                        "-o",
                        "ControlPath=none",
                        "-o",
                        "ExitOnForwardFailure=yes",
                        "-o",
                        "ServerAliveInterval=15",
                        "-fN",
                        "-L",
                        f"{args.port}:127.0.0.1:{args.port}",
                        SSH_HOST,
                    ],
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
        _validate_health(health, args)
        client = VideoVAMRPCClient(
            policy=args.policy,
            port=args.port,
            ssh_host=SSH_HOST,
            timeout=args.rpc_timeout,
            health=health,
            fps=args.camera_fps or args.fps,
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
