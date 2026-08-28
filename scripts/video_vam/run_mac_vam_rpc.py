#!/usr/bin/env python3
"""Mac-side Video-VAM rollout: SSH RPC on abakus, execute action chunks on the SO-101.

This is a real hardware rollout, not a dry print of chunks. Flags match
``lerobot-rollout`` where they apply. Cosmos/LTX stay on abakus; the Mac owns
USB, cameras, and ``send_action``.

Example (same shape as lerobot-rollout):

    python run_mac_vam_rpc.py \\
        --robot.type=so101_follower \\
        --robot.port=/dev/tty.usbmodemXXXX \\
        --robot.id=so101 \\
        --robot.use_degrees=true \\
        --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 10}}' \\
        --robot.max_relative_target='{"shoulder_pan": 10, "shoulder_lift": 10, "elbow_flex": 10, "wrist_flex": 10, "wrist_roll": 10, "gripper": 20}' \\
        --fps=10 --duration=30 --task="take cube out of box" \\
        --keep-server
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import math
import re
import shutil
import subprocess
import time
from collections import deque
from io import BytesIO
from typing import Any

SSH_HOST = "abakus"


def _ssh_executable() -> str:
    path = shutil.which("ssh")
    if path is None:
        raise RuntimeError("ssh not found on PATH")
    return path


DEFAULT_PORT = 8765
TMUX_SESSION = "video-vam-rpc"
REPO = "/home/anton/lerobot-video-vam"
SERVER_WRAPPER = "scripts/video_vam/run_rpc_server.sh"
READY_TIMEOUT_S = 180.0
IMAGE_SIZE = (640, 480)
ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
MOTOR_NAMES = tuple(name.removesuffix(".pos") for name in ACTION_NAMES)
DEFAULT_CAMERAS = "{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 10}}"


class VideoVAMRPCClient:
    def __init__(self, *, policy: str, port: int = DEFAULT_PORT, ssh_host: str = SSH_HOST) -> None:
        if policy not in {"smolvla", "video_vam"}:
            raise ValueError("policy must be smolvla or video_vam")
        self.policy = policy
        self.port = port
        self.ssh_host = ssh_host
        self.frames: deque[str] = deque(maxlen=5)

    @staticmethod
    def _png_b64(frame: Any) -> str:
        from PIL import Image

        if not isinstance(frame, Image.Image):
            raise TypeError("frame must be a PIL Image")
        stream = BytesIO()
        frame.convert("RGB").save(stream, format="PNG")
        return base64.b64encode(stream.getvalue()).decode("ascii")

    def remember(self, frame: Any) -> None:
        self.frames.append(self._png_b64(frame))

    def request(self, frame: Any, state: list[float], *, feature_seed: int | None = None) -> dict[str, Any]:
        if tuple(frame.size) != IMAGE_SIZE:
            raise ValueError(f"frame must be 640x480, got {frame.size}")
        if len(state) != 6:
            raise ValueError("state must contain six floats")
        self.remember(frame)
        if self.policy == "video_vam" and len(self.frames) < 5:
            return {"need_more_frames": 5 - len(self.frames), "policy": self.policy}
        value: str | list[str] = encoded_latest(self) if self.policy == "smolvla" else list(self.frames)
        payload: dict[str, Any] = {
            "state": [float(x) for x in state],
            "images_front_u8_png_b64": value,
        }
        if feature_seed is not None:
            if feature_seed < 0:
                raise ValueError("feature_seed must be non-negative")
            payload["feature_seed"] = feature_seed
        command = [
            _ssh_executable(),
            self.ssh_host,
            "curl",
            "--fail-with-body",
            "-sS",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
            f"http://127.0.0.1:{self.port}/predict",
        ]
        completed = subprocess.run(
            command,
            input=json.dumps(payload).encode(),
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            err = completed.stderr.decode(errors="replace") or completed.stdout.decode(errors="replace")
            raise RuntimeError(err or "SSH RPC request failed")
        result = json.loads(completed.stdout)
        if not isinstance(result, dict):
            raise RuntimeError("RPC response must be a JSON object")
        if "error" in result:
            raise RuntimeError(result["error"])
        return result


def encoded_latest(client: VideoVAMRPCClient) -> str:
    if not client.frames:
        raise RuntimeError("no frames buffered")
    return client.frames[-1]


def _ssh(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run([_ssh_executable(), SSH_HOST, *args], capture_output=True, check=False)
    if check and completed.returncode:
        raise RuntimeError(
            completed.stderr.decode(errors="replace")
            or completed.stdout.decode(errors="replace")
            or "ssh failed"
        )
    return completed


def _healthz(port: int) -> bool:
    completed = _ssh(
        ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", f"http://127.0.0.1:{port}/healthz"],
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == b"200"


def _session_exists() -> bool:
    completed = _ssh(["tmux", "has-session", "-t", TMUX_SESSION], check=False)
    return completed.returncode == 0


def _start_server() -> None:
    remote = (
        f"cd {REPO} && tmux has-session -t {TMUX_SESSION} 2>/dev/null || "
        f"tmux new-session -d -s {TMUX_SESSION} 'bash {SERVER_WRAPPER}'"
    )
    _ssh(["bash", "-lc", remote])


def _stop_server() -> None:
    _ssh(["tmux", "kill-session", "-t", TMUX_SESSION], check=False)


def _wait_ready(port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _healthz(port):
            return
        time.sleep(2.0)
    raise TimeoutError(f"RPC server on {SSH_HOST}:{port} did not become ready within {timeout_s:.0f}s")


def _bool_arg(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def _parse_mapping(raw: str) -> Any:
    text = raw.strip()
    if not text:
        raise ValueError("empty mapping")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    quoted = re.sub(
        r"[A-Za-z_][A-Za-z0-9_]*",
        lambda match: match.group(0)
        if match.group(0) in {"true", "false", "null"}
        else json.dumps(match.group(0)),
        text,
    )
    try:
        return json.loads(quoted)
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse mapping {raw!r}") from exc


def _parse_max_relative_target(raw: str) -> dict[str, float]:
    value = _parse_mapping(raw)
    if not isinstance(value, dict):
        raise ValueError(
            "robot.max_relative_target must be a per-motor dict "
            f"(not a scalar). Expected keys: {list(MOTOR_NAMES)}"
        )
    missing = [name for name in MOTOR_NAMES if name not in value]
    if missing:
        raise ValueError(f"robot.max_relative_target is missing motors: {missing}")
    parsed = {name: float(value[name]) for name in MOTOR_NAMES}
    if any(not math.isfinite(limit) or limit <= 0 for limit in parsed.values()):
        raise ValueError("robot.max_relative_target values must be finite and positive")
    return parsed


def _parse_cameras(raw: str) -> dict[str, dict[str, Any]]:
    value = _parse_mapping(raw)
    if not isinstance(value, dict) or not value:
        raise ValueError("robot.cameras must be a non-empty mapping")
    cameras: dict[str, dict[str, Any]] = {}
    for name, spec in value.items():
        if not isinstance(spec, dict):
            raise ValueError(f"camera {name!r} must be a mapping")
        cam_type = spec.get("type")
        if cam_type != "opencv":
            raise ValueError(f"only opencv cameras are supported, got {cam_type!r} for {name}")
        if "index_or_path" not in spec:
            raise ValueError(f"camera {name!r} needs index_or_path")
        width = int(spec.get("width", IMAGE_SIZE[0]))
        height = int(spec.get("height", IMAGE_SIZE[1]))
        fps = int(spec.get("fps", 10))
        if (width, height) != IMAGE_SIZE:
            raise ValueError(f"camera {name!r} must be 640x480, got {width}x{height}")
        cameras[str(name)] = {
            "type": "opencv",
            "index_or_path": spec["index_or_path"],
            "width": width,
            "height": height,
            "fps": fps,
        }
    if "front" not in cameras:
        raise ValueError("robot.cameras must include a 'front' camera")
    return cameras


def _observation_to_pil(obs: dict[str, Any], camera_key: str = "front"):
    import numpy as np
    from PIL import Image

    if camera_key not in obs:
        raise KeyError(f"observation is missing camera {camera_key!r}; keys={sorted(obs)}")
    arr = obs[camera_key]
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        max_value = float(arr.max()) if arr.size else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    image = Image.fromarray(arr).convert("RGB")
    if image.size != IMAGE_SIZE:
        image = image.resize(IMAGE_SIZE)
    return image


def _observation_state(obs: dict[str, Any]) -> list[float]:
    missing = [name for name in ACTION_NAMES if name not in obs]
    if missing:
        raise KeyError(f"observation is missing joints {missing}; keys={sorted(obs)}")
    return [float(obs[name]) for name in ACTION_NAMES]


def _chunk_to_action(step: list[float]) -> dict[str, float]:
    if len(step) != 6:
        raise ValueError(f"action step must have 6 values, got {len(step)}")
    return {name: float(value) for name, value in zip(ACTION_NAMES, step, strict=True)}


def _make_robot(args: argparse.Namespace):
    try:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.so_follower import SO101FollowerConfig
    except ImportError as exc:
        raise SystemExit(
            "This rollout needs the same Python env as lerobot-rollout "
            f"(import lerobot failed: {exc}). Activate that env and retry."
        ) from exc

    if args.robot_type not in {"so101_follower", "so100_follower"}:
        raise ValueError(f"unsupported robot.type {args.robot_type!r}")
    cameras = {
        name: OpenCVCameraConfig(
            index_or_path=spec["index_or_path"],
            width=spec["width"],
            height=spec["height"],
            fps=spec["fps"],
        )
        for name, spec in args.robot_cameras.items()
    }
    config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cameras,
        use_degrees=args.robot_use_degrees,
        max_relative_target=args.robot_max_relative_target,
    )
    return make_robot_from_config(config)


def _sleep_until(deadline: float) -> None:
    remaining = deadline - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def _run_rollout(args: argparse.Namespace, client: VideoVAMRPCClient) -> None:
    robot = _make_robot(args)
    pending: list[list[float]] = []
    period = 1.0 / args.fps
    print(
        f"connecting {args.robot_type} port={args.robot_port} id={args.robot_id} "
        f"task={args.task!r} duration={args.duration}s fps={args.fps} "
        f"n_action_steps={args.n_action_steps}",
        flush=True,
    )
    print(
        "WARNING: abakus RPC joint limits are placeholders (-180..180 / gripper 0..100). "
        "Local max_relative_target is the Mac-side motion clip.",
        flush=True,
    )
    robot.connect()
    try:
        deadline = time.perf_counter() + args.duration
        next_tick = time.perf_counter()
        chunks = 0
        while time.perf_counter() < deadline:
            obs = robot.get_observation()
            frame = _observation_to_pil(obs, camera_key="front")
            state = _observation_state(obs)
            if not pending:
                result = client.request(frame, state, feature_seed=args.feature_seed)
                if result.get("need_more_frames"):
                    print(
                        f"warming up 5-frame history, need {result['need_more_frames']} more",
                        flush=True,
                    )
                    next_tick += period
                    _sleep_until(min(next_tick, deadline))
                    continue
                chunk = result.get("action_chunk")
                if not isinstance(chunk, list) or not chunk:
                    raise RuntimeError(f"RPC returned no action_chunk: {result}")
                pending = [list(step) for step in chunk[: args.n_action_steps]]
                chunks += 1
                print(
                    f"chunk {chunks}: {len(pending)} actions, rpc_latency={result.get('latency_s', '?')}s",
                    flush=True,
                )
            else:
                client.remember(frame)
            action = pending.pop(0)
            sent = robot.send_action(_chunk_to_action(action))
            print(
                "sent " + " ".join(f"{name}={sent[name]:.2f}" for name in ACTION_NAMES if name in sent),
                flush=True,
            )
            next_tick += period
            _sleep_until(min(next_tick, deadline))
        print(f"duration reached after {chunks} chunk(s)", flush=True)
    finally:
        robot.disconnect()


def _dry_frame():
    from PIL import Image

    return Image.new("RGB", IMAGE_SIZE, (0, 0, 0))


def _run_dry(args: argparse.Namespace, client: VideoVAMRPCClient) -> None:
    n_needed = 5 if args.policy == "video_vam" else 1
    for i in range(n_needed):
        result = client.request(_dry_frame(), [0.0] * 6, feature_seed=args.feature_seed)
        print(json.dumps({"i": i, **result}, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy", choices=("smolvla", "video_vam"), default="video_vam")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="abakus RPC port")
    parser.add_argument("--keep-server", action="store_true", help="leave the abakus tmux session running")
    parser.add_argument("--stop-server", action="store_true", help="stop even if we did not start it")
    parser.add_argument("--dry-run", action="store_true", help="RPC only; do not connect the arm")
    parser.add_argument("--robot.type", dest="robot_type", default="so101_follower")
    parser.add_argument("--robot.port", dest="robot_port", default=None)
    parser.add_argument("--robot.id", dest="robot_id", default=None)
    parser.add_argument("--robot.use_degrees", dest="robot_use_degrees", type=_bool_arg, default=True)
    parser.add_argument("--robot.cameras", dest="robot_cameras_raw", default=DEFAULT_CAMERAS)
    parser.add_argument(
        "--robot.max_relative_target",
        dest="robot_max_relative_target_raw",
        default=None,
        help="required per-motor dict in physical units",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--task", default="take cube out of box")
    parser.add_argument("--strategy.type", dest="strategy_type", default="base")
    parser.add_argument("--inference.type", dest="inference_type", choices=("sync", "rtc"), default="sync")
    parser.add_argument(
        "--inference.rtc.execution_horizon",
        dest="execution_horizon",
        type=int,
        default=None,
        help="steps to execute from each 30-step chunk before replanning (rtc default 10)",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="steps to execute from each chunk before replanning (default 30 sync / 10 rtc)",
    )
    parser.add_argument("--feature-seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.fps != 10:
        raise SystemExit("Video-VAM rollout requires --fps=10")
    if not args.robot_use_degrees:
        raise SystemExit("Video-VAM SO-101 rollout requires --robot.use_degrees=true")
    if args.strategy_type != "base":
        raise SystemExit("this launcher only implements --strategy.type=base")
    if args.inference_type == "rtc":
        print(
            "NOTE: --inference.type=rtc here means execute execution_horizon steps then replan. "
            "It is not overlapping RTC guidance.",
            flush=True,
        )

    default_horizon = 10 if args.inference_type == "rtc" else 30
    args.n_action_steps = args.n_action_steps or args.execution_horizon or default_horizon
    if args.n_action_steps < 1:
        raise SystemExit("--n-action-steps must be >= 1")

    if not args.dry_run:
        if not args.robot_port:
            raise SystemExit("hardware rollout requires --robot.port")
        if not args.robot_id:
            raise SystemExit("hardware rollout requires --robot.id")
        if not args.robot_max_relative_target_raw:
            raise SystemExit(
                "hardware rollout requires --robot.max_relative_target as a per-motor dict, e.g. "
                '\'{"shoulder_pan": 10, "shoulder_lift": 10, "elbow_flex": 10, '
                '"wrist_flex": 10, "wrist_roll": 10, "gripper": 20}\''
            )
        args.robot_cameras = _parse_cameras(args.robot_cameras_raw)
        args.robot_max_relative_target = _parse_max_relative_target(args.robot_max_relative_target_raw)

    started_here = False
    client = VideoVAMRPCClient(policy=args.policy, port=args.port)
    try:
        already = _healthz(args.port)
        if already:
            print(f"RPC already healthy on {SSH_HOST}:{args.port}", flush=True)
        else:
            print(f"starting RPC server in tmux session {TMUX_SESSION}", flush=True)
            _start_server()
            started_here = True
            _wait_ready(args.port, READY_TIMEOUT_S)
            print("RPC ready", flush=True)
        if args.dry_run:
            _run_dry(args, client)
        else:
            _run_rollout(args, client)
    finally:
        if args.stop_server or (started_here and not args.keep_server):
            print(f"stopping tmux session {TMUX_SESSION}", flush=True)
            _stop_server()
        elif started_here and args.keep_server:
            print(f"leaving {TMUX_SESSION} running on {SSH_HOST}", flush=True)
        elif _session_exists() and args.keep_server:
            print(f"leaving existing {TMUX_SESSION} running on {SSH_HOST}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
