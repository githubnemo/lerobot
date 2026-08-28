#!/usr/bin/env python3
"""Mac-side Video-VAM rollout: SSH RPC on abakus, execute action chunks on the SO-101.

This is a real hardware rollout, not a dry print of chunks. Flags match
``lerobot-rollout`` where they apply. Cosmos/LTX stay on abakus; the Mac owns
USB, cameras, and ``send_action``. Each launch ``git pull --ff-only``s
``/home/anton/lerobot-video-vam`` on abakus and restarts tmux when HEAD moved
or the running server is missing RTC.

Example (same shape as lerobot-rollout):

    python run_mac_vam_rpc.py \\
        --robot.type=so101_follower \\
        --robot.port=/dev/tty.usbmodemXXXX \\
        --robot.id=so101 \\
        --robot.use_degrees=true \\
        --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \\
        --inference.type=rtc --prefetch-steps=15 \\
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
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
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
DEFAULT_CAMERAS = "{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
DEFAULT_N_ACTION_STEPS = 30
# Per-tick clip in motor units (degrees for the arm, 0-100 for the gripper).
# 10 was clamping almost every Video-VAM step; 30/50 lets the chunk through.
DEFAULT_MAX_RELATIVE_TARGET = {
    "shoulder_pan": 30.0,
    "shoulder_lift": 30.0,
    "elbow_flex": 30.0,
    "wrist_flex": 30.0,
    "wrist_roll": 30.0,
    "gripper": 50.0,
}


class VideoVAMRPCClient:
    def __init__(self, *, policy: str, port: int = DEFAULT_PORT, ssh_host: str = SSH_HOST) -> None:
        if policy not in {"smolvla", "video_vam"}:
            raise ValueError("policy must be smolvla or video_vam")
        self.policy = policy
        self.port = port
        self.ssh_host = ssh_host
        self.frames: deque[str] = deque(maxlen=5)
        self.frame_times: deque[float] = deque(maxlen=5)
        self._lock = threading.Lock()

    @staticmethod
    def _png_b64(frame: Any) -> str:
        from PIL import Image

        if not isinstance(frame, Image.Image):
            raise TypeError("frame must be a PIL Image")
        stream = BytesIO()
        frame.convert("RGB").save(stream, format="PNG")
        return base64.b64encode(stream.getvalue()).decode("ascii")

    def remember(self, frame: Any) -> None:
        with self._lock:
            self.frames.append(self._png_b64(frame))
            self.frame_times.append(time.perf_counter())

    def history_spacing_ms(self) -> list[float]:
        with self._lock:
            times = list(self.frame_times)
        return [(later - earlier) * 1000.0 for earlier, later in zip(times, times[1:], strict=False)]

    def request(
        self,
        frame: Any,
        state: list[float],
        *,
        feature_seed: int | None = None,
        remember: bool = True,
        inference_delay: int | None = None,
        prev_chunk_left_over: list[list[float]] | None = None,
        execution_horizon: int | None = None,
    ) -> dict[str, Any]:
        if tuple(frame.size) != IMAGE_SIZE:
            raise ValueError(f"frame must be 640x480, got {frame.size}")
        if len(state) != 6:
            raise ValueError("state must contain six floats")
        with self._lock:
            if remember:
                self.frames.append(self._png_b64(frame))
                self.frame_times.append(time.perf_counter())
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
        if prev_chunk_left_over is not None:
            payload["prev_chunk_left_over"] = prev_chunk_left_over
            payload["inference_delay"] = 0 if inference_delay is None else int(inference_delay)
        if execution_horizon is not None:
            payload["execution_horizon"] = int(execution_horizon)
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


def _healthz_payload(port: int) -> dict[str, Any] | None:
    completed = _ssh(["curl", "-sS", f"http://127.0.0.1:{port}/healthz"], check=False)
    if completed.returncode:
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


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


def should_restart_rpc(
    *,
    updated: bool,
    healthy: bool,
    require_rtc: bool,
    rtc_advertised: bool,
) -> bool:
    """Restart after a pull when the process would otherwise keep stale code."""
    if not healthy or updated:
        return True
    return require_rtc and not rtc_advertised


def _sync_remote_repo() -> bool:
    """Fast-forward the abakus checkout. Returns True if HEAD moved."""
    before = _ssh(["git", "-C", REPO, "rev-parse", "HEAD"]).stdout.decode().strip()
    print(f"git pull --ff-only in {SSH_HOST}:{REPO}", flush=True)
    completed = _ssh(["bash", "-lc", f"git -C {REPO} pull --ff-only"])
    output = (completed.stdout.decode(errors="replace") + completed.stderr.decode(errors="replace")).strip()
    if output:
        print(output, flush=True)
    after = _ssh(["git", "-C", REPO, "rev-parse", "HEAD"]).stdout.decode().strip()
    if before != after:
        print(f"abakus repo updated {before[:12]} -> {after[:12]}", flush=True)
        return True
    print(f"abakus repo already at {after[:12]}", flush=True)
    return False


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
        lambda match: (
            match.group(0) if match.group(0) in {"true", "false", "null"} else json.dumps(match.group(0))
        ),
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
            warmup_s=3,
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


def _actions_from_result(result: dict[str, Any], n_action_steps: int) -> list[list[float]]:
    chunk = result.get("action_chunk")
    if not isinstance(chunk, list) or not chunk:
        raise RuntimeError(f"RPC returned no action_chunk: {result}")
    if n_action_steps > len(chunk):
        print(
            f"WARNING: --n-action-steps={n_action_steps} exceeds predicted chunk "
            f"({len(chunk)}); executing the full chunk.",
            flush=True,
        )
    return [list(step) for step in chunk[:n_action_steps]]


def planned_inference_delay(max_latency_s: float | None, fps: int) -> int:
    if not max_latency_s:
        return 0
    return max(0, math.ceil(max_latency_s * fps))


def leftover_payload(leftover: Any, execution_horizon: int) -> list[list[float]] | None:
    """Pad/truncate leftover original actions to the RTC prefix length."""
    import torch

    if leftover is None:
        return None
    steps = leftover if torch.is_tensor(leftover) else torch.as_tensor(leftover, dtype=torch.float32)
    if steps.ndim != 2 or steps.shape[-1] != 6:
        raise ValueError(f"leftover must have shape [T, 6], got {tuple(steps.shape)}")
    if steps.shape[0] == 0:
        return None
    if steps.shape[0] > execution_horizon:
        steps = steps[:execution_horizon]
    elif steps.shape[0] < execution_horizon:
        padded = torch.zeros((execution_horizon, steps.shape[1]), dtype=steps.dtype)
        padded[: steps.shape[0]] = steps
        steps = padded
    return [[float(value) for value in row] for row in steps.tolist()]


def _run_rollout_rtc(args: argparse.Namespace, client: VideoVAMRPCClient) -> None:
    import torch

    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.latency_tracker import LatencyTracker

    robot = _make_robot(args)
    period = 1.0 / args.fps
    rtc_config = RTCConfig(enabled=True, execution_horizon=args.execution_horizon)
    queue = ActionQueue(rtc_config)
    latency_tracker = LatencyTracker()
    prefetch_future: Future[dict[str, Any]] | None = None
    prefetch_meta: dict[str, Any] | None = None
    print(
        f"connecting {args.robot_type} port={args.robot_port} id={args.robot_id} "
        f"task={args.task!r} duration={args.duration}s fps={args.fps} "
        f"inference=rtc execution_horizon={args.execution_horizon} "
        f"queue_threshold={args.prefetch_steps} "
        f"max_relative_target={args.robot_max_relative_target}",
        flush=True,
    )
    print(
        "RTC: leftover prefix + inference delay, then ActionQueue.merge "
        "(replaces the queue; does not append a second chunk). "
        "Training contract: 10 Hz, 5 RGB frames at offsets [-4,-3,-2,-1,0].",
        flush=True,
    )
    print(
        "WARNING: abakus RPC joint limits are placeholders (-180..180 / gripper 0..100). "
        "Local max_relative_target is the Mac-side motion clip.",
        flush=True,
    )
    robot.connect()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vam-rpc")
    try:
        deadline = time.perf_counter() + args.duration
        next_tick = time.perf_counter()
        chunks = 0

        def merge_result(result: dict[str, Any], meta: dict[str, Any], *, reset_clock: bool) -> None:
            nonlocal chunks, next_tick
            if result.get("need_more_frames"):
                raise RuntimeError("RTC RPC returned a warmup response")
            steps = _actions_from_result(result, DEFAULT_N_ACTION_STEPS)
            original = torch.tensor(steps, dtype=torch.float32)
            wall_s = time.perf_counter() - meta["started"]
            # Blocking infers happen with an empty queue: the arm was not executing,
            # so skip 0 of the new chunk. Overlapping prefetch skips the steps that ran.
            real_delay = 0 if reset_clock else planned_inference_delay(wall_s, args.fps)
            queue.merge(original, original.clone(), real_delay, meta["idx_before"], task=args.task)
            latency_tracker.add(wall_s)
            chunks += 1
            print(
                f"chunk {chunks}: merged {len(steps)} actions delay={real_delay} "
                f"(planned {meta['delay']}) queue={queue.qsize()} "
                f"rpc_latency={result.get('latency_s', '?')}s wall={wall_s:.2f}s",
                flush=True,
            )
            if reset_clock:
                # First blocking infer consumed ~1s. Play the merged chunk at 10 Hz.
                next_tick = time.perf_counter()

        def take_prefetch() -> None:
            nonlocal prefetch_future, prefetch_meta
            if prefetch_future is None or prefetch_meta is None:
                return
            result = prefetch_future.result()
            meta = prefetch_meta
            prefetch_future = None
            prefetch_meta = None
            merge_result(result, meta, reset_clock=False)

        def start_prefetch(frame: Any, state: list[float]) -> None:
            nonlocal prefetch_future, prefetch_meta
            leftover = leftover_payload(queue.get_left_over(), args.execution_horizon)
            delay = planned_inference_delay(latency_tracker.max(), args.fps)
            prefetch_meta = {
                "idx_before": queue.get_action_index(),
                "started": time.perf_counter(),
                "delay": delay,
            }
            prefetch_future = executor.submit(
                client.request,
                frame,
                state,
                feature_seed=args.feature_seed,
                remember=False,
                inference_delay=delay,
                prev_chunk_left_over=leftover,
                execution_horizon=args.execution_horizon,
            )
            print(
                f"RTC infer in flight (queue={queue.qsize()} leftover="
                f"{0 if leftover is None else len(leftover)} delay={delay})",
                flush=True,
            )

        while time.perf_counter() < deadline:
            obs = robot.get_observation()
            frame = _observation_to_pil(obs, camera_key="front")
            state = _observation_state(obs)
            client.remember(frame)

            if prefetch_future is not None and prefetch_future.done():
                take_prefetch()

            if queue.empty():
                if prefetch_future is not None:
                    print("WARNING: RTC queue emptied while inference was in flight", flush=True)
                    take_prefetch()
                else:
                    started = time.perf_counter()
                    leftover = leftover_payload(queue.get_left_over(), args.execution_horizon)
                    delay = planned_inference_delay(latency_tracker.max(), args.fps)
                    result = client.request(
                        frame,
                        state,
                        feature_seed=args.feature_seed,
                        remember=False,
                        inference_delay=delay,
                        prev_chunk_left_over=leftover,
                        execution_horizon=args.execution_horizon,
                    )
                    if result.get("need_more_frames"):
                        print(
                            f"warming up 5-frame history, need {result['need_more_frames']} more",
                            flush=True,
                        )
                        next_tick += period
                        _sleep_until(min(next_tick, deadline))
                        continue
                    spacing = client.history_spacing_ms()
                    print(
                        f"history_dt_ms={[round(dt) for dt in spacing]}",
                        flush=True,
                    )
                    merge_result(
                        result,
                        {
                            "idx_before": queue.get_action_index(),
                            "started": started,
                            "delay": delay,
                        },
                        reset_clock=True,
                    )

            if prefetch_future is None and not queue.empty() and queue.qsize() <= args.prefetch_steps:
                start_prefetch(frame, state)

            action_t = queue.get()
            if action_t is None:
                raise RuntimeError("RTC queue empty after merge")
            sent = robot.send_action(_chunk_to_action(action_t.tolist()))
            print(
                "sent " + " ".join(f"{name}={sent[name]:.2f}" for name in ACTION_NAMES if name in sent),
                flush=True,
            )
            next_tick += period
            _sleep_until(min(next_tick, deadline))
        print(f"duration reached after {chunks} chunk(s)", flush=True)
    finally:
        if prefetch_future is not None:
            prefetch_future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        robot.disconnect()


def _run_rollout(args: argparse.Namespace, client: VideoVAMRPCClient) -> None:
    if args.inference_type == "rtc":
        _run_rollout_rtc(args, client)
        return
    robot = _make_robot(args)
    pending: list[list[float]] = []
    period = 1.0 / args.fps
    prefetch_future: Future[dict[str, Any]] | None = None
    print(
        f"connecting {args.robot_type} port={args.robot_port} id={args.robot_id} "
        f"task={args.task!r} duration={args.duration}s fps={args.fps} "
        f"n_action_steps={args.n_action_steps} prefetch_steps={args.prefetch_steps} "
        f"max_relative_target={args.robot_max_relative_target}",
        flush=True,
    )
    print(
        "Training contract: 10 Hz, 5 RGB frames at offsets [-4,-3,-2,-1,0] "
        "(100 ms apart), then execute 30 actions over 3.0 s. "
        "After a blocking RPC we do not catch up — the chunk plays at real 10 Hz.",
        flush=True,
    )
    print(
        "WARNING: abakus RPC joint limits are placeholders (-180..180 / gripper 0..100). "
        "Local max_relative_target is the Mac-side motion clip.",
        flush=True,
    )
    robot.connect()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vam-rpc")
    try:
        deadline = time.perf_counter() + args.duration
        next_tick = time.perf_counter()
        chunks = 0

        def take_prefetch() -> None:
            nonlocal prefetch_future, pending, chunks
            if prefetch_future is None:
                return
            result = prefetch_future.result()
            prefetch_future = None
            if result.get("need_more_frames"):
                raise RuntimeError("prefetch RPC returned a warmup response")
            steps = _actions_from_result(result, args.n_action_steps)
            pending.extend(steps)
            chunks += 1
            print(
                f"chunk {chunks}: +{len(steps)} actions queued ({len(pending)} pending), "
                f"rpc_latency={result.get('latency_s', '?')}s",
                flush=True,
            )

        while time.perf_counter() < deadline:
            obs = robot.get_observation()
            frame = _observation_to_pil(obs, camera_key="front")
            state = _observation_state(obs)
            client.remember(frame)

            if prefetch_future is not None and prefetch_future.done():
                take_prefetch()

            if not pending:
                if prefetch_future is not None:
                    take_prefetch()
                else:
                    result = client.request(frame, state, feature_seed=args.feature_seed, remember=False)
                    if result.get("need_more_frames"):
                        print(
                            f"warming up 5-frame history, need {result['need_more_frames']} more",
                            flush=True,
                        )
                        next_tick += period
                        _sleep_until(min(next_tick, deadline))
                        continue
                    pending.extend(_actions_from_result(result, args.n_action_steps))
                    chunks += 1
                    spacing = client.history_spacing_ms()
                    print(
                        f"chunk {chunks}: {len(pending)} actions, "
                        f"rpc_latency={result.get('latency_s', '?')}s, "
                        f"history_dt_ms={[round(dt) for dt in spacing]}",
                        flush=True,
                    )
                    # Inference just consumed ~1s. Do not dump the 30-step chunk
                    # as fast as possible to "catch up" — that would run motors
                    # and the next 5-frame window much faster than training's 10 Hz.
                    next_tick = time.perf_counter()

            if (
                args.prefetch_steps > 0
                and prefetch_future is None
                and 0 < len(pending) <= args.prefetch_steps
            ):
                prefetch_future = executor.submit(
                    client.request,
                    frame,
                    state,
                    feature_seed=args.feature_seed,
                    remember=False,
                )
                print(f"prefetching next chunk ({len(pending)} actions left)", flush=True)

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
        if prefetch_future is not None:
            prefetch_future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
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
        help=(
            "per-motor clip dict. Default is 30 deg / gripper 50 "
            "(was 10/20, which clamped almost every Video-VAM step)"
        ),
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--task", default="take cube out of box")
    parser.add_argument("--strategy.type", dest="strategy_type", default="base")
    parser.add_argument(
        "--inference.type",
        dest="inference_type",
        choices=("sync", "rtc"),
        default="sync",
        help="sync = open-loop chunks; rtc = leftover+delay merge (needs restarted RPC server)",
    )
    parser.add_argument(
        "--inference.rtc.execution_horizon",
        dest="execution_horizon",
        type=int,
        default=None,
        help="RTC leftover prefix length (default 10). Distinct from --n-action-steps.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help=(
            "sync: how many steps of each predicted chunk to execute "
            f"(default {DEFAULT_N_ACTION_STEPS}; the model predicts 30). Ignored for rtc."
        ),
    )
    parser.add_argument(
        "--prefetch-steps",
        type=int,
        default=None,
        help=(
            "sync: start the next RPC when this many executed steps remain "
            "(default 0 = no overlap). rtc: action-queue threshold to start overlapping "
            "inference (default 15)"
        ),
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
        args.execution_horizon = 10 if args.execution_horizon is None else args.execution_horizon
        if args.execution_horizon < 1:
            raise SystemExit("--inference.rtc.execution_horizon must be >= 1")
        if args.n_action_steps is not None and args.n_action_steps != DEFAULT_N_ACTION_STEPS:
            print(
                "NOTE: --n-action-steps is ignored with --inference.type=rtc "
                "(RTC merges the full 30-step chunk after skipping inference delay).",
                flush=True,
            )
        args.n_action_steps = DEFAULT_N_ACTION_STEPS
        if args.prefetch_steps is None:
            args.prefetch_steps = 15
        if args.prefetch_steps < 1:
            raise SystemExit("RTC requires --prefetch-steps >= 1 (action-queue threshold)")
    else:
        args.n_action_steps = args.n_action_steps or args.execution_horizon or DEFAULT_N_ACTION_STEPS
        if args.n_action_steps < 1:
            raise SystemExit("--n-action-steps must be >= 1")
        if args.prefetch_steps is None:
            args.prefetch_steps = 0
        if args.prefetch_steps < 0:
            raise SystemExit("--prefetch-steps must be >= 0")

    if not args.dry_run:
        if not args.robot_port:
            raise SystemExit("hardware rollout requires --robot.port")
        if not args.robot_id:
            raise SystemExit("hardware rollout requires --robot.id")
        if args.robot_max_relative_target_raw:
            args.robot_max_relative_target = _parse_max_relative_target(args.robot_max_relative_target_raw)
        else:
            args.robot_max_relative_target = dict(DEFAULT_MAX_RELATIVE_TARGET)
        args.robot_cameras = _parse_cameras(args.robot_cameras_raw)

    started_here = False
    client = VideoVAMRPCClient(policy=args.policy, port=args.port)
    try:
        updated = _sync_remote_repo()
        healthy = _healthz(args.port)
        health = _healthz_payload(args.port) if healthy else None
        rtc_advertised = bool(health and health.get("rtc") is True)
        if should_restart_rpc(
            updated=updated,
            healthy=healthy,
            require_rtc=args.inference_type == "rtc",
            rtc_advertised=rtc_advertised,
        ):
            if _session_exists() or healthy:
                print("restarting RPC server so it loads the pulled code", flush=True)
                _stop_server()
                drop_by = time.time() + 15.0
                while time.time() < drop_by and (_healthz(args.port) or _session_exists()):
                    time.sleep(0.5)
            print(f"starting RPC server in tmux session {TMUX_SESSION}", flush=True)
            _start_server()
            started_here = True
            _wait_ready(args.port, READY_TIMEOUT_S)
            print("RPC ready", flush=True)
        else:
            print(f"RPC already healthy on {SSH_HOST}:{args.port}", flush=True)
        if args.inference_type == "rtc":
            health = _healthz_payload(args.port)
            if not health or health.get("rtc") is not True:
                raise SystemExit(
                    "The abakus RPC server still does not advertise RTC after git pull. "
                    "Push this branch first, then rerun."
                )
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
