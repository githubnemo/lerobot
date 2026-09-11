#!/usr/bin/env python3
"""Minimal HTTP RPC server for SmolVLA or Video-VAM on abakus."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from PIL import Image

HOST = "127.0.0.1"
PORT = 8766
HISTORY = 5


def _frame(value: Any) -> torch.Tensor:
    if isinstance(value, str):
        try:
            raw = base64.b64decode(value, validate=True)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            return (
                torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
                .reshape(image.height, image.width, 3)
                .clone()
            )
        except Exception as exc:
            raise ValueError(f"invalid PNG base64 frame: {exc}") from exc
    if isinstance(value, list):
        if len(value) == HISTORY and all(isinstance(item, str) for item in value):
            return torch.stack([_frame(item) for item in value])
        tensor = torch.tensor(value, dtype=torch.uint8)
        if tensor.ndim == 3:
            return tensor
        if tensor.ndim == 4 and tensor.shape[0] == HISTORY:
            return tensor
    raise ValueError("frame must be PNG base64 or an [H,W,3] uint8 list")


def _validate_frame(frame: torch.Tensor) -> torch.Tensor:
    if frame.ndim != 3 or tuple(frame.shape[-1:]) != (3,):
        raise ValueError(f"frame must have shape [H,W,3], got {tuple(frame.shape)}")
    if tuple(frame.shape[:2]) != (480, 640):
        raise ValueError(f"frame must have shape [480,640,3], got {tuple(frame.shape)}")
    return frame.contiguous()


PROTOCOL_VERSION = 2
MAX_REQUEST_BYTES = 24 * 1024 * 1024


def _integer(value: Any, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def validate_chunk(value: Any, *, horizon: int, name: str = "action_chunk") -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tuple(tensor.shape) != (horizon, 6) or not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must be finite [{horizon}, 6]")
    return tensor


def checkpoint_identity(args: argparse.Namespace) -> dict[str, Any]:
    """Content-address local checkpoint artifacts; never infer identity from a missing config."""
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict) or config.get("type") != args.policy:
        raise ValueError("checkpoint config.type does not match requested policy")
    digest = hashlib.sha256()
    files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError("checkpoint has no safetensors weights")
    for path in files:
        digest.update(str(path.relative_to(checkpoint)).encode())
        before = path.stat()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("checkpoint changed while fingerprinting")
    # A working-tree server update must also invalidate a running instance.
    code = hashlib.sha256()
    for filename in ("rpc_server.py", "run_rpc_server.sh"):
        code.update(Path(__file__).with_name(filename).read_bytes())
    return {
        "protocol": PROTOCOL_VERSION,
        "policy": args.policy,
        "checkpoint": str(checkpoint),
        "fingerprint": digest.hexdigest(),
        "server_code": code.hexdigest(),
        "config": {
            "device": args.device,
            "compile": args.compile,
            "rtc": args.rtc,
            "execution_horizon": args.execution_horizon,
            "max_guidance_weight": args.max_guidance_weight,
            "joint_limits_min": args.joint_limits_min,
            "joint_limits_max": args.joint_limits_max,
        },
    }


def rtc_kwargs_from_payload(payload: dict[str, Any], *, horizon: int = 30) -> dict[str, Any]:
    leftover = payload.get("prev_chunk_left_over")
    delay = _integer(payload.get("inference_delay", 0), "inference_delay", 0, horizon)
    if leftover is None:
        if delay:
            raise ValueError("inference_delay requires prev_chunk_left_over")
        return {}
    tensor = torch.tensor(leftover, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[-1] != 6 or not 1 <= tensor.shape[0] <= horizon:
        raise ValueError(f"prev_chunk_left_over must have shape [T, 6], 1 <= T <= {horizon}")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("prev_chunk_left_over must be finite")
    if delay > tensor.shape[0]:
        raise ValueError("inference delay exceeds available prefix")
    return {"prev_chunk_left_over": tensor, "inference_delay": delay}


class RPCApplication:
    def __init__(self, args: argparse.Namespace) -> None:
        self.name = args.policy
        self.identity = checkpoint_identity(args)
        self.checkpoint_path = self.identity["checkpoint"]
        args.checkpoint = Path(self.checkpoint_path)
        self.instance_id = args.instance_id
        self.predict_lock = threading.Lock()
        self.rtc_enabled = args.rtc
        self.execution_horizon = args.execution_horizon
        self.lower = args.joint_limits_min
        self.upper = args.joint_limits_max
        self.device = torch.device(args.device)
        self.frame_index = 0
        if args.policy == "smolvla":
            from lerobot.policies.vam.action_rmse import SmolVLABackend

            self.backend = SmolVLABackend.from_pretrained(str(args.checkpoint), device=self.device)
            self.policy = self.backend.policy
            if self.policy.config.adapt_to_pi_aloha or self.policy.config.use_delta_joint_actions_aloha:
                raise ValueError("SO follower RPC does not support Aloha action transforms")
            if self.policy.config.action_feature.shape != (6,) and list(
                self.policy.config.action_feature.shape
            ) != [6]:
                raise ValueError("SO follower RPC requires six action dimensions")
            self.horizon = self.policy.config.chunk_size
        else:
            from lerobot.policies.vam.configuration_video_vam import VideoVAMConfig
            from lerobot.policies.vam.modeling_video_vam import HISTORY_FRAME_INDEX_KEY, VideoVAMPolicy

            config = VideoVAMConfig.from_pretrained(args.checkpoint)
            config.device = str(self.device)
            config.cosmos_torch_compile = args.compile
            if args.compile:
                config.cosmos_compile_friendly = True
            if args.joint_limits_min is not None:
                config.joint_limits_min = args.joint_limits_min
                config.joint_limits_max = args.joint_limits_max
            elif config.joint_limits_min is None or config.joint_limits_max is None:
                config.joint_limits_min = [-180.0, -180.0, -180.0, -180.0, -180.0, 0.0]
                config.joint_limits_max = [180.0, 180.0, 180.0, 180.0, 180.0, 100.0]
            self.backend = VideoVAMPolicy.from_pretrained(args.checkpoint, config=config)
            self.policy = self.backend
            self.horizon = 30
            self.lower, self.upper = config.joint_limits_min, config.joint_limits_max
            self.history_key = f"{config.camera_key}.history"
            self.history_index_key = HISTORY_FRAME_INDEX_KEY
            self.config = config
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        if not 1 <= self.execution_horizon <= self.horizon:
            raise ValueError("execution horizon exceeds policy chunk size")
        self.original_space = "normalized" if self.name == "smolvla" else "physical"
        self.policy.config.rtc_config = RTCConfig(
            enabled=args.rtc,
            execution_horizon=args.execution_horizon,
            max_guidance_weight=args.max_guidance_weight,
        )
        self.policy.init_rtc_processor()
        # Refuse a checkpoint changed during model loading.
        if checkpoint_identity(args) != self.identity:
            raise ValueError("checkpoint changed during model loading")

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "identity": self.identity,
            "instance_id": self.instance_id,
            "policy": self.name,
            "checkpoint": self.checkpoint_path,
            "rtc": self.rtc_enabled,
            "horizon": self.horizon,
            "original_space": self.original_space,
            "action_dim": 6,
            "history_size": HISTORY if self.name == "video_vam" else 1,
            "fps": 10,
            "action_units": ["degrees"] * 5 + ["range_0_100"],
        }

    @torch.no_grad()
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("identity") != self.identity or payload.get("instance_id") != self.instance_id:
            raise ValueError("server identity/instance mismatch")
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError("request_id is required")
        state = torch.tensor(payload.get("state"), dtype=torch.float32)
        if tuple(state.shape) != (6,) or not torch.isfinite(state).all().item():
            raise ValueError("state must contain six finite floats")
        frame_index = _integer(payload.get("frame_index", 0), "frame_index", 0, 2**63 - 1)
        kwargs = rtc_kwargs_from_payload(payload, horizon=self.horizon)
        if kwargs and not self.rtc_enabled:
            raise ValueError("RTC inputs require RTC-enabled server")
        if kwargs and payload.get("original_space") != self.original_space:
            raise ValueError("RTC original action space mismatch")
        if payload.get("execution_horizon", self.execution_horizon) != self.execution_horizon:
            raise ValueError("execution_horizon differs from server configuration")
        supplied_seed = payload.get("feature_seed")
        if supplied_seed is not None and (not isinstance(supplied_seed, int) or supplied_seed < 0):
            raise ValueError("feature_seed must be a non-negative integer")
        value = payload.get(
            "images_front_u8_png_b64", payload.get("images_front_u8_raw", payload.get("images_front_u8"))
        )
        if value is None:
            raise ValueError("missing images_front_u8_png_b64 or images_front_u8_raw")
        started = time.perf_counter()
        if self.name == "smolvla":
            frame = _validate_frame(_frame(value))
            observation = {
                "observation.images.front": frame.permute(2, 0, 1).float().div(255.0),
                "observation.state": state,
                "task": str(payload.get("task", "take cube out of box")),
            }
            processed = self.backend.preprocessor(observation)
            device, dtype = self.backend._device_and_dtype()
            processed = {
                key: value.to(device=device) if isinstance(value, torch.Tensor) else value
                for key, value in processed.items()
            }
            if "prev_chunk_left_over" in kwargs:
                kwargs["prev_chunk_left_over"] = kwargs["prev_chunk_left_over"].to(device=device, dtype=dtype)
            generator = torch.Generator(device=device).manual_seed(supplied_seed or 0)
            noise = torch.randn(
                (1, self.horizon, self.policy.config.max_action_dim),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            # Each RPC supplies a complete observation, independent of other clients.
            self.policy.reset()
            original = self.policy.predict_action_chunk(processed, noise=noise, **kwargs)
            actions = self.backend.postprocessor(original.clone())[0].cpu()
            original = original[0].cpu()
        else:
            frames = _frame(value)
            if frames.ndim != 4 or frames.shape[0] != HISTORY:
                raise ValueError("video_vam requires five frames in images_front_u8_png_b64 or raw list")
            frames = torch.stack([_validate_frame(frame) for frame in frames])
            batch = {
                self.history_key: frames.permute(3, 0, 1, 2).unsqueeze(0).float().div(255.0),
                "observation.state": state.unsqueeze(0),
                self.history_index_key: torch.tensor([frame_index], dtype=torch.long),
            }
            if supplied_seed is not None:
                kwargs["feature_noise_seed"] = supplied_seed
            if "prev_chunk_left_over" in kwargs and self.backend.rtc_processor is None:
                raise RuntimeError("RTC leftover received but rtc_processor is not initialized")
            actions = self.backend.predict_action_chunk(batch, **kwargs)[0].cpu()
            original = actions.clone()
        actions = validate_chunk(actions, horizon=self.horizon)
        original = validate_chunk(original, horizon=self.horizon, name="original_chunk")
        if (
            self.lower is not None
            and ((actions < torch.tensor(self.lower)) | (actions > torch.tensor(self.upper))).any().item()
        ):
            raise ValueError("predicted physical action exceeds joint limits")
        if ((actions[:, 5] < 0) | (actions[:, 5] > 100)).any().item():
            raise ValueError("predicted gripper action outside [0, 100]")
        self.frame_index += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            "action_chunk": actions.float().tolist(),
            "original_chunk": original.float().tolist(),
            "original_space": self.original_space,
            "identity": self.identity,
            "instance_id": self.instance_id,
            "request_id": request_id,
            "policy": self.name,
            "latency_s": time.perf_counter() - started,
        }

    def close(self) -> None:
        closer = getattr(self.backend, "close", None)
        if closer is not None:
            closer()


class Handler(BaseHTTPRequestHandler):
    app: RPCApplication

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10.0)

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self._send(404, {"error": "not found"})
            return
        self._send(200, self.app.health())

    def do_POST(self) -> None:
        if self.path != "/predict":
            self._send(404, {"error": "use POST /predict"})
            return
        try:
            self.connection.settimeout(10.0)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            # Health remains responsive, but never queue concurrent GPU predictions.
            if not self.app.predict_lock.acquire(blocking=False):
                self._send(409, {"error": "inference already in progress"})
                return
            try:
                result = self.app.predict(payload)
            finally:
                self.app.predict_lock.release()
            self._send(200, result)
        except Exception as exc:
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        try:
            self.end_headers()
            self.wfile.write(encoded)
        except (TimeoutError, BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: Any) -> None:
        print(format % args, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="Print identity without loading a model")
    parser.add_argument("--instance-id", default="unmanaged")
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy", choices=("smolvla", "video_vam"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--joint-limits-min", type=float, nargs=6)
    parser.add_argument("--joint-limits-max", type=float, nargs=6)
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=10,
        help="RTC leftover prefix length (overridable per request)",
    )
    parser.add_argument("--max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile the Cosmos DiT with max-autotune (default). Pass --no-compile for eager.",
    )
    args = parser.parse_args(argv)
    if args.execution_horizon < 1:
        parser.error("--execution-horizon must be >= 1")
    if not math.isfinite(args.max_guidance_weight) or args.max_guidance_weight <= 0:
        parser.error("--max-guidance-weight must be positive")
    if (args.joint_limits_min is None) != (args.joint_limits_max is None):
        parser.error("joint limits must be supplied together")
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    if args.joint_limits_min is not None and (
        any(not math.isfinite(x) for x in args.joint_limits_min + args.joint_limits_max)
        or any(low >= high for low, high in zip(args.joint_limits_min, args.joint_limits_max, strict=True))
    ):
        parser.error("joint limits must be finite and ordered")
    if args.describe:
        print(json.dumps(checkpoint_identity(args), sort_keys=True))
        return 0
    app = RPCApplication(args)
    Handler.app = app
    try:
        ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
