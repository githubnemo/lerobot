#!/usr/bin/env python3
"""Minimal HTTP RPC server for SmolVLA or Video-VAM on abakus."""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import torch
from PIL import Image

HOST = "127.0.0.1"
PORT = 8765
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


class RPCApplication:
    def __init__(self, args: argparse.Namespace) -> None:
        self.name = args.policy
        self.device = torch.device(args.device)
        self.frame_index = 0
        if args.policy == "smolvla":
            from lerobot.policies.vam.action_rmse import SmolVLABackend

            self.backend = SmolVLABackend.from_pretrained(str(args.checkpoint), device=self.device)
        else:
            from lerobot.policies.vam.configuration_video_vam import VideoVAMConfig
            from lerobot.policies.vam.modeling_video_vam import HISTORY_FRAME_INDEX_KEY, VideoVAMPolicy

            config = VideoVAMConfig.from_pretrained(args.checkpoint)
            config.device = str(self.device)
            if args.joint_limits_min is not None:
                config.joint_limits_min = args.joint_limits_min
                config.joint_limits_max = args.joint_limits_max
            if config.joint_limits_min is None or config.joint_limits_max is None:
                raise ValueError("video_vam requires --joint-limits-min and --joint-limits-max")
            self.backend = VideoVAMPolicy.from_pretrained(args.checkpoint, config=config)
            self.history_key = f"{config.camera_key}.history"
            self.history_index_key = HISTORY_FRAME_INDEX_KEY
            self.config = config

    @torch.no_grad()
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = torch.tensor(payload.get("state"), dtype=torch.float32)
        if tuple(state.shape) != (6,):
            raise ValueError("state must contain six floats")
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
            from lerobot.policies.vam.action_rmse import EvaluationBatch

            observation = {
                "observation.images.front": frame.permute(2, 0, 1).float().div(255.0),
                "observation.state": state,
                "task": str(payload.get("task", "cube out of box")),
            }
            batch = EvaluationBatch(
                sample_ids=(f"rpc-{self.frame_index}",),
                target_actions=torch.zeros(1, 30, 6),
                action_is_pad=torch.zeros(1, 30, dtype=torch.bool),
                current_state=state.unsqueeze(0),
                observations=(observation,),
            )
            actions = self.backend.predict_actions(batch, supplied_seed or 0)[0]
        else:
            frames = _frame(value)
            if frames.ndim != 4 or frames.shape[0] != HISTORY:
                raise ValueError("video_vam requires five frames in images_front_u8_png_b64 or raw list")
            frames = torch.stack([_validate_frame(frame) for frame in frames])
            batch = {
                self.history_key: frames.permute(3, 0, 1, 2).unsqueeze(0).float().div(255.0),
                "observation.state": state.unsqueeze(0),
                self.history_index_key: torch.tensor([self.frame_index], dtype=torch.long),
            }
            kwargs = {} if supplied_seed is None else {"feature_noise_seed": supplied_seed}
            actions = self.backend.predict_action_chunk(batch, **kwargs)[0].cpu()
        self.frame_index += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            "action_chunk": actions.float().tolist(),
            "policy": self.name,
            "latency_s": time.perf_counter() - started,
        }

    def close(self) -> None:
        closer = getattr(self.backend, "close", None)
        if closer is not None:
            closer()


class Handler(BaseHTTPRequestHandler):
    app: RPCApplication

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self._send(404, {"error": "not found"})
            return
        self._send(200, {"ok": True, "policy": self.app.name})

    def do_POST(self) -> None:
        if self.path != "/predict":
            self._send(404, {"error": "use POST /predict"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            result = self.app.predict(payload)
            self._send(200, result)
        except Exception as exc:
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        print(format % args, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("smolvla", "video_vam"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--joint-limits-min", type=float, nargs=6)
    parser.add_argument("--joint-limits-max", type=float, nargs=6)
    args = parser.parse_args(argv)
    if (args.joint_limits_min is None) != (args.joint_limits_max is None):
        parser.error("joint limits must be supplied together")
    print("Operator must hold the existing GPU lock before starting this server.", flush=True)
    app = RPCApplication(args)
    Handler.app = app
    try:
        HTTPServer((args.host, args.port), Handler).serve_forever()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
