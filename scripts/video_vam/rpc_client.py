#!/usr/bin/env python3
"""Mac-side SSH client for the Video-VAM RPC server."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image

SSH_HOST = "abakus"
DEFAULT_PORT = 8765


class VideoVAMRPCClient:
    def __init__(self, *, policy: str, port: int = DEFAULT_PORT) -> None:
        if policy not in {"smolvla", "video_vam"}:
            raise ValueError("policy must be smolvla or video_vam")
        self.policy = policy
        self.port = port
        self.frames: deque[str] = deque(maxlen=5)

    @staticmethod
    def _png_b64(frame: Image.Image) -> str:
        from io import BytesIO

        stream = BytesIO()
        frame.convert("RGB").save(stream, format="PNG")
        return base64.b64encode(stream.getvalue()).decode("ascii")

    def request(
        self, frame: Image.Image, state: list[float], *, feature_seed: int | None = None
    ) -> dict[str, Any]:
        if tuple(frame.size) != (640, 480):
            raise ValueError(f"frame must be 640x480, got {frame.size}")
        if len(state) != 6:
            raise ValueError("state must contain six floats")
        encoded = self._png_b64(frame)
        self.frames.append(encoded)
        if self.policy == "video_vam" and len(self.frames) < 5:
            return {"need_more_frames": 5 - len(self.frames), "policy": self.policy}
        value: str | list[str] = encoded if self.policy == "smolvla" else list(self.frames)
        payload: dict[str, Any] = {
            "state": [float(value) for value in state],
            "images_front_u8_png_b64": value,
        }
        if feature_seed is not None:
            if feature_seed < 0:
                raise ValueError("feature_seed must be non-negative")
            payload["feature_seed"] = feature_seed
        command = [
            "ssh",
            SSH_HOST,
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
            command, input=json.dumps(payload).encode(), stdout=subprocess.PIPE, check=False
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.decode(errors="replace") or "SSH RPC request failed")
        result = json.loads(completed.stdout)
        if not isinstance(result, dict):
            raise RuntimeError("RPC response must be a JSON object")
        return result


def _dry_frame() -> Image.Image:
    return Image.new("RGB", (640, 480), (0, 0, 0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("smolvla", "video_vam"), required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--state", type=float, nargs=6, default=[0.0] * 6)
    parser.add_argument("--feature-seed", type=int)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dry-json", action="store_true", help="send a fake 480x640x3 zero frame")
    args = parser.parse_args(argv)
    if bool(args.image) == args.dry_json:
        parser.error("choose exactly one of --image or --dry-json")
    frame = _dry_frame() if args.dry_json else Image.open(args.image).convert("RGB")
    result = VideoVAMRPCClient(policy=args.policy, port=args.port).request(
        frame, args.state, feature_seed=args.feature_seed
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
