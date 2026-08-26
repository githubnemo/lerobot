#!/usr/bin/env python3
"""Write small LeRobot config.json files beside custom Video-VAM expert runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.policies.vam.configuration_video_vam import VideoVAMConfig

DEFAULT_COSMOS_RUN = Path(
    "/home/anton/.cache/video-vam/runs/cosmos-scaling-20260825/smolexpert-prefix-retrain"
)
DEFAULT_LTX_RUN = Path("/home/anton/.cache/video-vam/runs/ltx25-smolexpert-20260826/run-a-pool2")


def _write(run_dir: Path, backend: str, *, overwrite: bool) -> None:
    target = run_dir / "config.json"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass --overwrite to replace it")
    if not (run_dir / "best.safetensors").is_file() or not (run_dir / "normalizer.safetensors").is_file():
        raise FileNotFoundError(f"{run_dir} is not a complete SmolExpert run")
    config = VideoVAMConfig(backend=backend, device="cuda")
    config._save_pretrained(run_dir)
    print(f"wrote {target} ({backend}; hardware limits intentionally unset)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos-run", type=Path, default=DEFAULT_COSMOS_RUN)
    parser.add_argument("--ltx-run", type=Path, default=DEFAULT_LTX_RUN)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    _write(args.cosmos_run, "cosmos", overwrite=args.overwrite)
    _write(args.ltx_run, "ltx", overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
