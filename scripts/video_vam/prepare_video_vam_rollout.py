#!/usr/bin/env python3
"""Write small LeRobot config.json files beside custom Video-VAM expert runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.policies.vam.configuration_video_vam import VideoVAMConfig

DEFAULT_COSMOS_RUN = Path(
    "/home/anton/.cache/video-vam/runs/cosmos-scaling-20260825/smolexpert-prefix-retrain"
)
DEFAULT_COSMOS_LORA_CHECKPOINT = Path(
    "/home/anton/.cache/video-vam/runs/cosmos2b-video-lora-20260828/fused-step6000.pt"
)
DEFAULT_LTX_RUN = Path("/home/anton/.cache/video-vam/runs/ltx25-smolexpert-20260826/run-a-pool2")
DEFAULT_LTX_UNPOOLED_RUN = Path("/home/anton/.cache/video-vam/runs/ltx25-smolexpert-20260826/run-b-unpooled")


def _write(
    run_dir: Path,
    backend: str,
    *,
    overwrite: bool,
    cosmos_checkpoint: Path | None = None,
) -> None:
    target = run_dir / "config.json"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass --overwrite to replace it")
    if not (run_dir / "best.safetensors").is_file() or not (run_dir / "normalizer.safetensors").is_file():
        raise FileNotFoundError(f"{run_dir} is not a complete SmolExpert run")
    config = VideoVAMConfig(backend=backend, device="cuda")
    if cosmos_checkpoint is not None:
        config.cosmos_checkpoint = cosmos_checkpoint
    config._save_pretrained(run_dir)
    print(f"wrote {target} ({backend}; hardware limits intentionally unset)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos-run", type=Path, default=DEFAULT_COSMOS_RUN)
    parser.add_argument("--cosmos-lora-run", type=Path)
    parser.add_argument("--ltx-run", type=Path, default=DEFAULT_LTX_RUN)
    parser.add_argument("--ltx-unpooled-run", type=Path, default=DEFAULT_LTX_UNPOOLED_RUN)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.cosmos_lora_run is not None:
        _write(
            args.cosmos_lora_run,
            "cosmos",
            overwrite=args.overwrite,
            cosmos_checkpoint=DEFAULT_COSMOS_LORA_CHECKPOINT,
        )
        return 0
    _write(args.cosmos_run, "cosmos", overwrite=args.overwrite)
    _write(args.ltx_run, "ltx", overwrite=args.overwrite)
    _write(args.ltx_unpooled_run, "ltx", overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
