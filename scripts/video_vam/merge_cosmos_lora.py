#!/usr/bin/env python3
"""Fuse a trained Cosmos LoRA adapter into the generic Cosmos checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from lerobot.policies.vam.cosmos_lora import (
    inject_lora,
    load_lora_state_dict,
    merge_lora_into_base,
)
from lerobot.policies.vam.cosmos_predict2_extractor import (
    CosmosPredict2Extractor,
    CosmosPredict2ExtractorConfig,
)

DEFAULT_BASE = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
)
DEFAULT_TOKENIZER = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
)
DEFAULT_LORA = Path(
    "/home/anton/.cache/video-vam/runs/cosmos2b-video-lora-20260828/paused-step6000/best_lora.safetensors"
)
DEFAULT_OUTPUT = Path("/home/anton/.cache/video-vam/runs/cosmos2b-video-lora-20260828/fused-step6000.pt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--lora-weights", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_lora_metadata(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"LoRA weights and sidecar are required: {path}")
    payload = json.loads(sidecar.read_text())
    lora = payload.get("lora")
    if not isinstance(lora, dict):
        raise ValueError(f"LoRA sidecar has no lora object: {sidecar}")
    rank = lora.get("rank")
    alpha = lora.get("alpha")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"invalid LoRA rank in {sidecar}: {rank!r}")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or float(alpha) <= 0:
        raise ValueError(f"invalid LoRA alpha in {sidecar}: {alpha!r}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "sidecar_path": str(sidecar.resolve()),
        "sidecar_sha256": sha256_file(sidecar),
        "rank": rank,
        "alpha": float(alpha),
        "step": payload.get("best_step", payload.get("step")),
        "val_video_loss": payload.get("best_val_video_loss"),
        "checkpoint_kind": payload.get("checkpoint_kind"),
    }


def git_commit() -> str | None:
    try:
        # Fixed argv; resolving git from PATH is intentional, like the rest of the repo tooling.
        result = subprocess.run(  # nosec B607
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    args = parse_args()
    base = args.base_checkpoint.expanduser().resolve()
    lora_path = args.lora_weights.expanduser().resolve()
    tokenizer = args.tokenizer.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to construct the official Cosmos backbone")
    for path in (base, tokenizer, lora_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite fused checkpoint: {output}")

    lora = load_lora_metadata(lora_path)
    if lora["checkpoint_kind"] != "best" or lora["step"] != 6000:
        raise ValueError(f"expected best step-6000 adapters, got {lora}")
    if lora["rank"] != 16 or lora["alpha"] != 16.0:
        raise ValueError(f"expected rank=16 alpha=16 adapters, got {lora}")

    print(f"loading base={base}", flush=True)
    extractor = CosmosPredict2Extractor(
        CosmosPredict2ExtractorConfig(
            checkpoint_path=base,
            tokenizer_path=tokenizer,
            device="cuda",
            dtype="bfloat16",
            hidden_layer=20,
            high_noise_sigma=80.0,
        )
    )
    backbone = extractor.backbone
    adapter_names = inject_lora(backbone, rank=int(lora["rank"]), alpha=float(lora["alpha"]))
    load_lora_state_dict(backbone, lora_path)
    merged = merge_lora_into_base(backbone)
    if tuple(merged) != tuple(adapter_names):
        raise RuntimeError("merged module list differs from injected module list")
    if any(module.__class__.__name__ == "LoRALinear" for module in backbone.modules()):
        raise RuntimeError("fused backbone still contains LoRALinear modules")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    state = {key: value.detach().cpu().contiguous() for key, value in backbone.state_dict().items()}
    print(f"saving fused checkpoint={output} tensors={len(state)}", flush=True)
    torch.save(state, temporary)
    os.replace(temporary, output)
    provenance = {
        "artifact": "cosmos_predict2_fused_checkpoint",
        "status": "fused_inference_checkpoint",
        "base_checkpoint": {
            "path": str(base),
            "sha256": sha256_file(base),
            "size_bytes": base.stat().st_size,
        },
        "lora": lora,
        "fused_checkpoint": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
        },
        "merge": {
            "formula": "W <- W + (alpha / rank) * B @ A",
            "rank": int(lora["rank"]),
            "alpha": float(lora["alpha"]),
            "step": lora["step"],
            "val_video_loss": lora["val_video_loss"],
            "merged_modules": len(merged),
            "hidden_layer": 20,
            "dtype": "bfloat16",
        },
        "compatibility": {
            "format": "flat torch state_dict",
            "load_checkpoint_strict": True,
        },
        "git_commit": git_commit(),
    }
    fused_checkpoint = provenance["fused_checkpoint"]
    if not isinstance(fused_checkpoint, dict):
        raise TypeError("fused_checkpoint provenance must be a dict")
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"fused_sha256={fused_checkpoint['sha256']}", flush=True)
    print(f"provenance={sidecar}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
