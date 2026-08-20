#!/usr/bin/env python3
"""Extract one LTX-2.5 representation and report the real resource contract."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lerobot.policies.vam.ltx_extractor import (
    LTXExtractor,
    LTXExtractorConfig,
    derive_window_seed,
)

DEFAULT_DATASET_ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_CHECKPOINT = Path(
    "/home/anton/.cache/video-vam/ltx-2.5-models/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
)
DEFAULT_VIDEO_VAE = Path(
    "/home/anton/.cache/video-vam/ltx-2.5-models/vae/ltx-2.5-video-vae-conv-bf16.safetensors"
)


class SmokeEncoder(nn.Module):
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return torch.zeros((video.shape[0], 128, 2, 15, 20), device=video.device, dtype=video.dtype)


class SmokeBackbone(nn.Module):
    def forward(self, *, latent: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.full(
            (latent.shape[0], 2400, 4096),
            0.5,
            device=latent.device,
            dtype=latent.dtype,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-index", type=int)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--video-vae", type=Path, default=DEFAULT_VIDEO_VAE)
    parser.add_argument("--prompt-embedding", type=Path, required=False)
    parser.add_argument("--dataset-revision", default="243370c3c08bcbd860133c4a0d658ea7c1d2e77e")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument(
        "--fake", action="store_true", help="Run the injected contract backend instead of LTX weights."
    )
    parser.add_argument("--allow-zero-prompt", action="store_true")
    return parser.parse_args(argv)


def load_prompt_embedding(path: Path | None, *, batch_size: int, device: torch.device) -> torch.Tensor:
    if path is None:
        return torch.zeros((batch_size, 1, 4096), dtype=torch.bfloat16, device=device)
    from safetensors.torch import load_file

    values = load_file(str(path), device="cpu")
    if "embedding" in values:
        embedding = values["embedding"]
    elif len(values) == 1:
        embedding = next(iter(values.values()))
    else:
        raise ValueError(f"Prompt artifact {path} must contain an 'embedding' tensor")
    if embedding.ndim == 2:
        embedding = embedding.unsqueeze(0)
    if embedding.ndim != 3 or embedding.shape[-1] != 4096 or embedding.shape[0] != batch_size:
        raise ValueError(
            f"Prompt embedding must have shape [B, sequence, 4096], got {tuple(embedding.shape)}"
        )
    return embedding.to(device=device, dtype=torch.bfloat16)


def _sample(args: argparse.Namespace) -> tuple[Any, torch.Tensor, int]:
    from scripts.video_vam.smoke_test_cosmos_extractor import load_real_sample

    dataset, prepared = load_real_sample(
        root=args.root,
        episode_index=args.episode,
        frame_index=args.frame_index,
    )
    return dataset, prepared.rgb_history, prepared.frame_index


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("The LTX smoke requires CUDA")
    device = torch.device("cuda")
    dataset, images, frame_index = _sample(args)
    prompt_embedding = load_prompt_embedding(args.prompt_embedding, batch_size=images.shape[0], device=device)
    if args.prompt_embedding is None and not args.allow_zero_prompt and not args.fake:
        raise ValueError(
            "Provide --prompt-embedding from the LTX Gemma encoder, or explicitly pass --allow-zero-prompt"
        )
    noise_seed = derive_window_seed(args.dataset_revision, args.episode, frame_index, args.global_seed)
    config = LTXExtractorConfig(
        checkpoint_path=args.checkpoint,
        video_vae_path=args.video_vae,
        device="cuda",
        dtype="bfloat16",
    )
    if args.fake:
        extractor = LTXExtractor(config, backbone=SmokeBackbone(), latent_encoder=SmokeEncoder())
    else:
        extractor = LTXExtractor(config)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    extraction = extractor.extract(images, prompt_embedding, noise_seed=noise_seed)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    print("LTX extractor smoke succeeded")
    print(f"  dataset:             {dataset.repo_id}@{dataset.revision}")
    print(f"  episode/frame:       {args.episode}/{frame_index}")
    print(f"  window seed:         {noise_seed}")
    print(f"  raw_hidden_shape:    {tuple(extraction.hidden_grid.shape)}")
    print(f"  context_shape:       {tuple(extraction.tokens.shape)}")
    print(f"  context_dtype:       {extraction.tokens.dtype}")
    print(f"  layer/noise:          {extraction.layer}/{extraction.sigma.tolist()}")
    print(f"  extraction_seconds:   {elapsed:.3f}")
    print(f"  peak_vram_allocated:  {peak_allocated} bytes ({peak_allocated / 2**30:.2f} GiB)")
    print(f"  peak_vram_reserved:    {peak_reserved} bytes ({peak_reserved / 2**30:.2f} GiB)")
    print(f"  provenance:           {extraction.provenance.to_dict()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - concise smoke CLI failure
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
