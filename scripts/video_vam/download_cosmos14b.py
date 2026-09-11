#!/usr/bin/env python3
"""Download Cosmos-1.0-Diffusion-14B weights for feature extraction.

Downloads the essential diffusers-format weights (transformer, VAE, tokenizer,
and configs) while omitting redundant NeMo distributed checkpoints and root model.pt.
Total download size is ~28.75 GB.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

PRIMARY_REPO = "Arsh9210/Cosmos-1.0-Diffusion-14B-Video2World"
OFFICIAL_NVIDIA_REPO = "nvidia/Cosmos-1.0-Diffusion-14B-Video2World"
DEFAULT_OUTPUT_DIR = Path("/home/anton/.cache/video-vam/cosmos-14b")

BASE_ALLOW_PATTERNS = [
    "transformer/*",
    "vae/*",
    "tokenizer/*",
    "scheduler/*",
    "*.json",
]

TEXT_ENCODER_PATTERNS = [
    "text_encoder/*",
]

EXPECTED_SHARDS = [
    "transformer/diffusion_pytorch_model-00001-of-00006.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00006.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00006.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00006.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00006.safetensors",
    "transformer/diffusion_pytorch_model-00006-of-00006.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Cosmos-1.0-Diffusion-14B weights for feature extraction."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=f"HuggingFace repo ID (default: {OFFICIAL_NVIDIA_REPO} if HF_TOKEN is present, else {PRIMARY_REPO}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--include-text-encoder",
        action="store_true",
        help="Also download T5 text encoder weights (+9.73 GB).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face authentication token.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent download workers (default: 4).",
    )
    return parser.parse_args()


def check_free_space(dest: Path, required_gb: float = 35.0) -> None:
    parent = dest.resolve()
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    free_bytes = shutil.disk_usage(parent).free
    free_gb = free_bytes / (1024**3)
    print(f"[Downloader] Available disk space at {parent}: {free_gb:.1f} GB")
    if free_gb < required_gb:
        raise RuntimeError(
            f"Insufficient disk space: required {required_gb:.1f} GB, available {free_gb:.1f} GB"
        )


def verify_download(output_dir: Path) -> bool:
    print("\n[Downloader] Verifying downloaded weight files...")
    all_ok = True
    total_bytes = 0
    for rel_path in EXPECTED_SHARDS:
        file_path = output_dir / rel_path
        if not file_path.is_file():
            print(f"  [MISSING] {rel_path}")
            all_ok = False
        else:
            sz = file_path.stat().st_size
            total_bytes += sz
            print(f"  [OK] {rel_path} ({sz / (1024**3):.2f} GB)")
    print(f"[Downloader] Total verified weight size: {total_bytes / (1024**3):.2f} GB")
    return all_ok


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    token = args.token

    # Determine repo ID
    repo_id = args.repo_id
    if repo_id is None:
        if token:
            repo_id = OFFICIAL_NVIDIA_REPO
        else:
            repo_id = PRIMARY_REPO

    allow_patterns = list(BASE_ALLOW_PATTERNS)
    est_size_gb = 29.0
    if args.include_text_encoder:
        allow_patterns.extend(TEXT_ENCODER_PATTERNS)
        est_size_gb += 10.0

    print("=" * 70)
    print("Cosmos-1.0-Diffusion-14B Background Downloader")
    print("=" * 70)
    print(f"Target Repo:        {repo_id}")
    print(f"Output Directory:   {output_dir}")
    print(f"Estimated Size:     ~{est_size_gb:.1f} GB")
    print(f"Include T5 Text:    {args.include_text_encoder}")
    print(f"Allow Patterns:     {allow_patterns}")
    print(f"Max Workers:        {args.max_workers}")
    print(f"HF Token Provided:  {'Yes' if token else 'No (using ungated mirror)'}")
    print("=" * 70)

    check_free_space(output_dir, required_gb=est_size_gb * 1.2)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed in the python environment.", file=sys.stderr)
        return 1

    start_time = time.time()
    try:
        print(f"\n[Downloader] Starting snapshot download from {repo_id}...")
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            local_dir=str(output_dir),
            token=token,
            max_workers=args.max_workers,
        )
        elapsed = time.time() - start_time
        print(f"\n[Downloader] Snapshot download completed in {elapsed:.1f}s ({elapsed / 60:.2f} min).")
        print(f"[Downloader] Files saved at: {downloaded_path}")
    except Exception as e:
        # If the official repo fails due to auth, attempt graceful fallback to mirror
        if repo_id == OFFICIAL_NVIDIA_REPO and not token:
            print(
                f"\n[Downloader] Failed to download from {repo_id} ({e}). Falling back to {PRIMARY_REPO}..."
            )
            repo_id = PRIMARY_REPO
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=allow_patterns,
                local_dir=str(output_dir),
                token=None,
                max_workers=args.max_workers,
            )
            elapsed = time.time() - start_time
            print(f"\n[Downloader] Fallback download completed in {elapsed:.1f}s ({elapsed / 60:.2f} min).")
        else:
            print(f"\n[Downloader] Download failed with error: {e}", file=sys.stderr)
            raise

    ok = verify_download(output_dir)
    if ok:
        print("\n[Downloader] ALL COSMOS-14B WEIGHTS DOWNLOADED AND VERIFIED SUCCESSFULLY!")
        return 0
    else:
        print("\n[Downloader] WARNING: Some shards were not verified.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
