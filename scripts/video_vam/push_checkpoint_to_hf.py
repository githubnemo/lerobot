#!/usr/bin/env python3
"""Upload a finished Video-VAM, FLUX 3 Action, or SmolVLA run directory to Hugging Face Hub.

Supports both:
1. `train_smolexpert.py` output dirs (`best.safetensors`, optional `best_backbone_lora.safetensors`,
   `config.json`, `normalizer.*`, `training_summary.json`, `metrics.jsonl`, `run_manifest.json`).
2. `lerobot-train` output dirs (`checkpoints/last/pretrained_model` or highest-step checkpoint).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def _load_hf_token() -> str | None:
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_FINE_GRAINED_ALL"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return val
    for candidate in (Path(".env"), Path.home() / ".env.ml"):
        if candidate.is_file():
            for line in candidate.read_text().splitlines():
                s = line.strip()
                if s.startswith("export "):
                    s = s[len("export ") :].strip()
                for env_key in ("HF_TOKEN=", "HF_TOKEN_FINE_GRAINED_ALL="):
                    if s.startswith(env_key):
                        val = s.split("=", 1)[1].strip().strip("\"'")
                        if val:
                            return val
    token_path = Path.home() / ".cache/huggingface/token"
    if token_path.is_file():
        val = token_path.read_text().strip()
        if val:
            return val
    return None


def _sanitize_repo_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return cleaned[:96]


def resolve_upload_source(run_dir: Path, *, include_last: bool = False) -> tuple[Path, list[str]]:
    """Return `(folder_to_upload, ignore_patterns)` for either VAM or lerobot-train runs."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    # 1. Direct VAM / SmolExpert run directory
    if (run_dir / "best.safetensors").is_file() or (run_dir / "last.safetensors").is_file():
        ignore = ["*.pyc", "__pycache__/*", "wandb/*"]
        if not include_last and (run_dir / "best.safetensors").is_file():
            ignore.extend(["last.safetensors", "last_backbone_lora.safetensors"])
        return run_dir, ignore

    # 2. Direct pretrained_model directory (e.g. lerobot-train checkpoint)
    if (run_dir / "config.json").is_file() and (
        (run_dir / "model.safetensors").is_file() or (run_dir / "adapter_model.safetensors").is_file()
    ):
        return run_dir, ["*.pyc", "__pycache__/*"]

    # 3. lerobot-train run dir containing checkpoints/<step|last>/pretrained_model
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        last_pm = checkpoints_dir / "last" / "pretrained_model"
        if last_pm.is_dir():
            return last_pm.resolve(), ["*.pyc", "__pycache__/*"]
        step_dirs = sorted(
            (p for p in checkpoints_dir.iterdir() if p.is_dir() and (p / "pretrained_model").is_dir()),
            key=lambda p: int(p.name) if p.name.isdigit() else -1,
        )
        if step_dirs:
            return (step_dirs[-1] / "pretrained_model").resolve(), ["*.pyc", "__pycache__/*"]

    raise FileNotFoundError(
        f"No recognizable checkpoint artifacts found in {run_dir} "
        "(expected best.safetensors or checkpoints/*/pretrained_model)"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Trained run directory to upload.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Target HF model repo ID (default: <owner>/<prefix><run_dir.name>).",
    )
    parser.add_argument(
        "--owner",
        type=str,
        default=os.environ.get("HF_UPLOAD_OWNER", "Orellius"),
        help="HF username/org when --repo-id is omitted (default: Orellius).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=os.environ.get("HF_REPO_PREFIX", "so101_"),
        help="Repo name prefix when --repo-id is omitted (default: so101_).",
    )
    parser.add_argument("--private", action="store_true", help="Create repo as private if it does not exist.")
    parser.add_argument(
        "--include-last", action="store_true", help="Also upload last*.safetensors for VAM runs."
    )
    parser.add_argument("--commit-message", type=str, default=None, help="Custom Hub commit message.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print resolved upload plan without uploading."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    upload_dir, ignore_patterns = resolve_upload_source(run_dir, include_last=args.include_last)

    repo_id = args.repo_id
    if not repo_id:
        slug = _sanitize_repo_name(run_dir.name)
        prefix = args.prefix or ""
        repo_name = slug if (prefix and slug.startswith(prefix)) else f"{prefix}{slug}"
        repo_id = f"{args.owner}/{_sanitize_repo_name(repo_name)}"

    commit_msg = args.commit_message or f"Upload trained checkpoint from {run_dir.name}"
    print(f"[hf-upload] run_dir:    {run_dir}")
    print(f"[hf-upload] upload_dir: {upload_dir}")
    print(f"[hf-upload] repo_id:    {repo_id}")
    print(f"[hf-upload] ignore:     {ignore_patterns}")

    if args.dry_run:
        print("[hf-upload] dry-run complete (no upload performed).")
        return 0

    token = _load_hf_token()
    if not token:
        print(
            "[hf-upload] ERROR: HF_TOKEN not found in env, .env, ~/.env.ml, or ~/.cache/huggingface/token.",
            file=sys.stderr,
        )
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
    commit_info = api.upload_folder(
        folder_path=str(upload_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=ignore_patterns,
        commit_message=commit_msg,
    )

    # For lerobot-train runs, also upload top-level evaluation JSONs if present alongside checkpoints/
    if upload_dir != run_dir:
        for extra_name in ("training_summary.json", "eval_metrics.json", "flux3_eval.json"):
            extra_path = run_dir / extra_name
            if extra_path.is_file():
                api.upload_file(
                    path_or_fileobj=str(extra_path),
                    path_in_repo=extra_name,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Add {extra_name} for {run_dir.name}",
                )

    record = {
        "repo_id": repo_id,
        "upload_dir": str(upload_dir),
        "commit_url": str(commit_info),
    }
    (run_dir / "hf_upload.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"[hf-upload] SUCCESS -> https://huggingface.co/{repo_id} ({commit_info})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
