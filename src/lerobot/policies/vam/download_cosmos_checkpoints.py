"""Safely plan or execute pinned Cosmos-Predict2 checkpoint downloads."""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

REPOSITORY_ID = "jonpai/mimic-video"
REVISION = "f28339034831e3c2374be075e622e1ff38ebe0f8"
DEFAULT_OUTPUT_DIR = Path("/home/anton/.cache/video-vam/mimic-video-f2833903")
BRIDGE_PATTERN = "video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt"
T5_PATTERN = "text_encoder/t5-11b/*"
SMALL_FILES_ALLOWANCE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class Artifact:
    pattern: str
    relative_path: str
    size: int
    sha256: str


ARTIFACTS = {
    "generic": Artifact(
        "video_backbone/v2w_pretrained_cosmos.pt",
        "video_backbone/v2w_pretrained_cosmos.pt",
        3913017214,
        "36a994e5a7d55ccf7ca513534f78cc92faa12c8f0e29b0232bedcf22848a55ad",
    ),
    "bridge": Artifact(
        BRIDGE_PATTERN,
        BRIDGE_PATTERN,
        3913047915,
        "2723645edb158661af6d8e1c623ec21a63626d96e959799abb0b758397ac8c88",
    ),
    "tokenizer": Artifact(
        "video_backbone/tokenizer/*",
        "video_backbone/tokenizer/tokenizer.pth",
        507609880,
        "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
    ),
    "t5": Artifact(
        T5_PATTERN,
        "text_encoder/t5-11b/pytorch_model.bin",
        45229452544,
        "5fdc64177b14b0f72437fea171a752c626733f67755842528c75b90cacd1c807",
    ),
}


@dataclass(frozen=True, slots=True)
class CosmosCheckpointDownloadPlan:
    repo_id: str
    revision: str
    allow_patterns: tuple[str, ...]
    artifacts: tuple[Artifact, ...]

    def huggingface_kwargs(self, output_dir: Path) -> dict[str, object]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "allow_patterns": list(self.allow_patterns),
            "local_dir": str(output_dir),
            "max_workers": 1,
        }

    @property
    def known_bytes(self) -> int:
        return sum(artifact.size for artifact in self.artifacts)

    @property
    def required_bytes_with_buffer(self) -> int:
        # Small configs/metadata are covered in addition to the known LFS files.
        return math.ceil((self.known_bytes + SMALL_FILES_ALLOWANCE_BYTES) * 1.2)


def build_plan(
    *, include_bridge_lora: bool = False, include_t5: bool = False
) -> CosmosCheckpointDownloadPlan:
    patterns = [ARTIFACTS["generic"].pattern, ARTIFACTS["tokenizer"].pattern]
    artifacts = [ARTIFACTS["generic"], ARTIFACTS["tokenizer"]]
    if include_bridge_lora:
        patterns.append(BRIDGE_PATTERN)
        artifacts.append(ARTIFACTS["bridge"])
    if include_t5:
        patterns.append(T5_PATTERN)
        artifacts.append(ARTIFACTS["t5"])
    return CosmosCheckpointDownloadPlan(REPOSITORY_ID, REVISION, tuple(patterns), tuple(artifacts))


def _existing_parent(path: Path) -> Path:
    path = path.expanduser().absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


class DiskUsageResult(Protocol):
    free: int


def check_free_space(
    output_dir: Path,
    plan: CosmosCheckpointDownloadPlan,
    *,
    disk_usage: Callable[[Path], DiskUsageResult] = shutil.disk_usage,
) -> int:
    destination = _existing_parent(output_dir)
    free = disk_usage(destination).free
    required = plan.required_bytes_with_buffer
    if free < required:
        raise RuntimeError(
            f"Insufficient free space: need {required:,} bytes "
            f"(known {plan.known_bytes:,} + {SMALL_FILES_ALLOWANCE_BYTES:,} small-file allowance, "
            f"20% buffer), have {free:,} at {destination}"
        )
    return free


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_download(output_dir: Path, plan: CosmosCheckpointDownloadPlan) -> None:
    failures: list[str] = []
    for artifact in plan.artifacts:
        path = output_dir / artifact.relative_path
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        size = path.stat().st_size
        if size != artifact.size:
            failures.append(f"size mismatch {path}: expected {artifact.size}, got {size}")
            continue
        actual = _sha256_file(path)
        if actual != artifact.sha256:
            failures.append(f"SHA256 mismatch {path}: expected {artifact.sha256}, got {actual}")
    if failures:
        raise RuntimeError("Checkpoint verification failed:\n- " + "\n- ".join(failures))


def execute_download(plan: CosmosCheckpointDownloadPlan, output_dir: Path) -> None:
    check_free_space(output_dir, plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required only with --execute") from exc
    snapshot_download(**plan.huggingface_kwargs(output_dir))
    verify_download(output_dir, plan)


def delete_t5(
    output_dir: Path,
    *,
    execute: bool,
    expected_sha256: str | None,
    prompt_embedding_verified: bool,
) -> None:
    t5 = ARTIFACTS["t5"]
    if expected_sha256 != t5.sha256:
        raise ValueError("--expected-sha256 must exactly match the pinned T5 SHA256")
    if not prompt_embedding_verified:
        raise ValueError("Pass --prompt-embedding-verified after independently verifying the embedding")
    path = output_dir / t5.relative_path
    if not path.is_file():
        raise FileNotFoundError(f"T5 artifact not found: {path}")
    if path.stat().st_size != t5.size or _sha256_file(path) != t5.sha256:
        raise RuntimeError("Refusing deletion: T5 size or SHA256 mismatch")
    if not execute:
        print(f"Dry-run: would delete only {path}")
        return
    path.unlink()
    print(f"Deleted only {path}; tokenizer/config files were preserved")


def _print_plan(plan: CosmosCheckpointDownloadPlan, output_dir: Path) -> None:
    print(f"repo_id={plan.repo_id}")
    print(f"revision={plan.revision}")
    print(f"output_dir={output_dir}")
    print(f"known_bytes={plan.known_bytes}")
    print(f"required_bytes_with_20_percent_buffer={plan.required_bytes_with_buffer}")
    print("allow_patterns=")
    for pattern in plan.allow_patterns:
        print(f"  - {pattern}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("download", "delete-t5"), default="download")
    parser.add_argument("--include-bridge-lora", action="store_true")
    parser.add_argument("--include-t5", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true", help="Required for network or deletion mutation")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--prompt-embedding-verified", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser()

    if args.command == "delete-t5":
        delete_t5(
            output,
            execute=args.execute,
            expected_sha256=args.expected_sha256,
            prompt_embedding_verified=args.prompt_embedding_verified,
        )
        return 0

    plan = build_plan(include_bridge_lora=args.include_bridge_lora, include_t5=args.include_t5)
    _print_plan(plan, output)
    if not args.execute:
        print("Dry-run only; pass --execute to download and verify.")
        return 0
    execute_download(plan, output)
    print("Download and verification complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
