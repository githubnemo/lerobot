"""Small, single-process helpers for portable weights-only run artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def default_run_dir(repo_root: Path, prefix: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / "outputs" / "train" / f"{prefix}-{stamp}-{uuid4().hex[:8]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    """Replace one file only after serialization succeeds; never follow target symlinks."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        writer(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    atomic_write(path, lambda temporary: temporary.write_text(content, encoding="utf-8"))


def code_identity(repo_root: Path) -> dict[str, Any]:
    try:

        def git(*args: str) -> str:
            return subprocess.check_output(
                ["git", "--no-pager", "--no-optional-locks", "-C", str(repo_root), *args],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            ).strip()

        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def save_weights_artifact(
    output_dir: Path,
    role: str,
    state: Mapping[str, Any],
    step: int,
    manifest: dict[str, Any],
    *,
    metric: float | None = None,
) -> None:
    """Publish weights then manifest. Hashes detect interruption between the two replaces.

    No optimizer/RNG state is saved. Each file is atomic, not the entire run bundle.
    """
    from safetensors.torch import save_file

    if role not in {"best", "last"}:
        raise ValueError("Artifact role must be best or last")
    path = output_dir / f"{role}.safetensors"
    atomic_write(
        path,
        lambda temporary: save_file(
            state, str(temporary), metadata={"role": role, "optimizer_step": str(step), "resumable": "false"}
        ),
    )
    manifest["checkpoints"][role] = {
        "path": path.name,
        "sha256": sha256_file(path),
        "optimizer_step": step,
        "metric": metric,
        "resumable": False,
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)
