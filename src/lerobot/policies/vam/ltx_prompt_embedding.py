"""Strict prompt artifact for the official LTX-2.5 Gemma-4 encoder."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

DEFAULT_PROMPT = "take cube out of box"
EXPECTED_SHAPE = (1, 1024, 4096)
EXPECTED_DTYPE = torch.bfloat16
LTX_REPOSITORY = "https://huggingface.co/Lightricks/LTX-2.5"
LTX_MODEL_REVISION = "6c7e5e573ac1667efc83407806fe9b0b93730e60"
LTX_SOURCE_COMMIT = "400fd31054597515f47125691032c04b1c3ee24e"
GEMMA_VERSION = "gemma4-12b-ltx-v1"
LTX_MODEL_VERSION = "2.5.0"
SCHEMA_VERSION = 1
EMBEDDING_KEY = "prompt_embedding"
ATTENTION_MASK_KEY = "attention_mask"


class LTXPromptEmbeddingError(ValueError):
    """An LTX prompt artifact violates the frozen experiment contract."""


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor value, shape, and dtype on CPU."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a source checkpoint in bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LTXPromptProvenance:
    """Pinned producer identity and value hashes for one Gemma-4 context."""

    prompt: str
    model_repository: str
    model_revision: str
    source_commit: str
    gemma_version: str
    ltx_model_version: str
    text_encoder_path: str
    text_encoder_size_bytes: int
    text_encoder_sha256: str
    transformer_path: str
    transformer_size_bytes: int
    transformer_sha256: str
    shape: tuple[int, int, int]
    dtype: str
    valid_tokens: int
    embedding_sha256: str
    attention_mask_sha256: str
    torch_version: str
    generation_time_utc: str
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LTXPromptProvenance:
        expected = set(
            cls(
                prompt="",
                model_repository="",
                model_revision="",
                source_commit="",
                gemma_version="",
                ltx_model_version="",
                text_encoder_path="/",
                text_encoder_size_bytes=0,
                text_encoder_sha256="",
                transformer_path="/",
                transformer_size_bytes=0,
                transformer_sha256="",
                shape=EXPECTED_SHAPE,
                dtype="bfloat16",
                valid_tokens=1,
                embedding_sha256="",
                attention_mask_sha256="",
                torch_version="",
                generation_time_utc="1970-01-01T00:00:00Z",
            ).to_dict()
        )
        if set(payload) != expected:
            raise LTXPromptEmbeddingError(
                f"prompt provenance keys mismatch: missing={sorted(expected - set(payload))}, "
                f"extra={sorted(set(payload) - expected)}"
            )
        fixed = {
            "schema_version": SCHEMA_VERSION,
            "prompt": DEFAULT_PROMPT,
            "model_repository": LTX_REPOSITORY,
            "model_revision": LTX_MODEL_REVISION,
            "source_commit": LTX_SOURCE_COMMIT,
            "gemma_version": GEMMA_VERSION,
            "ltx_model_version": LTX_MODEL_VERSION,
            "dtype": "bfloat16",
        }
        for name, expected_value in fixed.items():
            if payload[name] != expected_value:
                raise LTXPromptEmbeddingError(
                    f"prompt provenance {name} must be {expected_value!r}, got {payload[name]!r}"
                )
        shape = payload["shape"]
        if not isinstance(shape, list) or tuple(shape) != EXPECTED_SHAPE:
            raise LTXPromptEmbeddingError(f"prompt provenance shape must be {list(EXPECTED_SHAPE)}")
        for name in ("text_encoder_path", "transformer_path", "torch_version", "generation_time_utc"):
            if not isinstance(payload[name], str) or not payload[name]:
                raise LTXPromptEmbeddingError(f"prompt provenance {name} must be non-empty")
        for name in ("text_encoder_path", "transformer_path"):
            if not Path(payload[name]).is_absolute():
                raise LTXPromptEmbeddingError(f"prompt provenance {name} must be absolute")
        for name in ("text_encoder_size_bytes", "transformer_size_bytes", "valid_tokens"):
            if type(payload[name]) is not int or payload[name] <= 0:
                raise LTXPromptEmbeddingError(f"prompt provenance {name} must be positive")
        if payload["valid_tokens"] > EXPECTED_SHAPE[1]:
            raise LTXPromptEmbeddingError("prompt provenance valid_tokens exceeds sequence length")
        for name in (
            "text_encoder_sha256",
            "transformer_sha256",
            "embedding_sha256",
            "attention_mask_sha256",
        ):
            value = payload[name]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise LTXPromptEmbeddingError(f"prompt provenance {name} must be lowercase SHA256")
        timestamp = payload["generation_time_utc"]
        try:
            parsed = datetime.fromisoformat(timestamp.removesuffix("Z") + "+00:00")
        except ValueError as exc:
            raise LTXPromptEmbeddingError("generation_time_utc must be RFC3339 UTC") from exc
        if not timestamp.endswith("Z") or parsed.tzinfo != UTC:
            raise LTXPromptEmbeddingError("generation_time_utc must be RFC3339 UTC")
        return cls(**{**payload, "shape": tuple(shape)})


@dataclass(frozen=True, slots=True)
class LTXPromptArtifact:
    """Validated CPU tensors and their producer provenance."""

    embedding: torch.Tensor
    attention_mask: torch.Tensor
    provenance: LTXPromptProvenance


def validate_prompt_tensors(embedding: torch.Tensor, attention_mask: torch.Tensor) -> None:
    """Reject zero fallbacks, malformed values, and inconsistent padding."""
    if tuple(embedding.shape) != EXPECTED_SHAPE or embedding.dtype != EXPECTED_DTYPE:
        raise LTXPromptEmbeddingError(
            f"prompt embedding must be bfloat16 {EXPECTED_SHAPE}, got {embedding.dtype} {tuple(embedding.shape)}"
        )
    if tuple(attention_mask.shape) != EXPECTED_SHAPE[:2] or attention_mask.dtype != torch.bool:
        raise LTXPromptEmbeddingError("attention mask must be bool [1, 1024]")
    if not torch.isfinite(embedding).all().item():
        raise LTXPromptEmbeddingError("prompt embedding must be finite")
    valid_tokens = int(attention_mask.sum().item())
    if valid_tokens <= 0:
        raise LTXPromptEmbeddingError("prompt attention mask contains no valid tokens")
    if not torch.count_nonzero(embedding).item():
        raise LTXPromptEmbeddingError("zero prompt embeddings are forbidden")
    if torch.count_nonzero(embedding.masked_select(~attention_mask.unsqueeze(-1))).item():
        raise LTXPromptEmbeddingError("padded prompt embedding rows must be exactly zero")


def build_provenance(
    embedding: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    text_encoder_path: str | Path,
    transformer_path: str | Path,
    generation_time_utc: str | None = None,
) -> LTXPromptProvenance:
    """Hash official source files and construct strict provenance."""
    validate_prompt_tensors(embedding, attention_mask)
    text_encoder = Path(text_encoder_path).expanduser().resolve()
    transformer = Path(transformer_path).expanduser().resolve()
    for path in (text_encoder, transformer):
        if not path.is_file():
            raise FileNotFoundError(path)
    generated = generation_time_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return LTXPromptProvenance(
        prompt=DEFAULT_PROMPT,
        model_repository=LTX_REPOSITORY,
        model_revision=LTX_MODEL_REVISION,
        source_commit=LTX_SOURCE_COMMIT,
        gemma_version=GEMMA_VERSION,
        ltx_model_version=LTX_MODEL_VERSION,
        text_encoder_path=str(text_encoder),
        text_encoder_size_bytes=text_encoder.stat().st_size,
        text_encoder_sha256=file_sha256(text_encoder),
        transformer_path=str(transformer),
        transformer_size_bytes=transformer.stat().st_size,
        transformer_sha256=file_sha256(transformer),
        shape=EXPECTED_SHAPE,
        dtype="bfloat16",
        valid_tokens=int(attention_mask.sum().item()),
        embedding_sha256=tensor_sha256(embedding),
        attention_mask_sha256=tensor_sha256(attention_mask),
        torch_version=torch.__version__,
        generation_time_utc=generated,
    )


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def save_ltx_prompt_artifact(
    artifact: LTXPromptArtifact, path: str | Path, *, overwrite: bool = False
) -> Path:
    """Atomically persist a pickle-free prompt artifact and JSON sidecar."""
    path = Path(path).expanduser()
    validate_prompt_tensors(artifact.embedding, artifact.attention_mask)
    provenance = LTXPromptProvenance.from_dict(artifact.provenance.to_dict())
    if provenance.embedding_sha256 != tensor_sha256(artifact.embedding):
        raise LTXPromptEmbeddingError("embedding hash does not match provenance")
    if provenance.attention_mask_sha256 != tensor_sha256(artifact.attention_mask):
        raise LTXPromptEmbeddingError("attention mask hash does not match provenance")
    if not overwrite and (path.exists() or _sidecar(path).exists()):
        raise FileExistsError(f"refusing to overwrite LTX prompt artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    sidecar_temporary = _sidecar(path).with_name(f".{_sidecar(path).name}.{os.getpid()}.tmp")
    try:
        save_file(
            {
                EMBEDDING_KEY: artifact.embedding.detach().cpu().contiguous(),
                ATTENTION_MASK_KEY: artifact.attention_mask.detach().cpu().contiguous(),
            },
            str(temporary),
            metadata={"format": "ltx_gemma4_prompt", "schema_version": str(SCHEMA_VERSION)},
        )
        sidecar_temporary.write_text(json.dumps(provenance.to_dict(), indent=2) + "\n")
        os.replace(temporary, path)
        os.replace(sidecar_temporary, _sidecar(path))
    finally:
        temporary.unlink(missing_ok=True)
        sidecar_temporary.unlink(missing_ok=True)
    return path


def load_ltx_prompt_artifact(path: str | Path) -> LTXPromptArtifact:
    """Load and fully validate an LTX prompt artifact; never substitute zeros."""
    path = Path(path).expanduser()
    if not path.is_file() or not _sidecar(path).is_file():
        raise FileNotFoundError(f"LTX prompt artifact and sidecar are required: {path}")
    tensors = load_file(str(path), device="cpu")
    if set(tensors) != {EMBEDDING_KEY, ATTENTION_MASK_KEY}:
        raise LTXPromptEmbeddingError("LTX prompt safetensors keys are malformed")
    embedding = tensors[EMBEDDING_KEY]
    attention_mask = tensors[ATTENTION_MASK_KEY].to(dtype=torch.bool)
    validate_prompt_tensors(embedding, attention_mask)
    try:
        payload = json.loads(_sidecar(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LTXPromptEmbeddingError(f"could not read LTX prompt sidecar: {exc}") from exc
    provenance = LTXPromptProvenance.from_dict(payload)
    if provenance.embedding_sha256 != tensor_sha256(embedding):
        raise LTXPromptEmbeddingError("LTX prompt embedding hash mismatch")
    if provenance.attention_mask_sha256 != tensor_sha256(attention_mask):
        raise LTXPromptEmbeddingError("LTX prompt attention mask hash mismatch")
    if provenance.valid_tokens != int(attention_mask.sum().item()):
        raise LTXPromptEmbeddingError("LTX prompt valid-token count mismatch")
    return LTXPromptArtifact(embedding, attention_mask, provenance)
