"""Generate and validate the frozen Cosmos T5 prompt embedding artifact.

This module owns the small, pickle-free artifact consumed by the frozen Cosmos
Video2World extractor.  The tokenizer/model imports remain in the CLI so that
importing LeRobot never loads Transformers or the 45 GB T5 checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

DEFAULT_PROMPT = "take cube out of box"
DEFAULT_MAX_LENGTH = 512
EXPECTED_EMBEDDING_SHAPE = (1, DEFAULT_MAX_LENGTH, 1024)
EXPECTED_EMBEDDING_DTYPE = torch.bfloat16
EXPECTED_DTYPE_NAME = "bfloat16"

MODEL_REPOSITORY = "jonpai/mimic-video"
MODEL_REVISION = "f28339034831e3c2374be075e622e1ff38ebe0f8"
MODEL_FILENAME = "pytorch_model.bin"
MODEL_SIZE_BYTES = 45_229_452_544
T5_SHA256 = "5fdc64177b14b0f72437fea171a752c626733f67755842528c75b90cacd1c807"
MIMIC_VIDEO_COMMIT = "e3355dbc93132b576c02f920a59b4fc18a4f5906"
TOKENIZATION_API = "batch_encode_plus_or_tokenizer_call_same_kwargs"
DEFAULT_MODEL_PATH = Path("/home/anton/.cache/video-vam/mimic-video-f2833903/text_encoder/t5-11b")
DEFAULT_OUTPUT_PATH = Path(
    "/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors"
)

EMBEDDING_KEY = "prompt_embedding"
ATTENTION_MASK_KEY = "attention_mask"
_ARTIFACT_KEYS = frozenset({EMBEDDING_KEY, ATTENTION_MASK_KEY})
_SCHEMA_VERSION = 1
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HASH_CHUNK_BYTES = 1024 * 1024


class PromptEmbeddingError(RuntimeError):
    """Base error for prompt embedding artifact failures."""


class PromptEmbeddingValidationError(PromptEmbeddingError, ValueError):
    """An artifact or generation input violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class PromptEmbeddingProvenance:
    """JSON provenance for one official T5 prompt embedding."""

    prompt: str
    max_length: int
    tokenization_api: str
    model_repo: str
    model_revision: str
    model_path: str
    model_file: str
    model_size_bytes: int
    t5_sha256: str
    mimic_commit: str
    transformers_version: str
    torch_version: str
    token_ids_sha256: str
    output_sha256: str
    dtype: str
    shape: tuple[int, int, int]
    generation_time_utc: str
    schema_version: int = _SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the stable sidecar representation."""
        return {
            "schema_version": self.schema_version,
            "prompt": self.prompt,
            "max_length": self.max_length,
            "tokenization_api": self.tokenization_api,
            "model_repo": self.model_repo,
            "model_revision": self.model_revision,
            "model_path": self.model_path,
            "model_file": self.model_file,
            "model_size_bytes": self.model_size_bytes,
            "t5_sha256": self.t5_sha256,
            "mimic_commit": self.mimic_commit,
            "transformers_version": self.transformers_version,
            "torch_version": self.torch_version,
            "token_ids_sha256": self.token_ids_sha256,
            "output_sha256": self.output_sha256,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "generation_time_utc": self.generation_time_utc,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PromptEmbeddingProvenance:
        """Parse and strictly validate a provenance sidecar."""
        # Empty-string field sentinels for key discovery, not credentials.
        expected_keys = set(
            cls(  # nosec B106
                prompt="",
                max_length=0,
                tokenization_api="",
                model_repo="",
                model_revision="",
                model_path="/",
                model_file="",
                model_size_bytes=0,
                t5_sha256="",
                mimic_commit="",
                transformers_version="",
                torch_version="",
                token_ids_sha256="",
                output_sha256="",
                dtype="",
                shape=(0, 0, 0),
                generation_time_utc="",
            ).to_dict()
        )
        if set(payload) != expected_keys:
            missing = sorted(expected_keys - set(payload))
            extra = sorted(set(payload) - expected_keys)
            raise PromptEmbeddingValidationError(
                f"Provenance keys must match exactly; missing={missing}, extra={extra}"
            )

        def string(name: str) -> str:
            value = payload[name]
            if not isinstance(value, str) or not value:
                raise PromptEmbeddingValidationError(f"Provenance field {name!r} must be a non-empty string")
            return value

        schema_version = payload["schema_version"]
        if type(schema_version) is not int or schema_version != _SCHEMA_VERSION:
            raise PromptEmbeddingValidationError("Unsupported provenance schema_version")
        prompt = payload["prompt"]
        if prompt != DEFAULT_PROMPT:
            raise PromptEmbeddingValidationError(f"Unexpected prompt {prompt!r}")
        max_length = payload["max_length"]
        if type(max_length) is not int or max_length != DEFAULT_MAX_LENGTH:
            raise PromptEmbeddingValidationError("Provenance max_length must be exactly 512")
        tokenization_api = string("tokenization_api")
        if tokenization_api != TOKENIZATION_API:
            raise PromptEmbeddingValidationError("Unexpected tokenizer API compatibility provenance")
        model_repo = string("model_repo")
        if model_repo != MODEL_REPOSITORY:
            raise PromptEmbeddingValidationError(f"Unexpected model_repo {model_repo!r}")
        model_revision = string("model_revision")
        if model_revision != MODEL_REVISION:
            raise PromptEmbeddingValidationError(f"Unexpected model_revision {model_revision!r}")
        model_path = string("model_path")
        if not Path(model_path).is_absolute():
            raise PromptEmbeddingValidationError("Provenance model_path must be absolute")
        model_file = string("model_file")
        if model_file != MODEL_FILENAME:
            raise PromptEmbeddingValidationError(f"Unexpected model_file {model_file!r}")
        model_size_bytes = payload["model_size_bytes"]
        if type(model_size_bytes) is not int or model_size_bytes != MODEL_SIZE_BYTES:
            raise PromptEmbeddingValidationError("Provenance model_size_bytes does not match the pinned T5")
        t5_sha256 = string("t5_sha256")
        if t5_sha256 != T5_SHA256:
            raise PromptEmbeddingValidationError("Provenance t5_sha256 does not match the pinned T5")
        mimic_commit = string("mimic_commit")
        if mimic_commit != MIMIC_VIDEO_COMMIT:
            raise PromptEmbeddingValidationError(
                "Provenance mimic_commit does not match the official algorithm"
            )
        transformers_version = string("transformers_version")
        torch_version = string("torch_version")
        token_ids_sha256 = string("token_ids_sha256")
        output_sha256 = string("output_sha256")
        for name, value in (("token_ids_sha256", token_ids_sha256), ("output_sha256", output_sha256)):
            if _HEX_SHA256.fullmatch(value) is None:
                raise PromptEmbeddingValidationError(f"Provenance {name} must be a lowercase SHA256")
        dtype = payload["dtype"]
        if dtype != EXPECTED_DTYPE_NAME:
            raise PromptEmbeddingValidationError(f"Provenance dtype must be {EXPECTED_DTYPE_NAME!r}")
        shape = payload["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(value) is not int for value in shape)
            or tuple(shape) != EXPECTED_EMBEDDING_SHAPE
        ):
            raise PromptEmbeddingValidationError(f"Provenance shape must be {list(EXPECTED_EMBEDDING_SHAPE)}")
        generation_time_utc = string("generation_time_utc")
        if not generation_time_utc.endswith("Z"):
            raise PromptEmbeddingValidationError("generation_time_utc must use a UTC Z suffix")
        try:
            parsed_time = datetime.fromisoformat(generation_time_utc[:-1] + "+00:00")
        except ValueError as exc:
            raise PromptEmbeddingValidationError("generation_time_utc must be RFC3339") from exc
        if parsed_time.tzinfo != UTC:
            raise PromptEmbeddingValidationError("generation_time_utc must be UTC")

        return cls(
            schema_version=schema_version,
            prompt=prompt,
            max_length=max_length,
            tokenization_api=tokenization_api,
            model_repo=model_repo,
            model_revision=model_revision,
            model_path=model_path,
            model_file=model_file,
            model_size_bytes=model_size_bytes,
            t5_sha256=t5_sha256,
            mimic_commit=mimic_commit,
            transformers_version=transformers_version,
            torch_version=torch_version,
            token_ids_sha256=token_ids_sha256,
            output_sha256=output_sha256,
            dtype=dtype,
            shape=tuple(shape),
            generation_time_utc=generation_time_utc,
        )


@dataclass(frozen=True, slots=True)
class PromptEmbeddingArtifact:
    """Embedding and mask paired with their strict provenance."""

    embedding: torch.Tensor
    attention_mask: torch.Tensor
    provenance: PromptEmbeddingProvenance


@dataclass(frozen=True, slots=True)
class PromptEmbeddingGeneration:
    """In-memory result of one tokenizer/model forward pass."""

    embedding: torch.Tensor
    attention_mask: torch.Tensor
    token_ids: torch.Tensor


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash the tensor's contiguous native byte representation."""
    value = tensor.detach().cpu().contiguous()
    raw_bytes = value.view(torch.uint8).reshape(-1).numpy().tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


def _sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(".json")


def _get_encoded_value(encoded: Any, key: str) -> torch.Tensor:
    try:
        value = encoded[key]
    except (KeyError, TypeError):
        value = getattr(encoded, key, None)
    if not isinstance(value, torch.Tensor):
        raise PromptEmbeddingValidationError(f"Tokenizer output must contain tensor {key!r}")
    return value


def _validate_tokenized_inputs(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
    if tuple(input_ids.shape) != (1, DEFAULT_MAX_LENGTH):
        raise PromptEmbeddingValidationError(
            f"input_ids must have shape [1, 512], got {tuple(input_ids.shape)}"
        )
    if tuple(attention_mask.shape) != tuple(input_ids.shape):
        raise PromptEmbeddingValidationError("attention_mask must have the same shape as input_ids")
    if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
        raise PromptEmbeddingValidationError("input_ids must be a rank-2 integer tensor")
    if attention_mask.dtype not in (torch.bool, torch.int32, torch.int64):
        raise PromptEmbeddingValidationError("attention_mask must be boolean or integer")
    mask = attention_mask.to(dtype=torch.bool)
    length = int(mask.sum().item())
    if not 0 < length <= DEFAULT_MAX_LENGTH:
        raise PromptEmbeddingValidationError("attention_mask must contain at least one token")
    expected_mask = torch.cat(
        [
            torch.ones(length, dtype=torch.bool, device=mask.device),
            torch.zeros(DEFAULT_MAX_LENGTH - length, dtype=torch.bool, device=mask.device),
        ]
    )
    if not torch.equal(mask[0], expected_mask):
        raise PromptEmbeddingValidationError("attention_mask must contain prompt tokens followed by padding")


def _model_input_device(model: Any) -> torch.device:
    """Return the real device of the input embedding in an Accelerate-sharded model."""
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.device.type != "meta":
            return weight.device
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            if parameter.device.type != "meta":
                return parameter.device
    raise PromptEmbeddingValidationError("Could not determine the T5 input embedding device")


def generate_prompt_embedding(
    tokenizer: Any,
    model: Any,
    *,
    prompt: str = DEFAULT_PROMPT,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> PromptEmbeddingGeneration:
    """Run the exact mimic-video T5 preprocessing and zero padded positions."""
    if prompt != DEFAULT_PROMPT or max_length != DEFAULT_MAX_LENGTH:
        raise PromptEmbeddingValidationError(
            "This artifact is fixed to the official prompt and max_length=512"
        )
    tokenization_kwargs = {
        "return_tensors": "pt",
        "truncation": True,
        "padding": "max_length",
        "max_length": DEFAULT_MAX_LENGTH,
        "return_length": True,
    }
    batch_encode_plus = getattr(tokenizer, "batch_encode_plus", None)
    if callable(batch_encode_plus):
        batch = batch_encode_plus([prompt], **tokenization_kwargs)
    elif callable(tokenizer):
        batch = tokenizer([prompt], **tokenization_kwargs)
    else:
        raise PromptEmbeddingValidationError(
            "Tokenizer must expose callable batch_encode_plus or be callable"
        )
    input_ids = _get_encoded_value(batch, "input_ids")
    attention_mask = _get_encoded_value(batch, "attention_mask")
    _validate_tokenized_inputs(input_ids, attention_mask)

    input_device = _model_input_device(model)
    input_ids_for_model = input_ids.to(device=input_device)
    attention_mask_for_model = attention_mask.to(device=input_device)
    eval_method = getattr(model, "eval", None)
    if callable(eval_method):
        eval_method()
    with torch.inference_mode():
        output = model(input_ids=input_ids_for_model, attention_mask=attention_mask_for_model)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, Mapping):
        hidden = output.get("last_hidden_state")
    if not isinstance(hidden, torch.Tensor):
        raise PromptEmbeddingValidationError("T5 forward output must contain last_hidden_state")
    if tuple(hidden.shape) != EXPECTED_EMBEDDING_SHAPE:
        raise PromptEmbeddingValidationError(
            f"last_hidden_state must have shape {EXPECTED_EMBEDDING_SHAPE}, got {tuple(hidden.shape)}"
        )
    if not torch.is_floating_point(hidden) or not torch.isfinite(hidden).all().item():
        raise PromptEmbeddingValidationError(
            "last_hidden_state must contain only finite floating-point values"
        )

    embedding = hidden.detach().to(dtype=EXPECTED_EMBEDDING_DTYPE).clone()
    mask_cpu = attention_mask.to(device="cpu", dtype=torch.bool)
    for row, length in enumerate(mask_cpu.sum(dim=1).tolist()):
        embedding[row, int(length) :] = 0
    if not torch.isfinite(embedding).all().item():
        raise PromptEmbeddingValidationError("Converted prompt embedding contains non-finite values")
    return PromptEmbeddingGeneration(
        embedding=embedding.cpu(),
        attention_mask=mask_cpu,
        token_ids=input_ids.detach().cpu().contiguous(),
    )


def build_provenance(
    generation: PromptEmbeddingGeneration,
    *,
    model_path: Path,
    transformers_version: str,
    torch_version: str,
    generation_time_utc: str | None = None,
) -> PromptEmbeddingProvenance:
    """Create provenance after the model output has been normalized."""
    if tuple(generation.embedding.shape) != EXPECTED_EMBEDDING_SHAPE:
        raise PromptEmbeddingValidationError("Cannot build provenance for the wrong embedding shape")
    if generation.embedding.dtype != EXPECTED_EMBEDDING_DTYPE:
        raise PromptEmbeddingValidationError("Cannot build provenance for a non-bfloat16 embedding")
    if not Path(model_path).is_absolute():
        raise PromptEmbeddingValidationError("model_path must be absolute in provenance")
    timestamp = generation_time_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return PromptEmbeddingProvenance(
        prompt=DEFAULT_PROMPT,
        max_length=DEFAULT_MAX_LENGTH,
        tokenization_api=TOKENIZATION_API,
        model_repo=MODEL_REPOSITORY,
        model_revision=MODEL_REVISION,
        model_path=str(Path(model_path)),
        model_file=MODEL_FILENAME,
        model_size_bytes=MODEL_SIZE_BYTES,
        t5_sha256=T5_SHA256,
        mimic_commit=MIMIC_VIDEO_COMMIT,
        transformers_version=transformers_version,
        torch_version=torch_version,
        token_ids_sha256=_tensor_sha256(generation.token_ids),
        output_sha256=_tensor_sha256(generation.embedding),
        dtype=EXPECTED_DTYPE_NAME,
        shape=EXPECTED_EMBEDDING_SHAPE,
        generation_time_utc=timestamp,
    )


def _validate_artifact_tensors(embedding: torch.Tensor, attention_mask: torch.Tensor) -> None:
    if tuple(embedding.shape) != EXPECTED_EMBEDDING_SHAPE:
        raise PromptEmbeddingValidationError(
            f"embedding must have shape {EXPECTED_EMBEDDING_SHAPE}, got {tuple(embedding.shape)}"
        )
    if embedding.dtype != EXPECTED_EMBEDDING_DTYPE:
        raise PromptEmbeddingValidationError("embedding must have dtype torch.bfloat16")
    if not torch.isfinite(embedding).all().item():
        raise PromptEmbeddingValidationError("embedding must contain only finite values")
    if tuple(attention_mask.shape) != (1, DEFAULT_MAX_LENGTH) or attention_mask.dtype != torch.bool:
        raise PromptEmbeddingValidationError("attention_mask must be bool with shape [1, 512]")
    length = int(attention_mask.sum().item())
    if not 0 < length <= DEFAULT_MAX_LENGTH:
        raise PromptEmbeddingValidationError("attention_mask must contain at least one token")
    if not torch.equal(
        attention_mask[0],
        torch.cat(
            [
                torch.ones(length, dtype=torch.bool, device=attention_mask.device),
                torch.zeros(DEFAULT_MAX_LENGTH - length, dtype=torch.bool, device=attention_mask.device),
            ]
        ),
    ):
        raise PromptEmbeddingValidationError("attention_mask must contain prompt tokens followed by padding")
    if not torch.equal(embedding[0, length:], torch.zeros_like(embedding[0, length:])):
        raise PromptEmbeddingValidationError("padded embedding positions must be exactly zero")


def save_prompt_embedding(
    artifact: PromptEmbeddingArtifact,
    output_path: Path,
    *,
    overwrite: bool = False,
    provenance_path: Path | None = None,
) -> tuple[Path, Path]:
    """Save validated tensors as safetensors plus a JSON sidecar."""
    output_path = Path(output_path).expanduser()
    sidecar = Path(provenance_path).expanduser() if provenance_path else _sidecar_path(output_path)
    if (output_path.exists() or sidecar.exists()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing prompt embedding output; pass --overwrite: {output_path}"
        )
    embedding = artifact.embedding.detach().cpu().contiguous()
    attention_mask = artifact.attention_mask.detach().cpu().contiguous()
    _validate_artifact_tensors(embedding, attention_mask)
    provenance = artifact.provenance
    if tuple(provenance.shape) != EXPECTED_EMBEDDING_SHAPE or provenance.dtype != EXPECTED_DTYPE_NAME:
        raise PromptEmbeddingValidationError("Provenance shape/dtype does not match the tensors")
    if provenance.output_sha256 != _tensor_sha256(embedding):
        raise PromptEmbeddingValidationError("Provenance output_sha256 does not match the embedding")
    provenance = PromptEmbeddingProvenance.from_dict(provenance.to_dict())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    temp_sidecar = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
    try:
        save_file({EMBEDDING_KEY: embedding, ATTENTION_MASK_KEY: attention_mask}, str(temp_output))
        temp_sidecar.write_text(json.dumps(provenance.to_dict(), indent=2, sort_keys=False) + "\n")
        os.replace(temp_output, output_path)
        os.replace(temp_sidecar, sidecar)
    finally:
        temp_output.unlink(missing_ok=True)
        temp_sidecar.unlink(missing_ok=True)
    return output_path, sidecar


def load_prompt_embedding(
    output_path: Path, *, provenance_path: Path | None = None
) -> PromptEmbeddingArtifact:
    """Load and strictly validate the safetensors artifact and JSON sidecar."""
    output_path = Path(output_path).expanduser()
    sidecar = Path(provenance_path).expanduser() if provenance_path else _sidecar_path(output_path)
    if not output_path.is_file():
        raise FileNotFoundError(f"Prompt embedding safetensors file not found: {output_path}")
    tensors = load_file(str(output_path), device="cpu")
    if set(tensors) != _ARTIFACT_KEYS:
        raise PromptEmbeddingValidationError(
            f"Safetensors keys must be exactly {sorted(_ARTIFACT_KEYS)}, got {sorted(tensors)}"
        )
    embedding = tensors[EMBEDDING_KEY]
    attention_mask = tensors[ATTENTION_MASK_KEY]
    _validate_artifact_tensors(embedding, attention_mask)
    if not sidecar.is_file():
        raise FileNotFoundError(f"Prompt embedding provenance sidecar not found: {sidecar}")
    try:
        payload = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PromptEmbeddingValidationError(f"Could not read provenance JSON {sidecar}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PromptEmbeddingValidationError("Provenance JSON must contain an object")
    provenance = PromptEmbeddingProvenance.from_dict(payload)
    if tuple(provenance.shape) != tuple(embedding.shape):
        raise PromptEmbeddingValidationError("Provenance shape does not match the safetensors embedding")
    if provenance.output_sha256 != _tensor_sha256(embedding):
        raise PromptEmbeddingValidationError(
            "Provenance output_sha256 does not match the safetensors embedding"
        )
    return PromptEmbeddingArtifact(embedding, attention_mask, provenance)


def _sha256_file(path: Path) -> str:
    """Hash a local model file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ModelVerification:
    """Verified local T5 model file details."""

    model_dir: Path
    model_file: Path
    size_bytes: int
    sha256: str
    hash_verified: bool


def verify_t5_model(
    model_dir: Path,
    *,
    skip_hash: bool = False,
    expected_size: int = MODEL_SIZE_BYTES,
    expected_sha256: str = T5_SHA256,
) -> ModelVerification:
    """Verify presence, size, and (unless explicitly skipped) streamed SHA256."""
    model_dir = Path(model_dir).expanduser()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"T5 model directory not found: {model_dir}")
    model_file = model_dir / MODEL_FILENAME
    if not model_file.is_file():
        raise FileNotFoundError(f"T5 model file not found: {model_file}")
    size_bytes = model_file.stat().st_size
    if size_bytes != expected_size:
        raise PromptEmbeddingValidationError(
            f"T5 model size mismatch for {model_file}: expected {expected_size}, got {size_bytes}"
        )
    if skip_hash:
        return ModelVerification(model_dir, model_file, size_bytes, expected_sha256, False)
    actual_sha256 = _sha256_file(model_file)
    if actual_sha256 != expected_sha256:
        raise PromptEmbeddingValidationError(
            f"T5 model SHA256 mismatch for {model_file}: expected {expected_sha256}, got {actual_sha256}"
        )
    return ModelVerification(model_dir, model_file, size_bytes, actual_sha256, True)
