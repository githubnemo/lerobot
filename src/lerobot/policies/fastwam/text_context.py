# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Safe, provenance-checked storage for precomputed FastWAM text context."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

TEXT_CONTEXT_FORMAT_VERSION = 1
_CONTEXT_TENSOR_KEY = "context"
_CONTEXT_MASK_TENSOR_KEY = "context_mask"
_PROVENANCE_FIELDS = (
    "model_id",
    "tokenizer_model_id",
    "text_encoder_model_id",
    "dtype",
    "prompt_template",
    "tokenizer_max_len",
    "context_len",
    "context_dim",
)


@dataclass(frozen=True)
class TextContextProvenance:
    """Configuration identity recorded beside a precomputed context artifact."""

    model_id: str
    tokenizer_model_id: str
    text_encoder_model_id: str
    dtype: str
    prompt_template: str
    tokenizer_max_len: int
    context_len: int
    context_dim: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TextContextProvenance:
        missing = [key for key in _PROVENANCE_FIELDS if key not in value]
        if missing:
            raise ValueError(f"Text context provenance is missing field(s): {missing}")
        data = {key: value[key] for key in _PROVENANCE_FIELDS}
        data["tokenizer_max_len"] = int(data["tokenizer_max_len"])
        data["context_len"] = int(data["context_len"])
        data["context_dim"] = int(data["context_dim"])
        return cls(**data)


def format_task_prompt(task: str, prompt_template: str) -> str:
    """Apply the exact FastWAM task template used by online prompt encoding."""
    try:
        return prompt_template.format(task=str(task))
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"FastWAM prompt_template must be format-compatible with `{{task}}`; got {prompt_template!r}."
        ) from exc


def format_task_prompts(tasks: Sequence[str], prompt_template: str) -> list[str]:
    """Format and de-duplicate task strings while preserving first-seen order."""
    formatted: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        prompt = format_task_prompt(task, prompt_template)
        if prompt not in seen:
            formatted.append(prompt)
            seen.add(prompt)
    if not formatted:
        raise ValueError("At least one task/prompt is required to build a text context artifact.")
    return formatted


def _normalize_prompts(prompts: str | Sequence[str]) -> list[str]:
    if isinstance(prompts, str):
        return [prompts]
    values = [str(prompt) for prompt in prompts]
    if not values:
        raise ValueError("At least one prompt is required.")
    return values


@torch.no_grad()
def encode_wan_text_context(
    tokenizer: Any,
    text_encoder: Any,
    prompts: str | Sequence[str],
    *,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the canonical UMT5 -> FastWAM context conversion.

    This is deliberately shared by online ``FastWAM.encode_prompt`` and the offline
    precompute script. Padding embeddings are zeroed, while the returned mask is all
    true over the fixed tokenizer length, matching the native Wan/FastWAM contract.
    """
    prompt_list = _normalize_prompts(prompts)
    ids, tokenizer_mask = tokenizer(prompt_list, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    tokenizer_mask = tokenizer_mask.to(device, dtype=torch.bool)
    prompt_emb = text_encoder(ids, tokenizer_mask)
    if prompt_emb.ndim != 3:
        raise ValueError(f"UMT5 text encoder must return [B,L,D], got {tuple(prompt_emb.shape)}")
    if prompt_emb.shape[:2] != tokenizer_mask.shape:
        raise ValueError(
            "UMT5 text encoder output shape does not match tokenizer mask: "
            f"embeddings={tuple(prompt_emb.shape)}, mask={tuple(tokenizer_mask.shape)}"
        )
    prompt_emb = prompt_emb.clone()
    seq_lens = tokenizer_mask.gt(0).sum(dim=1).long()
    for row, length in enumerate(seq_lens.tolist()):
        prompt_emb[row, length:] = 0
    # Cross-attention sees a fixed-length context; padding is represented by zero vectors.
    context_mask = torch.ones_like(tokenizer_mask, dtype=torch.bool)
    return prompt_emb, context_mask


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def save_text_context_artifact(
    path: str | Path,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    prompts: Sequence[str],
    provenance: TextContextProvenance | Mapping[str, Any],
) -> Path:
    """Atomically save tensors and JSON provenance for a prompt-indexed artifact."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    context = context.detach().to(device="cpu")
    context_mask = context_mask.detach().to(device="cpu", dtype=torch.bool)
    if context.ndim != 3 or context_mask.ndim != 2:
        raise ValueError(
            f"Text context tensors must be [N,L,D]/[N,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
        )
    if context.shape[:2] != context_mask.shape:
        raise ValueError(
            f"Text context shape must match mask shape, got {tuple(context.shape)} and {tuple(context_mask.shape)}"
        )
    prompt_list = list(prompts)
    if len(prompt_list) != context.shape[0] or len(set(prompt_list)) != len(prompt_list):
        raise ValueError("Artifact prompts must be unique and have one entry per context row.")
    if isinstance(provenance, TextContextProvenance):
        provenance_obj = provenance
    else:
        provenance_obj = TextContextProvenance.from_dict(provenance)
    if (context.shape[1], context.shape[2]) != (provenance_obj.context_len, provenance_obj.context_dim):
        raise ValueError(
            "Context shape does not match provenance: "
            f"tensor={tuple(context.shape[1:])}, provenance={(provenance_obj.context_len, provenance_obj.context_dim)}"
        )
    if str(context.dtype).removeprefix("torch.") != provenance_obj.dtype:
        raise ValueError(
            f"Context dtype {context.dtype} does not match provenance dtype {provenance_obj.dtype!r}."
        )

    prompt_to_index = {prompt: index for index, prompt in enumerate(prompt_list)}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_file(
            {_CONTEXT_TENSOR_KEY: context, _CONTEXT_MASK_TENSOR_KEY: context_mask},
            str(temporary),
            metadata={"format": "fastwam_text_context", "format_version": str(TEXT_CONTEXT_FORMAT_VERSION)},
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    tensors_sha256 = _sha256_file(path)
    metadata = {
        "format": "fastwam_text_context",
        "format_version": TEXT_CONTEXT_FORMAT_VERSION,
        "tensor_file": path.name,
        "tensor_file_sha256": tensors_sha256,
        "tensor_keys": [_CONTEXT_TENSOR_KEY, _CONTEXT_MASK_TENSOR_KEY],
        "context_shape": list(context.shape),
        "context_mask_shape": list(context_mask.shape),
        "prompt_to_index": prompt_to_index,
        "prompt_hashes": {prompt: _prompt_hash(prompt) for prompt in prompt_list},
        "prompts_sha256": hashlib.sha256(_canonical_json(prompt_list).encode("utf-8")).hexdigest(),
        "provenance": provenance_obj.as_dict(),
    }
    _atomic_write_json(_sidecar_path(path), metadata)
    return path


@dataclass
class TextContextArtifact:
    """Loaded CPU-resident context rows indexed by formatted prompt string."""

    path: Path
    context: torch.Tensor
    context_mask: torch.Tensor
    prompt_to_index: dict[str, int]
    provenance: TextContextProvenance

    @property
    def prompts(self) -> tuple[str, ...]:
        return tuple(prompt for prompt, _ in sorted(self.prompt_to_index.items(), key=lambda item: item[1]))

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.context.shape)  # type: ignore[return-value]

    def validate_prompts(self, prompts: str | Sequence[str]) -> None:
        values = _normalize_prompts(prompts)
        missing = sorted(set(values).difference(self.prompt_to_index))
        if missing:
            raise KeyError(
                "FastWAM text context artifact is missing formatted prompt(s): "
                + ", ".join(repr(prompt) for prompt in missing)
            )

    def lookup(
        self,
        prompts: str | Sequence[str],
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve a scalar or batch of formatted prompts without text-model construction."""
        values = _normalize_prompts(prompts)
        self.validate_prompts(values)
        indices = torch.tensor([self.prompt_to_index[prompt] for prompt in values], dtype=torch.long)
        context = self.context.index_select(0, indices)
        mask = self.context_mask.index_select(0, indices)
        if device is not None:
            context = context.to(device=device)
            mask = mask.to(device=device)
        if dtype is not None:
            context = context.to(dtype=dtype)
        return context, mask


def load_text_context_artifact(
    path: str | Path,
    *,
    expected_provenance: TextContextProvenance | Mapping[str, Any] | None = None,
) -> TextContextArtifact:
    """Load and validate a safe tensor artifact plus its required JSON sidecar."""
    path = Path(path).expanduser()
    sidecar = _sidecar_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FastWAM text context artifact does not exist: {path}")
    if not sidecar.is_file():
        raise FileNotFoundError(f"FastWAM text context provenance sidecar does not exist: {sidecar}")
    metadata = json.loads(sidecar.read_text())
    if metadata.get("format") != "fastwam_text_context":
        raise ValueError(f"Unsupported FastWAM text context artifact format in {sidecar}.")
    if int(metadata.get("format_version", -1)) != TEXT_CONTEXT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported FastWAM text context artifact version {metadata.get('format_version')!r}; "
            f"expected {TEXT_CONTEXT_FORMAT_VERSION}."
        )
    expected_sha256 = metadata.get("tensor_file_sha256")
    actual_sha256 = _sha256_file(path)
    if expected_sha256 != actual_sha256:
        raise ValueError(
            f"FastWAM text context artifact hash mismatch for {path}: "
            f"sidecar={expected_sha256!r}, actual={actual_sha256!r}."
        )
    provenance = TextContextProvenance.from_dict(metadata.get("provenance", {}))
    if expected_provenance is not None:
        expected = (
            expected_provenance
            if isinstance(expected_provenance, TextContextProvenance)
            else TextContextProvenance.from_dict(expected_provenance)
        )
        mismatches = {
            key: (getattr(provenance, key), getattr(expected, key))
            for key in _PROVENANCE_FIELDS
            if getattr(provenance, key) != getattr(expected, key)
        }
        if mismatches:
            raise ValueError(f"FastWAM text context provenance mismatch: {mismatches}")

    tensors = load_file(str(path), device="cpu")
    missing_keys = {_CONTEXT_TENSOR_KEY, _CONTEXT_MASK_TENSOR_KEY}.difference(tensors)
    if missing_keys:
        raise ValueError(f"FastWAM text context artifact is missing tensor key(s): {sorted(missing_keys)}")
    context = tensors[_CONTEXT_TENSOR_KEY]
    context_mask = tensors[_CONTEXT_MASK_TENSOR_KEY].to(dtype=torch.bool)
    if tuple(context.shape) != tuple(metadata.get("context_shape", ())):
        raise ValueError(f"Context tensor shape disagrees with sidecar: {tuple(context.shape)}")
    if tuple(context_mask.shape) != tuple(metadata.get("context_mask_shape", ())):
        raise ValueError(f"Context mask shape disagrees with sidecar: {tuple(context_mask.shape)}")
    if tuple(context.shape[1:]) != (provenance.context_len, provenance.context_dim):
        raise ValueError(
            f"Context tensor shape {tuple(context.shape)} disagrees with provenance "
            f"({provenance.context_len}, {provenance.context_dim})."
        )
    prompt_to_index = {
        str(prompt): int(index) for prompt, index in metadata.get("prompt_to_index", {}).items()
    }
    if sorted(prompt_to_index.values()) != list(range(context.shape[0])):
        raise ValueError(
            "FastWAM text context prompt mapping must contain exactly one index per context row."
        )
    prompts = [prompt for prompt, _ in sorted(prompt_to_index.items(), key=lambda item: item[1])]
    prompt_hashes = metadata.get("prompt_hashes", {})
    if any(prompt_hashes.get(prompt) != _prompt_hash(prompt) for prompt in prompts):
        raise ValueError("FastWAM text context prompt hash mismatch in provenance sidecar.")
    actual_prompts_sha256 = hashlib.sha256(_canonical_json(prompts).encode("utf-8")).hexdigest()
    if metadata.get("prompts_sha256") != actual_prompts_sha256:
        raise ValueError("FastWAM text context prompt-list hash mismatch in provenance sidecar.")
    return TextContextArtifact(
        path=path,
        context=context,
        context_mask=context_mask,
        prompt_to_index=prompt_to_index,
        provenance=provenance,
    )
