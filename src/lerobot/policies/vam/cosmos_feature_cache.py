"""Strict safetensors cache artifacts for frozen Cosmos representations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

CACHE_SCHEMA_VERSION = 2
CONTEXT_KEY = "context"
STATE_KEY = "state"
TARGET_ACTION_KEY = "target_action"
ACTION_IS_PAD_KEY = "action_is_pad"
_CACHE_KEYS = frozenset({CONTEXT_KEY, STATE_KEY, TARGET_ACTION_KEY, ACTION_IS_PAD_KEY})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TE_EXTRA_STATE_KEY = re.compile(r"^blocks\.\d+\.cross_attn\.attn_op\._extra_state$")

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "dataset",
        "source_shapes",
        "weights",
        "prompt_embedding",
        "extractor",
        "upstream_commits",
        "runtime",
        "tensor_roles",
        "output",
    }
)
_SOURCE_SHAPE_KEYS = frozenset(
    {
        "sample_camera",
        "sample_state",
        "sample_action",
        "rgb_history",
        "prompt_embedding",
        "raw_hidden",
        "context",
        "state",
        "target_action",
        "action_is_pad",
    }
)
_WEIGHT_KEYS = frozenset(
    {
        "checkpoint_path",
        "checkpoint_size_bytes",
        "checkpoint_sha256",
        "tokenizer_path",
        "tokenizer_size_bytes",
        "tokenizer_sha256",
        "checkpoint_kind",
        "bridge_lora",
    }
)
_PROMPT_KEYS = frozenset({"artifact_path", "output_sha256", "token_ids_sha256", "shape", "dtype"})
_EXTRACTOR_KEYS = frozenset(
    {
        "device",
        "dtype",
        "backend",
        "high_noise_sigma",
        "seed",
        "hidden_layer",
        "stop_after_step",
        "input_shape",
        "preprocess",
        "conditioning",
        "official_resolution",
        "official_positional_latent_max_h",
        "official_positional_latent_max_w",
        "bridge_lora",
        "extractor_input_keys",
        "excluded_from_extractor",
        "checkpoint_ignored_metadata_keys",
        "checkpoint_ignored_metadata_count",
        "noise_seed",
    }
)
_RUNTIME_KEYS = frozenset(
    {
        "load_seconds",
        "prompt_load_seconds",
        "extraction_seconds",
        "load_peak_allocated_bytes",
        "load_peak_reserved_bytes",
        "extraction_peak_allocated_bytes",
        "extraction_peak_reserved_bytes",
        "torch_version",
        "cuda_version",
        "gpu_name",
        "python_version",
    }
)
_ROLE_KEYS = frozenset({CONTEXT_KEY, STATE_KEY, TARGET_ACTION_KEY})
_OUTPUT_KEYS = frozenset(
    {
        "raw_hidden_shape",
        "raw_hidden_dtype",
        "context_shape",
        "context_dtype",
        "context_sha256",
        "state_shape",
        "state_dtype",
        "state_sha256",
        "target_action_shape",
        "target_action_dtype",
        "target_action_sha256",
        "action_is_pad_shape",
        "action_is_pad_dtype",
        "action_is_pad_sha256",
    }
)


class CosmosFeatureCacheError(RuntimeError):
    """Base error for a Cosmos feature-cache artifact."""


class CosmosFeatureCacheValidationError(CosmosFeatureCacheError, ValueError):
    """Raised when cache tensors or provenance violate the frozen contract."""


@dataclass(frozen=True, slots=True)
class CosmosFeatureCacheProvenance:
    """Strict JSON provenance attached to one frozen feature cache."""

    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe copy of the validated provenance."""
        return json.loads(json.dumps(self.payload))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CosmosFeatureCacheProvenance:
        """Strictly validate and wrap a provenance mapping."""
        validated = _validate_provenance(payload)
        return cls(validated)


@dataclass(frozen=True, slots=True)
class CosmosFeatureCacheArtifact:
    """Detached representation, decoder state, and explicitly labelled action label."""

    context: torch.Tensor
    state: torch.Tensor
    target_action: torch.Tensor
    action_is_pad: torch.Tensor
    provenance: CosmosFeatureCacheProvenance


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a tensor's contiguous native byte representation."""
    value = tensor.detach().cpu().contiguous()
    raw_bytes = value.view(torch.uint8).reshape(-1).numpy().tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CosmosFeatureCacheValidationError(f"{name} must be a lowercase SHA256")
    return value


def _strict_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise CosmosFeatureCacheValidationError(
            f"{name} keys must match exactly; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _shape(value: Any, name: str, *, rank: int | None = None) -> list[int]:
    if (
        not isinstance(value, list)
        or any(type(dimension) is not int or dimension <= 0 for dimension in value)
        or (rank is not None and len(value) != rank)
    ):
        raise CosmosFeatureCacheValidationError(f"{name} must be a list of positive integer dimensions")
    return value


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CosmosFeatureCacheValidationError(f"{name} must be a non-empty string")
    return value


def _non_negative_number(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise CosmosFeatureCacheValidationError(f"{name} must be non-negative numeric")
    return float(value)


def _validate_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise CosmosFeatureCacheValidationError("Provenance JSON must contain an object")
    _strict_keys(payload, _TOP_LEVEL_KEYS, "provenance")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != CACHE_SCHEMA_VERSION:
        raise CosmosFeatureCacheValidationError("Unsupported feature-cache schema_version")

    dataset = payload["dataset"]
    if not isinstance(dataset, Mapping):
        raise CosmosFeatureCacheValidationError("dataset provenance must be an object")
    _strict_keys(
        dataset,
        frozenset(
            {"repo_id", "revision", "episode_index", "frame_index", "window_indices", "window_offsets", "fps"}
        ),
        "dataset",
    )
    _non_empty_string(dataset["repo_id"], "dataset.repo_id")
    _non_empty_string(dataset["revision"], "dataset.revision")
    for name in ("episode_index", "frame_index", "fps"):
        if type(dataset[name]) is not int or dataset[name] < 0:
            raise CosmosFeatureCacheValidationError(f"dataset.{name} must be a non-negative integer")
    if (
        not isinstance(dataset["window_indices"], list)
        or len(dataset["window_indices"]) != 5
        or any(type(index) is not int or index < 0 for index in dataset["window_indices"])
    ):
        raise CosmosFeatureCacheValidationError(
            "dataset.window_indices must contain five non-negative integers"
        )
    if dataset["window_offsets"] != [-4, -3, -2, -1, 0]:
        raise CosmosFeatureCacheValidationError(
            "dataset.window_offsets must be the causal [-4, -3, -2, -1, 0]"
        )

    source_shapes = payload["source_shapes"]
    if not isinstance(source_shapes, Mapping):
        raise CosmosFeatureCacheValidationError("source_shapes provenance must be an object")
    _strict_keys(source_shapes, _SOURCE_SHAPE_KEYS, "source_shapes")
    for name, value in source_shapes.items():
        _shape(value, f"source_shapes.{name}")

    weights = payload["weights"]
    if not isinstance(weights, Mapping):
        raise CosmosFeatureCacheValidationError("weights provenance must be an object")
    _strict_keys(weights, _WEIGHT_KEYS, "weights")
    for name in ("checkpoint_path", "tokenizer_path", "checkpoint_kind"):
        _non_empty_string(weights[name], f"weights.{name}")
    for name in ("checkpoint_size_bytes", "tokenizer_size_bytes"):
        if type(weights[name]) is not int or weights[name] <= 0:
            raise CosmosFeatureCacheValidationError(f"weights.{name} must be a positive integer")
    _sha256(weights["checkpoint_sha256"], "weights.checkpoint_sha256")
    _sha256(weights["tokenizer_sha256"], "weights.tokenizer_sha256")
    if weights["checkpoint_kind"] not in {"generic_cosmos", "bridge_fused"}:
        raise CosmosFeatureCacheValidationError("weights.checkpoint_kind is unsupported")
    if weights["bridge_lora"] is not None and not isinstance(weights["bridge_lora"], Mapping):
        raise CosmosFeatureCacheValidationError("weights.bridge_lora must be null or an object")

    prompt = payload["prompt_embedding"]
    if not isinstance(prompt, Mapping):
        raise CosmosFeatureCacheValidationError("prompt_embedding provenance must be an object")
    _strict_keys(prompt, _PROMPT_KEYS, "prompt_embedding")
    _non_empty_string(prompt["artifact_path"], "prompt_embedding.artifact_path")
    _sha256(prompt["output_sha256"], "prompt_embedding.output_sha256")
    _sha256(prompt["token_ids_sha256"], "prompt_embedding.token_ids_sha256")
    if _shape(prompt["shape"], "prompt_embedding.shape", rank=3) != [1, 512, 1024]:
        raise CosmosFeatureCacheValidationError("prompt_embedding.shape must be [1, 512, 1024]")
    if prompt["dtype"] != "bfloat16":
        raise CosmosFeatureCacheValidationError("prompt_embedding.dtype must be bfloat16")

    extractor = payload["extractor"]
    if not isinstance(extractor, Mapping):
        raise CosmosFeatureCacheValidationError("extractor provenance must be an object")
    extractor_keys = set(extractor)
    if extractor_keys not in {_EXTRACTOR_KEYS, _EXTRACTOR_KEYS | {"random_init_seed"}}:
        raise CosmosFeatureCacheValidationError("extractor provenance keys are malformed")
    for name in ("device", "dtype", "backend", "preprocess", "conditioning", "official_resolution"):
        _non_empty_string(extractor[name], f"extractor.{name}")
    if extractor["dtype"] != "bfloat16" or extractor["backend"] != "minimal_a2a":
        raise CosmosFeatureCacheValidationError(
            "extractor dtype/backend are not the approved frozen settings"
        )
    random_init_seed = extractor.get("random_init_seed")
    if random_init_seed is not None and (type(random_init_seed) is not int or random_init_seed < 0):
        raise CosmosFeatureCacheValidationError("extractor.random_init_seed must be null or non-negative")
    for name in ("seed", "noise_seed"):
        if type(extractor[name]) is not int or extractor[name] < 0:
            raise CosmosFeatureCacheValidationError(f"extractor.{name} must be a non-negative integer")
    sigma = extractor["high_noise_sigma"]
    if not isinstance(sigma, (int, float)) or isinstance(sigma, bool) or not math.isfinite(float(sigma)):
        raise CosmosFeatureCacheValidationError("extractor.high_noise_sigma must be a finite number")
    if float(sigma) <= 0.0 or extractor["hidden_layer"] != 20 or extractor["stop_after_step"] != 0:
        raise CosmosFeatureCacheValidationError("extractor sigma/layer/step do not match the frozen contract")
    _shape(extractor["input_shape"], "extractor.input_shape", rank=5)
    for name in ("official_positional_latent_max_h", "official_positional_latent_max_w"):
        if type(extractor[name]) is not int or extractor[name] <= 0:
            raise CosmosFeatureCacheValidationError(f"extractor.{name} must be a positive integer")
    if extractor["bridge_lora"] is not None and not isinstance(extractor["bridge_lora"], Mapping):
        raise CosmosFeatureCacheValidationError("extractor.bridge_lora must be null or an object")
    if extractor["extractor_input_keys"] != ["rgb_history", "prompt_embedding"]:
        raise CosmosFeatureCacheValidationError("extractor input keys must exclude labels and state")
    if extractor["excluded_from_extractor"] != ["state", "target_action"]:
        raise CosmosFeatureCacheValidationError(
            "state/action must be explicitly excluded from extractor inputs"
        )
    ignored_keys = extractor["checkpoint_ignored_metadata_keys"]
    if (
        not isinstance(ignored_keys, list)
        or any(not isinstance(key, str) or _TE_EXTRA_STATE_KEY.fullmatch(key) is None for key in ignored_keys)
        or ignored_keys != sorted(set(ignored_keys))
    ):
        raise CosmosFeatureCacheValidationError(
            "checkpoint ignored metadata keys are not the exact sorted TE allowlist"
        )
    if type(extractor["checkpoint_ignored_metadata_count"]) is not int or extractor[
        "checkpoint_ignored_metadata_count"
    ] != len(ignored_keys):
        raise CosmosFeatureCacheValidationError(
            "checkpoint ignored metadata count does not match its key list"
        )

    commits = payload["upstream_commits"]
    if not isinstance(commits, Mapping):
        raise CosmosFeatureCacheValidationError("upstream_commits provenance must be an object")
    _strict_keys(commits, frozenset({"lerobot", "mimic_video", "vendor_manifest"}), "upstream_commits")
    for name, value in commits.items():
        _non_empty_string(value, f"upstream_commits.{name}")

    runtime = payload["runtime"]
    if not isinstance(runtime, Mapping):
        raise CosmosFeatureCacheValidationError("runtime provenance must be an object")
    _strict_keys(runtime, _RUNTIME_KEYS, "runtime")
    for name in ("load_seconds", "prompt_load_seconds", "extraction_seconds"):
        _non_negative_number(runtime[name], f"runtime.{name}")
    for name in (
        "load_peak_allocated_bytes",
        "load_peak_reserved_bytes",
        "extraction_peak_allocated_bytes",
        "extraction_peak_reserved_bytes",
    ):
        if type(runtime[name]) is not int or runtime[name] < 0:
            raise CosmosFeatureCacheValidationError(f"runtime.{name} must be a non-negative integer")
    for name in ("torch_version", "cuda_version", "gpu_name", "python_version"):
        _non_empty_string(runtime[name], f"runtime.{name}")

    roles = payload["tensor_roles"]
    if not isinstance(roles, Mapping):
        raise CosmosFeatureCacheValidationError("tensor_roles provenance must be an object")
    _strict_keys(roles, _ROLE_KEYS, "tensor_roles")
    expected_roles = {
        CONTEXT_KEY: "representation",
        STATE_KEY: "action_decoder_conditioning_not_extractor_input",
        TARGET_ACTION_KEY: "label_only_not_extractor_input",
    }
    if dict(roles) != expected_roles:
        raise CosmosFeatureCacheValidationError(
            "tensor_roles must distinguish representation, state, and action label"
        )

    output = payload["output"]
    if not isinstance(output, Mapping):
        raise CosmosFeatureCacheValidationError("output provenance must be an object")
    _strict_keys(output, _OUTPUT_KEYS, "output")
    for name in ("raw_hidden_shape", "context_shape", "state_shape", "target_action_shape"):
        _shape(output[name], f"output.{name}")
    for name in ("raw_hidden_dtype", "context_dtype", "state_dtype", "target_action_dtype"):
        _non_empty_string(output[name], f"output.{name}")
    for name in ("context_sha256", "state_sha256", "target_action_sha256"):
        _sha256(output[name], f"output.{name}")
    return json.loads(json.dumps(payload))


def derive_window_seed(dataset_revision: str, episode_index: int, frame_index: int, global_seed: int) -> int:
    """Derive a stable NumPy-compatible noise seed for one causal cache window."""
    if not isinstance(dataset_revision, str) or not dataset_revision:
        raise ValueError("dataset_revision must be a non-empty string")
    if any(type(value) is not int or value < 0 for value in (episode_index, frame_index, global_seed)):
        raise ValueError("episode_index, frame_index, and global_seed must be non-negative integers")
    material = f"{dataset_revision}\0{episode_index}\0{frame_index}\0{global_seed}".encode()
    # arch_invariant_rand delegates to NumPy RandomState, whose accepted seed
    # range is [0, 2**32 - 2].
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**32 - 1)


def validate_cache_tensors(
    context: torch.Tensor,
    state: torch.Tensor,
    target_action: torch.Tensor,
    action_is_pad: torch.Tensor | None = None,
    verify_values: bool = True,
) -> None:
    """Validate cache tensors; finite-value scanning can be reused from first verification."""
    if not isinstance(context, torch.Tensor) or context.ndim != 3 or context.shape[-1] != 2048:
        raise CosmosFeatureCacheValidationError("context must have shape [B, N, 2048]")
    if context.dtype != torch.bfloat16:
        raise CosmosFeatureCacheValidationError("context must have dtype torch.bfloat16")
    if context.requires_grad:
        raise CosmosFeatureCacheValidationError("context must be detached before caching")
    if verify_values and not torch.isfinite(context).all().item():
        raise CosmosFeatureCacheValidationError("context must contain only finite values")
    if not isinstance(state, torch.Tensor) or state.ndim != 3 or tuple(state.shape[1:]) != (1, 6):
        raise CosmosFeatureCacheValidationError("state must have shape [B, 1, 6]")
    if (
        not isinstance(target_action, torch.Tensor)
        or target_action.ndim != 3
        or tuple(target_action.shape[1:]) != (30, 6)
    ):
        raise CosmosFeatureCacheValidationError("target_action must have shape [B, 30, 6]")
    if action_is_pad is None:
        raise CosmosFeatureCacheValidationError("action_is_pad is required by feature-cache schema v2")
    if (
        not isinstance(action_is_pad, torch.Tensor)
        or action_is_pad.ndim != 2
        or tuple(action_is_pad.shape[1:]) != (30,)
        or action_is_pad.dtype != torch.bool
    ):
        raise CosmosFeatureCacheValidationError("action_is_pad must have boolean shape [B, 30]")
    if action_is_pad.shape[0] != context.shape[0]:
        raise CosmosFeatureCacheValidationError("action_is_pad batch dimension must match cached tensors")
    if bool(action_is_pad.all().item()):
        raise CosmosFeatureCacheValidationError("action_is_pad cannot mark every action token as padded")
    for name, tensor in (("state", state), ("target_action", target_action)):
        if not torch.is_floating_point(tensor):
            raise CosmosFeatureCacheValidationError(f"{name} must be floating point")
        if verify_values and not torch.isfinite(tensor).all().item():
            raise CosmosFeatureCacheValidationError(f"{name} must contain only finite values")
        if tensor.requires_grad:
            raise CosmosFeatureCacheValidationError(f"{name} must be detached before caching")


def build_feature_cache_provenance(
    *,
    dataset: Mapping[str, Any],
    source_shapes: Mapping[str, list[int]],
    weights: Mapping[str, Any],
    prompt_embedding: Mapping[str, Any],
    extractor: Mapping[str, Any],
    upstream_commits: Mapping[str, str],
    runtime: Mapping[str, Any],
    context: torch.Tensor,
    state: torch.Tensor,
    target_action: torch.Tensor,
    action_is_pad: torch.Tensor,
    raw_hidden_shape: tuple[int, ...] | list[int],
    raw_hidden_dtype: torch.dtype,
) -> CosmosFeatureCacheProvenance:
    """Build output/hash provenance after validating the candidate tensors."""
    validate_cache_tensors(context, state, target_action, action_is_pad)
    source = dict(source_shapes)
    source.setdefault("context", list(context.shape))
    source.setdefault("state", list(state.shape))
    source.setdefault("target_action", list(target_action.shape))
    source.setdefault("action_is_pad", list(action_is_pad.shape))
    output = {
        "raw_hidden_shape": list(raw_hidden_shape),
        "raw_hidden_dtype": str(raw_hidden_dtype).removeprefix("torch."),
        "context_shape": list(context.shape),
        "context_dtype": str(context.dtype).removeprefix("torch."),
        "context_sha256": tensor_sha256(context),
        "state_shape": list(state.shape),
        "state_dtype": str(state.dtype).removeprefix("torch."),
        "state_sha256": tensor_sha256(state),
        "target_action_shape": list(target_action.shape),
        "target_action_dtype": str(target_action.dtype).removeprefix("torch."),
        "target_action_sha256": tensor_sha256(target_action),
        "action_is_pad_shape": list(action_is_pad.shape),
        "action_is_pad_dtype": str(action_is_pad.dtype).removeprefix("torch."),
        "action_is_pad_sha256": tensor_sha256(action_is_pad),
    }
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": dict(dataset),
        "source_shapes": source,
        "weights": dict(weights),
        "prompt_embedding": dict(prompt_embedding),
        "extractor": dict(extractor),
        "upstream_commits": dict(upstream_commits),
        "runtime": dict(runtime),
        "tensor_roles": {
            CONTEXT_KEY: "representation",
            STATE_KEY: "action_decoder_conditioning_not_extractor_input",
            TARGET_ACTION_KEY: "label_only_not_extractor_input",
        },
        "output": output,
    }
    return CosmosFeatureCacheProvenance.from_dict(payload)


def _sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(".json")


def _normalized_artifact(
    artifact: CosmosFeatureCacheArtifact, *, verify_hashes: bool = True
) -> CosmosFeatureCacheArtifact:
    context = artifact.context.detach().cpu().contiguous()
    state = artifact.state.detach().cpu().contiguous()
    target_action = artifact.target_action.detach().cpu().contiguous()
    action_is_pad = artifact.action_is_pad.detach().cpu().contiguous()
    validate_cache_tensors(context, state, target_action, action_is_pad, verify_values=verify_hashes)
    provenance = CosmosFeatureCacheProvenance.from_dict(artifact.provenance.to_dict())
    output = provenance.payload["output"]
    expected = {
        "context_shape": list(context.shape),
        "context_dtype": str(context.dtype).removeprefix("torch."),
        "state_shape": list(state.shape),
        "state_dtype": str(state.dtype).removeprefix("torch."),
        "target_action_shape": list(target_action.shape),
        "target_action_dtype": str(target_action.dtype).removeprefix("torch."),
        "action_is_pad_shape": list(action_is_pad.shape),
        "action_is_pad_dtype": str(action_is_pad.dtype).removeprefix("torch."),
    }
    if verify_hashes:
        expected.update(
            {
                "context_sha256": tensor_sha256(context),
                "state_sha256": tensor_sha256(state),
                "target_action_sha256": tensor_sha256(target_action),
                "action_is_pad_sha256": tensor_sha256(action_is_pad),
            }
        )
    for name, value in expected.items():
        if output[name] != value:
            raise CosmosFeatureCacheValidationError(f"Provenance output.{name} does not match cached tensors")
    return CosmosFeatureCacheArtifact(context, state, target_action, action_is_pad, provenance)


def save_feature_cache(
    artifact: CosmosFeatureCacheArtifact,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Save a validated cache as safetensors plus a strict JSON sidecar."""
    output_path = Path(output_path).expanduser()
    if output_path.suffix != ".safetensors":
        raise CosmosFeatureCacheValidationError("Feature cache output must use the .safetensors suffix")
    sidecar = _sidecar_path(output_path)
    if (output_path.exists() or sidecar.exists()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing feature cache; pass --overwrite: {output_path}"
        )
    normalized = _normalized_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_fd, tensor_tmp = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    sidecar_fd, sidecar_tmp = tempfile.mkstemp(prefix=f".{sidecar.name}.", suffix=".tmp", dir=sidecar.parent)
    os.close(tensor_fd)
    try:
        save_file(
            {
                CONTEXT_KEY: normalized.context,
                STATE_KEY: normalized.state,
                TARGET_ACTION_KEY: normalized.target_action,
                ACTION_IS_PAD_KEY: normalized.action_is_pad,
            },
            tensor_tmp,
        )
        with os.fdopen(sidecar_fd, "w") as stream:
            json.dump(normalized.provenance.to_dict(), stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tensor_tmp, output_path)
        os.replace(sidecar_tmp, sidecar)
    except BaseException:
        for temporary in (tensor_tmp, sidecar_tmp):
            with suppress(FileNotFoundError):
                os.unlink(temporary)
        raise
    return output_path, sidecar


def load_feature_cache(output_path: Path, *, verify_hashes: bool = True) -> CosmosFeatureCacheArtifact:
    """Load a cache; hash verification is enabled unless explicitly disabled."""
    output_path = Path(output_path).expanduser()
    sidecar = _sidecar_path(output_path)
    if not output_path.is_file():
        raise FileNotFoundError(f"Feature cache safetensors file not found: {output_path}")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Feature cache provenance sidecar not found: {sidecar}")
    try:
        payload = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CosmosFeatureCacheValidationError(
            f"Could not read feature cache provenance {sidecar}: {exc}"
        ) from exc
    provenance = CosmosFeatureCacheProvenance.from_dict(payload)
    tensors = load_file(str(output_path), device="cpu")
    if set(tensors) != _CACHE_KEYS:
        raise CosmosFeatureCacheValidationError(
            f"Feature cache keys must be exactly {sorted(_CACHE_KEYS)}, got {sorted(tensors)}"
        )
    return _normalized_artifact(
        CosmosFeatureCacheArtifact(
            tensors[CONTEXT_KEY],
            tensors[STATE_KEY],
            tensors[TARGET_ACTION_KEY],
            tensors[ACTION_IS_PAD_KEY],
            provenance,
        ),
        verify_hashes=verify_hashes,
    )


def verify_feature_cache(output_path: Path) -> CosmosFeatureCacheArtifact:
    """Strictly load a cache and return it only after the round-trip checks pass."""
    return load_feature_cache(output_path)


# Explicit aliases make the ownership boundary obvious to callers.
save_cosmos_feature_cache = save_feature_cache
load_cosmos_feature_cache = load_feature_cache
verify_cosmos_feature_cache = verify_feature_cache
