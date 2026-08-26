"""Shared physical-action RMSE protocol and policy backends."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

ACTION_HORIZON = 30
ACTION_DIM = 6


class ActionRMSEError(ValueError):
    """Raised when comparison tensors do not satisfy the frozen protocol."""


@dataclass
class EvaluationBatch:
    """One batch of identical samples presented to every backend."""

    sample_ids: tuple[str, ...]
    target_actions: Tensor
    action_is_pad: Tensor
    current_state: Tensor
    contexts: Tensor | None = None
    observations: tuple[dict[str, Any], ...] | None = None


class ActionPredictionBackend(Protocol):
    """Return raw physical actions in stored units."""

    name: str

    def predict_actions(self, batch: EvaluationBatch, seed: int) -> Tensor:
        """Return a ``[B, 30, 6]`` tensor in raw physical units."""


def _validate_batch(batch: EvaluationBatch) -> None:
    batch_size = len(batch.sample_ids)
    if batch.target_actions.shape != (batch_size, ACTION_HORIZON, ACTION_DIM):
        raise ActionRMSEError("target_actions must have shape [B, 30, 6]")
    if batch.action_is_pad.shape != (batch_size, ACTION_HORIZON):
        raise ActionRMSEError("action_is_pad must have shape [B, 30]")
    if batch.action_is_pad.dtype is not torch.bool:
        raise ActionRMSEError("action_is_pad must have boolean dtype")
    if batch.current_state.shape not in {(batch_size, ACTION_DIM), (batch_size, 1, ACTION_DIM)}:
        raise ActionRMSEError("current_state must have shape [B, 6] or [B, 1, 6]")
    if not torch.isfinite(batch.target_actions).all().item():
        raise ActionRMSEError("target_actions must be finite")


def _validate_prediction(prediction: Tensor, batch_size: int) -> None:
    if prediction.shape != (batch_size, ACTION_HORIZON, ACTION_DIM):
        raise ActionRMSEError("predictions must have shape [B, 30, 6]")
    if not torch.isfinite(prediction).all().item():
        raise ActionRMSEError("predictions must be finite")


def aggregate_action_rmse(prediction: Tensor, target: Tensor, action_is_pad: Tensor) -> dict[str, Any]:
    """Compute masked per-joint and aggregate RMSE in raw physical units."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ActionRMSEError("prediction and target must have the same [B, 30, 6] shape")
    if tuple(prediction.shape[1:]) != (ACTION_HORIZON, ACTION_DIM):
        raise ActionRMSEError("prediction and target must have shape [B, 30, 6]")
    if action_is_pad.shape != prediction.shape[:2] or action_is_pad.dtype is not torch.bool:
        raise ActionRMSEError("action_is_pad must have boolean shape [B, 30]")
    valid = (~action_is_pad).unsqueeze(-1)
    valid_count = int(valid.sum().item())
    valid_scalars = valid_count * ACTION_DIM
    if valid_scalars == 0:
        raise ActionRMSEError("action_is_pad leaves no valid scalar action positions")
    squared_error = (prediction.float() - target.float()).square()
    masked_squared_error = squared_error * valid
    per_joint_rmse = torch.sqrt(masked_squared_error.sum(dim=(0, 1)) / valid_count)
    aggregate_rmse = torch.sqrt(masked_squared_error.sum() / valid_scalars)
    return {
        "per_joint_rmse_deg": [float(value) for value in per_joint_rmse.cpu()],
        "aggregate_rmse_deg": float(aggregate_rmse.cpu().item()),
        "n_valid_action_tokens": valid_count,
        "n_valid_scalars": valid_scalars,
    }


class StateRepeatBackend:
    """Predict the current proprioceptive position at every future step."""

    name = "state_repeat"

    def predict_actions(self, batch: EvaluationBatch, seed: int) -> Tensor:
        del seed
        _validate_batch(batch)
        state = batch.current_state[:, 0] if batch.current_state.ndim == 3 else batch.current_state
        return state[:, None, :].expand(-1, ACTION_HORIZON, -1).clone()


class MeanActionBackend:
    """Predict one fixed training-set mean action for every future step."""

    name = "mean_action"

    def __init__(self, mean_action: Tensor) -> None:
        if mean_action.shape != (ACTION_DIM,) or not torch.isfinite(mean_action).all().item():
            raise ActionRMSEError("mean_action must be finite with shape [6]")
        self.mean_action = mean_action.float()

    def predict_actions(self, batch: EvaluationBatch, seed: int) -> Tensor:
        del seed
        _validate_batch(batch)
        return (
            self.mean_action.to(batch.target_actions.device)[None, None, :]
            .expand(len(batch.sample_ids), ACTION_HORIZON, ACTION_DIM)
            .clone()
        )


class VAMBackend:
    """Adapt a World2Action decoder to the shared raw-action interface."""

    name = "vam"

    def __init__(self, decoder: Any, *, context_timestep: Tensor | None = None) -> None:
        self.decoder = decoder
        self.context_timestep = context_timestep

    def _decoder_device(self, fallback: torch.device) -> torch.device:
        denoiser = getattr(self.decoder, "denoiser", None)
        if denoiser is None:
            return fallback
        return next(denoiser.parameters()).device

    @torch.no_grad()
    def predict_actions(self, batch: EvaluationBatch, seed: int) -> Tensor:
        _validate_batch(batch)
        if batch.contexts is None:
            raise ActionRMSEError("VAMBackend requires batch.contexts")
        state = batch.current_state
        if state.ndim == 2:
            state = state[:, None, :]
        device = self._decoder_device(state.device)
        state = state.to(device=device)
        contexts = batch.contexts.to(device=device)
        predictions = []
        for index in range(len(batch.sample_ids)):
            context_time = None
            if self.context_timestep is not None:
                source = self.context_timestep
                offset = 0 if source.numel() == 1 else index
                context_time = source[offset : offset + 1].to(device=device)
            predictions.append(
                self.decoder.sample_actions(
                    state[index : index + 1],
                    contexts[index : index + 1],
                    seed=seed + index,
                    context_timestep=context_time,
                )
            )
        prediction = torch.cat(predictions, dim=0).cpu()
        _validate_prediction(prediction, len(batch.sample_ids))
        return prediction.float()


class SmolVLABackend:
    """Adapt a standard LeRobot SmolVLA checkpoint to raw physical actions."""

    name = "smolvla"

    def __init__(
        self, policy: Any, preprocessor: Any | None = None, postprocessor: Any | None = None
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    @classmethod
    def from_pretrained(cls, checkpoint: str, *, device: torch.device) -> SmolVLABackend:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor import (
            PolicyProcessorPipeline,
            policy_action_to_transition,
            transition_to_policy_action,
        )

        policy = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint, config_filename="policy_preprocessor.json"
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint,
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        return cls(policy, preprocessor, postprocessor)

    def _device_and_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.policy.parameters())
        # SmolVLA is mixed precision: the VLM may be bfloat16 while the action
        # expert stays float32, so the noise must match the action projection.
        projection = getattr(getattr(self.policy, "model", None), "action_in_proj", None)
        dtype = projection.weight.dtype if projection is not None else parameter.dtype
        return parameter.device, dtype

    @torch.no_grad()
    def predict_actions(self, batch: EvaluationBatch, seed: int) -> Tensor:
        _validate_batch(batch)
        if batch.observations is None or len(batch.observations) != len(batch.sample_ids):
            raise ActionRMSEError("SmolVLABackend requires one raw observation per sample")
        device, dtype = self._device_and_dtype()
        predictions = []
        chunk_size = int(self.policy.config.chunk_size)
        max_action_dim = int(self.policy.config.max_action_dim)
        for index, observation in enumerate(batch.observations):
            processed = self.preprocessor(observation) if self.preprocessor is not None else observation
            processed = {
                key: value.to(device=device) if isinstance(value, Tensor) else value
                for key, value in processed.items()
            }
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + index)
            noise = torch.randn(
                (1, chunk_size, max_action_dim), device=device, dtype=dtype, generator=generator
            )
            if hasattr(self.policy, "reset"):
                self.policy.reset()
            action = self.policy.predict_action_chunk(processed, noise=noise)
            if self.postprocessor is not None:
                action = self.postprocessor(action)
            if isinstance(action, Tensor):
                action = action.cpu()
            if not isinstance(action, Tensor):
                raise ActionRMSEError("SmolVLA postprocessor must return a tensor")
            predictions.append(action[:, :ACTION_HORIZON, :ACTION_DIM].to(dtype=torch.float32))
        prediction = torch.cat(predictions, dim=0)
        _validate_prediction(prediction, len(batch.sample_ids))
        return prediction


def _fastwam_config_from_train_config(checkpoint: Path, device: torch.device) -> Any:
    """Load FastWAM config while honoring the training run's text artifact."""
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig

    train_config_path = checkpoint / "train_config.json"
    if not train_config_path.is_file():
        raise ActionRMSEError(f"FastWAM checkpoint is missing {train_config_path}")
    train_config = json.loads(train_config_path.read_text())
    policy_config = train_config.get("policy")
    if not isinstance(policy_config, dict):
        raise ActionRMSEError(f"FastWAM train config has no policy object: {train_config_path}")
    if policy_config.get("load_text_encoder") is not False:
        raise ActionRMSEError("FastWAM RMSE requires train_config.json policy.load_text_encoder=false")
    text_context_value = policy_config.get("text_context_path")
    if not isinstance(text_context_value, str) or not text_context_value:
        raise ActionRMSEError(
            "FastWAM train config must provide policy.text_context_path when the text encoder is disabled"
        )
    text_context_path = Path(text_context_value).expanduser()
    if not text_context_path.is_file():
        raise ActionRMSEError(f"FastWAM text context artifact does not exist: {text_context_path}")

    config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(config, FastWAMConfig):
        raise ActionRMSEError(
            f"FastWAM checkpoint config has type {type(config).__name__}, expected FastWAMConfig"
        )
    config.load_text_encoder = False
    config.text_context_path = str(text_context_path)
    config.device = str(device)
    return config


class FastWAMBackend:
    """Adapt a native FastWAM checkpoint to the frozen raw-action protocol."""

    name = "fastwam"

    def __init__(
        self, policy: Any, preprocessor: Any | None = None, postprocessor: Any | None = None
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        config = getattr(policy, "config", None)
        action_horizon = getattr(config, "action_horizon", None)
        action_dim = getattr(config, "action_dim", None)
        if action_horizon != 32:
            raise ActionRMSEError(f"FastWAM checkpoint must have action_horizon=32, got {action_horizon}")
        if action_dim != ACTION_DIM:
            raise ActionRMSEError(f"FastWAM checkpoint must have action_dim={ACTION_DIM}, got {action_dim}")

    @classmethod
    def from_pretrained(cls, checkpoint: str, *, device: torch.device) -> FastWAMBackend:
        """Load FastWAM and its checkpoint-native processor pipelines."""
        from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
        from lerobot.processor import (
            PolicyProcessorPipeline,
            policy_action_to_transition,
            transition_to_policy_action,
        )

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        config = _fastwam_config_from_train_config(checkpoint_path, device)
        policy = FastWAMPolicy.from_pretrained(str(checkpoint_path), config=config).to(device).eval()
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint_path, config_filename="policy_preprocessor.json"
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint_path,
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        return cls(policy, preprocessor, postprocessor)

    @torch.no_grad()
    def predict_actions(self, batch: EvaluationBatch, seed: int) -> Tensor:
        _validate_batch(batch)
        if batch.observations is None or len(batch.observations) != len(batch.sample_ids):
            raise ActionRMSEError("FastWAMBackend requires one raw observation per sample")

        predictions = []
        for index, observation in enumerate(batch.observations):
            processed = self.preprocessor(observation) if self.preprocessor is not None else observation
            if not isinstance(processed, dict):
                raise ActionRMSEError("FastWAM preprocessor must return a mapping")
            # FastWAM consumes the seed as an inference kwarg from the processed batch.
            processed = dict(processed)
            processed["seed"] = seed + index
            if hasattr(self.policy, "reset"):
                self.policy.reset()
            action = self.policy.predict_action_chunk(processed)
            if self.postprocessor is not None:
                action = self.postprocessor(action)
            if not isinstance(action, Tensor):
                raise ActionRMSEError("FastWAM postprocessor must return a tensor")
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1:
                raise ActionRMSEError(f"FastWAM action must have shape [1, 32, 6], got {tuple(action.shape)}")
            if action.shape[1:] != (32, ACTION_DIM):
                raise ActionRMSEError(
                    f"FastWAM action must have shape [1, 32, {ACTION_DIM}], got {tuple(action.shape)}"
                )
            # The native action chunk is 32 steps; the protocol scores only its first 30.
            predictions.append(action[:, :ACTION_HORIZON].to(device="cpu", dtype=torch.float32))

        prediction = torch.cat(predictions, dim=0)
        _validate_prediction(prediction, len(batch.sample_ids))
        return prediction


class VLAJEPABackend:
    """Adapt a native VLA-JEPA checkpoint to raw physical actions.

    VLA-JEPA's action head samples its own ``torch.randn`` initial action state
    and currently ignores the public ``noise`` argument.  The evaluator therefore
    forks the torch RNG and seeds it immediately around each policy call, rather
    than attempting to provide a noise tensor that the policy will not consume.
    """

    name = "vla_jepa"
    camera_key = "observation.images.front"

    def __init__(
        self, policy: Any, preprocessor: Any | None = None, postprocessor: Any | None = None
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        config = getattr(policy, "config", None)
        chunk_size = getattr(config, "chunk_size", None)
        try:
            chunk_matches = int(chunk_size) == ACTION_HORIZON
        except (TypeError, ValueError):
            chunk_matches = False
        if not chunk_matches:
            raise ActionRMSEError(
                f"VLA-JEPA checkpoint must have chunk_size={ACTION_HORIZON}, got {chunk_size}"
            )
        action_dim = getattr(config, "action_dim", ACTION_DIM)
        try:
            action_matches = int(action_dim) == ACTION_DIM
        except (TypeError, ValueError):
            action_matches = False
        if not action_matches:
            raise ActionRMSEError(f"VLA-JEPA checkpoint must have action_dim={ACTION_DIM}, got {action_dim}")
        state_dim = getattr(config, "state_dim", ACTION_DIM)
        try:
            state_matches = state_dim is None or int(state_dim) == ACTION_DIM
        except (TypeError, ValueError):
            state_matches = False
        if not state_matches:
            raise ActionRMSEError(f"VLA-JEPA checkpoint must have state_dim={ACTION_DIM}, got {state_dim}")

    @classmethod
    def from_pretrained(cls, checkpoint: str, *, device: torch.device) -> VLAJEPABackend:
        """Load the policy and the checkpoint's own processor pipelines."""
        from lerobot.policies.vla_jepa import VLAJEPAPolicy
        from lerobot.processor import (
            PolicyProcessorPipeline,
            policy_action_to_transition,
            transition_to_policy_action,
        )

        policy = VLAJEPAPolicy.from_pretrained(checkpoint).to(device).eval()
        # Native VLA-JEPA returns actions via config.device; keep that setting in
        # sync when evaluating a checkpoint saved with a different device.
        policy.config.device = str(device)
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint, config_filename="policy_preprocessor.json"
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint,
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        return cls(policy, preprocessor, postprocessor)

    def _device(self, fallback: torch.device) -> torch.device:
        try:
            return next(self.policy.parameters()).device
        except (AttributeError, StopIteration):
            return fallback

    @staticmethod
    def _seed_torch(seed: int, device: torch.device) -> None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    @torch.no_grad()
    def predict_actions(self, batch: EvaluationBatch, seed: int) -> Tensor:
        _validate_batch(batch)
        if batch.observations is None or len(batch.observations) != len(batch.sample_ids):
            raise ActionRMSEError("VLAJEPABackend requires one raw observation per sample")

        device = self._device(batch.target_actions.device)
        predictions = []
        for index, observation in enumerate(batch.observations):
            if self.camera_key not in observation:
                raise ActionRMSEError(f"VLA-JEPA observation is missing `{self.camera_key}`")
            processed = self.preprocessor(observation) if self.preprocessor is not None else observation
            processed = {
                key: value.to(device=device) if isinstance(value, Tensor) else value
                for key, value in processed.items()
            }

            # The action head samples torch.randn internally; the supplied noise
            # parameter is ignored by VLAJEPAPolicy.predict_action_chunk.  Keep
            # all RNG state isolated so each anchor has a reproducible seed.
            cuda_devices = []
            if device.type == "cuda":
                cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
            with torch.random.fork_rng(devices=cuda_devices):
                self._seed_torch(seed + index, device)
                if hasattr(self.policy, "reset"):
                    self.policy.reset()
                action = self.policy.predict_action_chunk(processed)
            if self.postprocessor is not None:
                action = self.postprocessor(action)
            if not isinstance(action, Tensor):
                raise ActionRMSEError("VLA-JEPA postprocessor must return a tensor")
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1:
                raise ActionRMSEError(
                    f"VLA-JEPA action must have shape [1, 30, 6], got {tuple(action.shape)}"
                )
            predictions.append(action[:, :ACTION_HORIZON, :ACTION_DIM].to(device="cpu", dtype=torch.float32))

        prediction = torch.cat(predictions, dim=0)
        _validate_prediction(prediction, len(batch.sample_ids))
        return prediction


def evaluate_backend(
    backend: ActionPredictionBackend, batches: list[EvaluationBatch], *, seed: int
) -> dict[str, Any]:
    """Aggregate squared errors globally; never average batch RMSEs."""
    if not batches:
        raise ActionRMSEError("at least one evaluation batch is required")
    squared_by_joint = torch.zeros(ACTION_DIM, dtype=torch.float64)
    prefix_horizons = (1, 10)
    prefix_squared_by_joint = {
        horizon: torch.zeros(ACTION_DIM, dtype=torch.float64) for horizon in prefix_horizons
    }
    prefix_valid_action_tokens = dict.fromkeys(prefix_horizons, 0)
    timestep_squared = torch.zeros(ACTION_HORIZON, dtype=torch.float64)
    timestep_valid_scalars = torch.zeros(ACTION_HORIZON, dtype=torch.int64)
    valid_action_tokens = 0
    sample_ids: list[str] = []
    for batch_index, batch in enumerate(batches):
        _validate_batch(batch)
        prediction = backend.predict_actions(batch, seed + batch_index * 100_000)
        _validate_prediction(prediction, len(batch.sample_ids))
        valid = (~batch.action_is_pad).unsqueeze(-1)
        squared_error = (prediction.cpu().double() - batch.target_actions.cpu().double()).square()
        squared_by_joint += (squared_error * valid).sum(dim=(0, 1))
        valid_action_tokens += int(valid.sum().item())
        for horizon in prefix_horizons:
            prefix_valid = valid[:, :horizon]
            prefix_squared_by_joint[horizon] += (squared_error[:, :horizon] * prefix_valid).sum(dim=(0, 1))
            prefix_valid_action_tokens[horizon] += int(prefix_valid.sum().item())
        timestep_squared += (squared_error * valid).sum(dim=(0, 2))
        timestep_valid_scalars += valid.squeeze(-1).sum(dim=0).to(torch.int64) * ACTION_DIM
        sample_ids.extend(batch.sample_ids)
    if valid_action_tokens == 0:
        raise ActionRMSEError("evaluation contains no valid action tokens")
    if any(prefix_valid_action_tokens[horizon] == 0 for horizon in prefix_horizons):
        raise ActionRMSEError("evaluation contains no valid executed-prefix action tokens")
    per_joint = torch.sqrt(squared_by_joint / valid_action_tokens)
    executed_prefix = {}
    for horizon in prefix_horizons:
        valid_tokens = prefix_valid_action_tokens[horizon]
        prefix_per_joint = torch.sqrt(prefix_squared_by_joint[horizon] / valid_tokens)
        valid_timesteps = timestep_valid_scalars[:horizon] > 0
        mean_timestep_rmse = torch.sqrt(
            timestep_squared[:horizon][valid_timesteps] / timestep_valid_scalars[:horizon][valid_timesteps]
        ).mean()
        executed_prefix[f"first_{horizon}"] = {
            "n_valid_action_tokens": valid_tokens,
            "per_joint_rmse_mixed_units": [float(value) for value in prefix_per_joint],
            "aggregate_rmse_mixed_units": float(
                torch.sqrt(prefix_squared_by_joint[horizon].sum() / (valid_tokens * ACTION_DIM))
            ),
            "mean_per_timestep_rmse_mixed_units": float(mean_timestep_rmse),
        }
    return {
        "backend": backend.name,
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "n_valid_action_tokens": valid_action_tokens,
        "n_valid_scalars": valid_action_tokens * ACTION_DIM,
        "per_joint_rmse_deg": [float(value) for value in per_joint],
        "per_joint_units": ["degrees", "degrees", "degrees", "degrees", "degrees", "range_0_100"],
        "aggregate_rmse_deg": float(torch.sqrt(squared_by_joint.sum() / (valid_action_tokens * ACTION_DIM))),
        "executed_prefix": executed_prefix,
        "aggregation": "sqrt(sum(masked_squared_error) / n_valid_scalars)",
        "seed": seed,
    }
