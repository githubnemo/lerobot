"""Shared physical-action RMSE protocol and policy backends."""

from __future__ import annotations

from dataclasses import dataclass
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


def evaluate_backend(
    backend: ActionPredictionBackend, batches: list[EvaluationBatch], *, seed: int
) -> dict[str, Any]:
    """Aggregate squared errors globally; never average batch RMSEs."""
    if not batches:
        raise ActionRMSEError("at least one evaluation batch is required")
    squared_by_joint = torch.zeros(ACTION_DIM, dtype=torch.float64)
    valid_action_tokens = 0
    sample_ids: list[str] = []
    for batch_index, batch in enumerate(batches):
        _validate_batch(batch)
        prediction = backend.predict_actions(batch, seed + batch_index * 100_000)
        _validate_prediction(prediction, len(batch.sample_ids))
        valid = (~batch.action_is_pad).unsqueeze(-1)
        squared_by_joint += (
            (prediction.cpu().double() - batch.target_actions.cpu().double()).square() * valid
        ).sum(dim=(0, 1))
        valid_action_tokens += int(valid.sum().item())
        sample_ids.extend(batch.sample_ids)
    if valid_action_tokens == 0:
        raise ActionRMSEError("evaluation contains no valid action tokens")
    per_joint = torch.sqrt(squared_by_joint / valid_action_tokens)
    return {
        "backend": backend.name,
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "n_valid_action_tokens": valid_action_tokens,
        "n_valid_scalars": valid_action_tokens * ACTION_DIM,
        "per_joint_rmse_deg": [float(value) for value in per_joint],
        "aggregate_rmse_deg": float(torch.sqrt(squared_by_joint.sum() / (valid_action_tokens * ACTION_DIM))),
        "aggregation": "sqrt(sum(masked_squared_error) / n_valid_scalars)",
        "seed": seed,
    }
