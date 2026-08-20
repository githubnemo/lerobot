from dataclasses import dataclass

import pytest
import torch
from torch import Tensor, nn

from lerobot.policies.vam.action_rmse import (
    EvaluationBatch,
    MeanActionBackend,
    SmolVLABackend,
    StateRepeatBackend,
    VAMBackend,
    aggregate_action_rmse,
    evaluate_backend,
)


def make_batch() -> EvaluationBatch:
    return EvaluationBatch(
        sample_ids=("ep32-frame7",),
        target_actions=torch.ones(1, 30, 6),
        action_is_pad=torch.cat(
            (torch.zeros(1, 2, dtype=torch.bool), torch.ones(1, 28, dtype=torch.bool)), dim=1
        ),
        current_state=torch.zeros(1, 6),
    )


def test_aggregate_matches_hand_computation_and_masks_padding() -> None:
    batch = make_batch()
    prediction = torch.zeros_like(batch.target_actions)
    prediction[:, 2:] = 1000.0
    result = aggregate_action_rmse(prediction, batch.target_actions, batch.action_is_pad)
    assert result["n_valid_scalars"] == 12
    assert result["per_joint_rmse_deg"] == pytest.approx([1.0] * 6)
    assert result["aggregate_rmse_deg"] == pytest.approx(1.0)


def test_mean_action_baseline_repeats_training_mean() -> None:
    batch = make_batch()
    mean_action = torch.arange(6, dtype=torch.float32)
    prediction = MeanActionBackend(mean_action).predict_actions(batch, seed=99)
    assert torch.equal(prediction[:, 0].squeeze(0), mean_action)
    assert torch.equal(prediction[:, -1].squeeze(0), mean_action)


def test_state_repeat_baseline_is_current_state_in_raw_units() -> None:
    batch = make_batch()
    batch.current_state = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    prediction = StateRepeatBackend().predict_actions(batch, seed=99)
    assert prediction.shape == (1, 30, 6)
    assert torch.equal(prediction[:, 0], batch.current_state)
    assert torch.equal(prediction[:, -1], batch.current_state)


class FakeVAMDecoder:
    def sample_actions(self, state: Tensor, context: Tensor, *, seed: int, context_timestep=None) -> Tensor:
        del context, context_timestep
        generator = torch.Generator(device=state.device)
        generator.manual_seed(seed)
        return torch.rand((state.shape[0], 30, 6), generator=generator, device=state.device)


def test_vam_backend_is_deterministic_for_fixed_seed() -> None:
    batch = make_batch()
    batch.contexts = torch.zeros(1, 2, 4, dtype=torch.float32)
    first = evaluate_backend(VAMBackend(FakeVAMDecoder()), [batch], seed=123)
    second = evaluate_backend(VAMBackend(FakeVAMDecoder()), [batch], seed=123)
    assert first == second


@dataclass
class FakeSmolConfig:
    chunk_size: int = 50
    max_action_dim: int = 32


class FakeSmolPolicy(nn.Module):
    config = FakeSmolConfig()

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(()))

    def predict_action_chunk(self, batch: dict, *, noise: Tensor) -> Tensor:
        del batch
        return noise[:, :, :6]

    def reset(self) -> None:
        return None


def test_smol_backend_uses_fixed_noise_and_first_thirty_actions() -> None:
    batch = make_batch()
    batch.observations = ({"task": "take cube out of box"},)
    backend = SmolVLABackend(FakeSmolPolicy())
    first = backend.predict_actions(batch, seed=7)
    second = backend.predict_actions(batch, seed=7)
    assert first.shape == (1, 30, 6)
    assert torch.equal(first, second)


def test_aggregate_rejects_all_padding() -> None:
    batch = make_batch()
    batch.action_is_pad[:] = True
    with pytest.raises(ValueError, match="no valid"):
        aggregate_action_rmse(
            torch.zeros_like(batch.target_actions), batch.target_actions, batch.action_is_pad
        )
