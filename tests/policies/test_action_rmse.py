from dataclasses import dataclass

import pytest
import torch
from torch import Tensor, nn

from lerobot.policies.vam.action_rmse import (
    EvaluationBatch,
    FastWAMBackend,
    MeanActionBackend,
    SmolVLABackend,
    StateRepeatBackend,
    VAMBackend,
    VLAJEPABackend,
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


@dataclass
class FakeFastWAMConfig:
    action_horizon: int = 32
    action_dim: int = 6


class FakeFastWAMPolicy(nn.Module):
    config = FakeFastWAMConfig()

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.chunk = torch.cat(
            (torch.ones(1, 30, 6), torch.full((1, 2, 6), 100.0)),
            dim=1,
        )

    def predict_action_chunk(self, batch: dict) -> Tensor:
        del batch
        return self.chunk + self.weight

    def reset(self) -> None:
        return None


def test_fastwam_backend_slices_native_chunk_and_masks_padding() -> None:
    batch = make_batch()
    batch.observations = ({"task": "take cube out of box"},)
    backend = FastWAMBackend(FakeFastWAMPolicy())

    prediction = backend.predict_actions(batch, seed=7)
    result = evaluate_backend(backend, [batch], seed=7)

    assert prediction.shape == (1, 30, 6)
    assert torch.equal(prediction, torch.ones(1, 30, 6))
    assert result["n_valid_action_tokens"] == 2
    assert result["aggregate_rmse_deg"] == pytest.approx(0.0)


@dataclass
class FakeVLAJEPAConfig:
    chunk_size: int = 30
    action_dim: int = 6


class FakeVLAJEPAActionPolicy(nn.Module):
    config = FakeVLAJEPAConfig()

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def predict_action_chunk(self, batch: dict) -> Tensor:
        del batch
        self.calls += 1
        # This intentionally ignores the optional noise argument, matching the
        # native action head that samples torch.randn internally.
        return torch.randn(1, 30, 6) + self.weight

    def reset(self) -> None:
        return None


def make_vla_batch(batch_size: int = 2) -> EvaluationBatch:
    return EvaluationBatch(
        sample_ids=tuple(f"sample-{index}" for index in range(batch_size)),
        target_actions=torch.zeros(batch_size, 30, 6),
        action_is_pad=torch.zeros(batch_size, 30, dtype=torch.bool),
        current_state=torch.zeros(batch_size, 6),
        observations=tuple(
            {"observation.images.front": torch.zeros(3, 8, 8), "task": "pick up the cube"}
            for _ in range(batch_size)
        ),
    )


def test_vla_jepa_backend_rejects_non_thirty_chunk() -> None:
    policy = FakeVLAJEPAActionPolicy()
    policy.config = FakeVLAJEPAConfig(chunk_size=29)
    with pytest.raises(ValueError, match="chunk_size=30"):
        VLAJEPABackend(policy)


def test_vla_jepa_backend_seeds_each_anchor_deterministically() -> None:
    batch = make_vla_batch()
    policy = FakeVLAJEPAActionPolicy()
    backend = VLAJEPABackend(policy)

    first = backend.predict_actions(batch, seed=123)
    second = backend.predict_actions(batch, seed=123)

    assert first.shape == (2, 30, 6)
    assert torch.equal(first, second)
    assert policy.calls == 4  # one native policy call per anchor and evaluation pass


def test_vla_jepa_backend_returns_postprocessed_physical_actions() -> None:
    batch = make_vla_batch(batch_size=1)
    raw_backend = VLAJEPABackend(FakeVLAJEPAActionPolicy())
    raw = raw_backend.predict_actions(batch, seed=9)

    physical_backend = VLAJEPABackend(
        FakeVLAJEPAActionPolicy(),
        postprocessor=lambda action: action * 2.0 + 7.0,
    )
    physical = physical_backend.predict_actions(batch, seed=9)

    assert torch.allclose(physical, raw * 2.0 + 7.0)
    assert physical.dtype == torch.float32


def test_aggregate_rejects_all_padding() -> None:
    batch = make_batch()
    batch.action_is_pad[:] = True
    with pytest.raises(ValueError, match="no valid"):
        aggregate_action_rmse(
            torch.zeros_like(batch.target_actions), batch.target_actions, batch.action_is_pad
        )
