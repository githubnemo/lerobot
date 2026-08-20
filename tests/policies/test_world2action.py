import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

import lerobot.policies.vam.world2action as world2action_module
from lerobot.policies.vam.world2action import (
    ActionStateNormalizer,
    NormalizationArtifact,
    World2ActionConfig,
    World2ActionDecoder,
    World2ActionValidationError,
)


class FakeDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros_like(kwargs["xt_B_HA_A"]) + self.scale * 0


def make_decoder(*, normalizer=None):
    fake = FakeDenoiser()
    return World2ActionDecoder(World2ActionConfig(device="cpu"), denoiser=fake, normalizer=normalizer), fake


class ShapeDenoiser(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, **kwargs):
        return self.output.to(device=kwargs["xt_B_HA_A"].device, dtype=kwargs["xt_B_HA_A"].dtype)


def call_shape_denoiser(output):
    decoder = World2ActionDecoder(
        World2ActionConfig(device="cpu"),
        denoiser=ShapeDenoiser(output),
    )
    state = torch.zeros(1, 1, 6)
    action_input = torch.zeros(1, 30, 6)
    context = torch.zeros(1, 2, 2048, dtype=torch.bfloat16)
    return decoder._call_denoiser(
        state,
        action_input,
        torch.zeros(1),
        torch.full((1, 1), 10.0),
        context,
        obs_dropout=0.0,
    )


def test_native_style_output_discards_only_state_token():
    actions = torch.arange(180, dtype=torch.float32).reshape(1, 30, 6)
    native_output = torch.cat((torch.full((1, 1, 6), -99.0), actions), dim=1)

    actual = call_shape_denoiser(native_output)

    assert actual.shape == (1, 30, 6)
    assert torch.equal(actual, actions)


class TupleShapeDenoiser(ShapeDenoiser):
    def forward(self, **kwargs):
        return super().forward(**kwargs), [torch.ones(1)]


def test_native_tuple_output_slices_prediction_and_ignores_hidden_states():
    actions = torch.arange(180, dtype=torch.float32).reshape(1, 30, 6)
    native_output = torch.cat((torch.full((1, 1, 6), -99.0), actions), dim=1)
    decoder = World2ActionDecoder(
        World2ActionConfig(device="cpu"),
        denoiser=TupleShapeDenoiser(native_output),
    )

    actual = decoder._call_denoiser(
        torch.zeros(1, 1, 6),
        torch.zeros(1, 30, 6),
        torch.zeros(1),
        torch.full((1, 1), 10.0),
        torch.zeros(1, 2, 2048, dtype=torch.bfloat16),
        obs_dropout=0.0,
    )

    assert torch.equal(actual, actions)


def test_explicitly_sliced_injected_denoiser_remains_supported():
    actions = torch.arange(180, dtype=torch.float32).reshape(1, 30, 6)

    actual = call_shape_denoiser(actions)

    assert torch.equal(actual, actions)


@pytest.mark.parametrize("action_tokens", [29, 32])
def test_denoiser_rejects_other_token_lengths(action_tokens):
    output = torch.zeros(1, action_tokens, 6)

    with pytest.raises(World2ActionValidationError, match="31.*30"):
        call_shape_denoiser(output)


def test_approved_config_and_uniform_scheduler_values():
    config = World2ActionConfig(device="cpu")
    assert config.state_shape == (1, 6)
    assert config.action_shape == (30, 6)
    assert config.max_horizon == 31
    assert config.in_channels == config.out_channels == 6
    assert config.context_channels == 2048
    assert config.context_layer == 20
    assert config.video_sigma == 10.0
    assert config.model_channels == 1024
    assert config.num_blocks == 24
    assert config.num_heads == 8
    assert config.use_adaln_lora is True
    assert config.adaln_lora_dim == 128
    assert config.pair_timestep_feature_rank == 1024
    assert config.attention_backend == "torch"
    decoder, _ = make_decoder()
    assert decoder.scheduler.alpha == 1.0
    assert decoder.scheduler.beta == 1.0
    assert decoder.scheduler.num_denoising_steps == 10


class FakeSACConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeNativeDenoiser(nn.Module):
    instances = []

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.initialized_value = nn.Parameter(torch.rand(()))
        self.__class__.instances.append(self)


def patch_native_builder(monkeypatch):
    FakeNativeDenoiser.instances.clear()
    monkeypatch.setattr(
        World2ActionDecoder,
        "_load_native_denoiser_components",
        staticmethod(lambda: (FakeNativeDenoiser, FakeSACConfig)),
    )


def test_native_builder_scratch_initializes_with_caller_rng(monkeypatch):
    patch_native_builder(monkeypatch)
    monkeypatch.setattr(
        world2action_module,
        "load_action_checkpoint_strict",
        lambda *args: pytest.fail("scratch construction must not load a checkpoint"),
    )
    torch.manual_seed(123)
    first = World2ActionDecoder(World2ActionConfig(device="cpu"))
    first_value = first.denoiser.initialized_value.detach().clone()
    torch.manual_seed(456)
    second = World2ActionDecoder(World2ActionConfig(device="cpu"))
    second_value = second.denoiser.initialized_value.detach().clone()

    assert len(FakeNativeDenoiser.instances) == 2
    assert torch.isfinite(first_value).item() and torch.isfinite(second_value).item()
    assert not torch.equal(first_value, second_value)
    assert first.denoiser.kwargs["max_horizon"] == 31
    assert first.denoiser.kwargs["in_channels"] == first.denoiser.kwargs["out_channels"] == 6


def test_native_builder_supplied_checkpoint_uses_strict_loader(monkeypatch, tmp_path):
    patch_native_builder(monkeypatch)
    checkpoint = tmp_path / "scratch-or-compatible.safetensors"
    checkpoint.touch()
    calls = []

    def fake_strict_loader(model, path, prefix):
        calls.append((model, path, prefix))

    monkeypatch.setattr(world2action_module, "load_action_checkpoint_strict", fake_strict_loader)
    decoder = World2ActionDecoder(
        World2ActionConfig(device="cpu", action_checkpoint_path=checkpoint, checkpoint_prefix="decoder."),
    )

    assert decoder.denoiser is FakeNativeDenoiser.instances[-1]
    assert calls == [(decoder.denoiser, checkpoint, "decoder.")]


def test_flow_matching_formula_and_context_detach():
    decoder, fake = make_decoder()
    state = torch.zeros(2, 1, 6)
    action = torch.ones(2, 30, 6)
    context = torch.zeros(2, 5, 2048, dtype=torch.bfloat16, requires_grad=True)
    t = torch.tensor([0.25, 0.75])
    epsilon = torch.full_like(action, 3.0)

    loss = decoder.flow_matching_loss(state, action, context, t=t, epsilon=epsilon)
    expected_input = (1.0 - t[0]) * 1.0 + t[0] * 3.0
    expected_input /= torch.sqrt((1.0 - t[0]).square() + t[0].square())
    assert torch.allclose(fake.calls[0]["xt_B_HA_A"][0], torch.full((30, 6), expected_input))
    assert torch.allclose(fake.calls[0]["timesteps_B_T"], t[:, None].expand(2, 31))
    assert fake.calls[0]["context_timesteps_B_1"].shape == (2, 1)
    assert torch.equal(fake.calls[0]["context_timesteps_B_1"], torch.full((2, 1), 10.0))
    assert fake.calls[0]["obs_dropout"] == 0.2
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(4.0)
    loss.backward()
    assert context.grad is None


def test_inference_is_seeded_and_runs_exactly_ten_steps_with_no_obs_dropout():
    decoder, fake = make_decoder()
    state = torch.zeros(2, 1, 6)
    context = torch.zeros(2, 7, 2048, dtype=torch.bfloat16)

    first = decoder.sample_actions(state, context, seed=123)
    assert first.shape == (2, 30, 6)
    assert len(fake.calls) == 10
    assert all(call["obs_dropout"] == 0.0 for call in fake.calls)
    assert all(call["timesteps_B_T"].shape == (2, 31) for call in fake.calls)
    fake.calls.clear()
    second = decoder.sample_actions(state, context, seed=123)
    assert torch.equal(first, second)
    fake.calls.clear()
    third = decoder.sample_actions(state, context, seed=124)
    assert not torch.equal(first, third)


def make_stats():
    state = torch.stack([torch.zeros(6), torch.ones(6)]).reshape(2, 1, 6)
    action = torch.stack([torch.zeros(30, 6), torch.ones(30, 6)])
    return ActionStateNormalizer.from_training_tensors(state, action)


def test_normalizer_round_trip_clamps_and_has_train_provenance(tmp_path):
    normalizer = make_stats()
    assert isinstance(normalizer, NormalizationArtifact)
    state = torch.full((1, 1, 6), 0.25)
    normalized = normalizer.normalize_state(state)
    assert torch.allclose(normalized, torch.full_like(normalized, -0.5))
    assert torch.allclose(normalizer.denormalize_state(normalized), state)
    assert torch.equal(normalizer.normalize_state(torch.full_like(state, 2.0)), torch.ones_like(state))
    assert torch.equal(normalizer.denormalize_state(torch.full_like(state, -2.0)), torch.zeros_like(state))

    artifact = tmp_path / "normalizer.safetensors"
    normalizer.save(artifact)
    loaded = ActionStateNormalizer.load(artifact)
    assert loaded.source_split == "train"
    assert torch.equal(loaded.action_min, normalizer.action_min)
    metadata = json.loads(artifact.with_suffix(".json").read_text())
    assert metadata["source_split"] == "train"
    metadata["source_split"] = "validation"
    artifact.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(World2ActionValidationError, match="contract"):
        ActionStateNormalizer.load(artifact)


def test_normalizer_rejects_degenerate_ranges_and_non_train_stats():
    with pytest.raises(World2ActionValidationError, match="source_split"):
        ActionStateNormalizer.from_training_tensors(
            torch.zeros(2, 1, 6), torch.zeros(2, 30, 6), source_split="validation"
        )
    with pytest.raises(World2ActionValidationError, match="degenerate"):
        ActionStateNormalizer.from_training_tensors(
            torch.zeros(2, 1, 6), torch.stack([torch.zeros(30, 6), torch.ones(30, 6)])
        )
    with pytest.raises(World2ActionValidationError, match="degenerate"):
        ActionStateNormalizer.from_training_tensors(
            torch.stack([torch.zeros(6), torch.ones(6)]).reshape(2, 1, 6), torch.zeros(2, 30, 6)
        )


def test_normalizer_shape_and_parameter_count_report():
    decoder, _ = make_decoder(normalizer=make_stats())
    report = decoder.parameter_count_report()
    assert report.decoder_parameters == 1
    assert report.trainable_decoder_parameters == 1
    assert report.normalizer_parameters == 0
    assert report.trainable_normalizer_parameters == 0
    with pytest.raises(World2ActionValidationError, match="shape"):
        decoder.sample_actions(torch.zeros(1, 6), torch.zeros(1, 4, 2048, dtype=torch.bfloat16))


def test_normalizer_load_rejects_malformed_tensor_shape(tmp_path):
    artifact = tmp_path / "bad-normalizer.safetensors"
    save_file(
        {
            "state_min": torch.zeros(5),
            "state_max": torch.ones(5),
            "action_min": torch.zeros(6),
            "action_max": torch.ones(6),
        },
        str(artifact),
    )
    artifact.with_suffix(".json").write_text(json.dumps(make_stats()._metadata()))
    with pytest.raises(World2ActionValidationError, match="shape"):
        ActionStateNormalizer.load(artifact)


def test_malformed_context_dtype_and_optional_time_are_rejected():
    decoder, _ = make_decoder()
    state = torch.zeros(1, 1, 6)
    action = torch.zeros(1, 30, 6)
    with pytest.raises(TypeError, match="bfloat16"):
        decoder.flow_matching_loss(state, action, torch.zeros(1, 4, 2048))
    with pytest.raises(World2ActionValidationError, match="context_timestep"):
        decoder.flow_matching_loss(
            state,
            action,
            torch.zeros(1, 4, 2048, dtype=torch.bfloat16),
            context_timestep=torch.ones(1),
        )


def test_flow_matching_mask_excludes_padded_tokens_and_rejects_all_padding():
    decoder, _ = make_decoder()
    state = torch.zeros(1, 1, 6)
    action = torch.ones(1, 30, 6)
    context = torch.zeros(1, 4, 2048, dtype=torch.bfloat16)
    t = torch.zeros(1)
    epsilon = torch.full_like(action, 3.0)
    padding = torch.zeros(1, 30, dtype=torch.bool)
    padding[:, 1:] = True
    masked = decoder.flow_matching_loss(state, action, context, t=t, epsilon=epsilon, action_is_pad=padding)
    assert masked.item() == pytest.approx(4.0)
    with pytest.raises(World2ActionValidationError, match="every action token"):
        decoder.flow_matching_loss(
            state, action, context, t=t, epsilon=epsilon, action_is_pad=torch.ones(1, 30, dtype=torch.bool)
        )
    with pytest.raises(World2ActionValidationError, match="boolean dtype"):
        decoder.flow_matching_loss(
            state, action, context, t=t, epsilon=epsilon, action_is_pad=torch.zeros(1, 30)
        )


def test_normalizer_excludes_padded_action_values():
    state = torch.stack([torch.zeros(6), torch.ones(6)]).reshape(2, 1, 6)
    action = torch.zeros(2, 30, 6)
    action[1].fill_(100.0)
    action[1, 0].fill_(1.0)
    padding = torch.zeros(2, 30, dtype=torch.bool)
    padding[1, 1:] = True
    normalizer = ActionStateNormalizer.from_training_tensors(state, action, action_is_pad=padding)
    assert torch.equal(normalizer.action_min, torch.zeros(6))
    assert torch.equal(normalizer.action_max, torch.ones(6))
