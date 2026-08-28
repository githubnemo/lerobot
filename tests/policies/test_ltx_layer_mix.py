import copy

import pytest
import torch
import torch.nn.functional as functional

from lerobot.policies.vam.ltx_action import (
    apply_ltx_context_transform as policy_context_transform,
)
from lerobot.policies.vam.ltx_layer_mix import (
    LTXAttentionDiagnosticsAccumulator,
    LTXLayerAttentionMix,
    LTXLayerGatedMix,
    LTXLayerScalarMix,
    apply_ltx_context_transform,
)
from lerobot.policies.vam.ltx_layer_mix_cache import (
    LTXLayerMixCacheError,
    validate_layer_mix_provenance,
)
from scripts.video_vam.build_ltx_layer_mix_cache import (
    apply_ltx_context_transform as builder_context_transform,
    parse_args as builder_parse_args,
)
from scripts.video_vam.train_ltx_layer_mix import parse_args as trainer_parse_args


def _contexts(dtype: torch.dtype = torch.float32) -> dict[int, torch.Tensor]:
    base = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    return {
        8: base.to(dtype),
        14: (base * 100 + 17).to(dtype),
        20: (base * 0.01 - 3).to(dtype),
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scalar_mix_shape_and_dtype_handling(dtype: torch.dtype):
    mixer = LTXLayerScalarMix((8, 14, 20), hidden_width=4, seed=7)
    output = mixer(_contexts(dtype))
    assert output.shape == (2, 3, 4)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()

    bad = _contexts(dtype)
    bad[14] = torch.zeros((2, 4, 4), dtype=dtype)
    with pytest.raises(ValueError, match="identical shapes"):
        mixer(bad)


def test_softmax_normalizes_and_each_layer_has_its_own_non_affine_layernorm():
    mixer = LTXLayerScalarMix((8, 14, 20), hidden_width=4, seed=3)
    assert len({id(norm) for norm in mixer.norms}) == 3
    assert all(not norm.elementwise_affine for norm in mixer.norms)
    with torch.no_grad():
        mixer.logits.copy_(torch.tensor([-1.0, 0.5, 2.0]))
        mixer.gain.fill_(1.7)

    contexts = _contexts()
    weights = torch.softmax(mixer.logits, dim=0)
    assert weights.sum().item() == pytest.approx(1.0)
    expected = 1.7 * sum(
        weight * norm(contexts[layer])
        for layer, weight, norm in zip(mixer.layers, weights, mixer.norms, strict=True)
    )
    assert torch.allclose(mixer(contexts), expected)

    diagnostics = mixer.diagnostics(contexts)
    assert sum(diagnostics["weights"].values()) == pytest.approx(1.0)
    assert set(diagnostics["mean_contribution_norms"]) == {8, 14, 20}


def test_scalar_mix_initialization_is_seeded_without_mutating_global_rng():
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    first = LTXLayerScalarMix((8, 14, 20), hidden_width=4, seed=19)
    after = torch.random.get_rng_state()
    second = LTXLayerScalarMix((8, 14, 20), hidden_width=4, seed=19)
    third = LTXLayerScalarMix((8, 14, 20), hidden_width=4, seed=20)

    assert torch.equal(before, after)
    assert torch.equal(first.logits, second.logits)
    assert not torch.equal(first.logits, third.logits)
    assert torch.equal(first(_contexts()), second(_contexts()))


@pytest.mark.parametrize(
    ("context_transform", "target_hw", "tokens"),
    [("pool2", (8, 10), 640), ("pool4", (4, 5), 160)],
)
def test_builder_and_policy_use_identical_ltx_context_transform(
    context_transform: str, target_hw: tuple[int, int], tokens: int
):
    hidden = torch.zeros((1, 8, 15, 20, 4096), dtype=torch.bfloat16)
    for frame in range(8):
        hidden[:, frame].fill_(frame)
    expected = (
        functional.adaptive_avg_pool2d(hidden.permute(0, 1, 4, 2, 3).reshape(8, 4096, 15, 20), target_hw)
        .reshape(1, 8, 4096, *target_hw)
        .permute(0, 1, 3, 4, 2)
        .reshape(1, tokens, 4096)
    )
    assert builder_context_transform is apply_ltx_context_transform is policy_context_transform
    pooled = builder_context_transform(hidden, context_transform)
    assert pooled.shape == (1, tokens, 4096)
    assert pooled.dtype == torch.bfloat16
    assert torch.equal(pooled, expected)
    assert [float(pooled[0, frame * (tokens // 8), 0]) for frame in range(8)] == list(range(8))


def test_transform_selects_truthful_default_artifact_paths():
    builder = builder_parse_args(["--context-transform", "pool4"])
    trainer = trainer_parse_args(["--context-transform", "pool4"])
    assert builder.train_output_dir.name.endswith("-pool4")
    assert builder.val_output_dir.name.endswith("-pool4")
    assert trainer.train_manifest.parent.name.endswith("-pool4")
    assert trainer.val_manifest.parent.name.endswith("-pool4")
    assert "pool4" in trainer.output_dir.name


@pytest.mark.parametrize(
    ("context_transform", "tokens", "grid"),
    [("pool2", 640, [8, 8, 10]), ("pool4", 160, [8, 4, 5])],
)
def test_layer_probe_provenance_validation_is_strict(context_transform: str, tokens: int, grid: list[int]):
    provenance = {
        "backbone": "LTX-2.5-22B-distilled",
        "tapped_layers": [8, 14, 20, 26, 34, 40],
        "deepest_layer": 40,
        "num_blocks": 48,
        "high_noise_sigma": 1.0,
        "context_transform": context_transform,
        "context_tokens_per_layer": tokens,
        "context_channels": 4096,
        "context_grid": grid,
        "context_dtype": "bfloat16",
        "one_forward_pass": True,
    }
    validate_layer_mix_provenance(provenance)

    invalid = copy.deepcopy(provenance)
    invalid["tapped_layers"] = [8, 14, 20, 26, 34]
    with pytest.raises(LTXLayerMixCacheError, match="producer contract"):
        validate_layer_mix_provenance(invalid)


def _independent_contexts(dtype: torch.dtype = torch.float32) -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(31)
    return {layer: torch.randn((2, 5, 8), generator=generator).to(dtype) for layer in (8, 14, 20)}


def _small_mixers(seed: int = 7):
    return (
        LTXLayerScalarMix((8, 14, 20), hidden_width=8, seed=seed),
        LTXLayerAttentionMix((8, 14, 20), hidden_width=8, attn_width=8, num_heads=2, seed=seed),
        LTXLayerGatedMix((8, 14, 20), hidden_width=8, seed=seed),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_all_mixers_have_matching_output_shape_and_dtype(dtype: torch.dtype):
    contexts = _independent_contexts(dtype)
    for mixer in (
        *_small_mixers(),
        LTXLayerGatedMix(
            (8, 14, 20),
            hidden_width=8,
            per_channel=True,
            seed=7,
            per_channel_chunk_tokens=2,
        ),
    ):
        output = mixer(contexts)
        assert output.shape == (2, 5, 8)
        assert output.dtype == torch.float32
        assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "mixer",
    [
        LTXLayerAttentionMix((8, 14, 20), hidden_width=8, attn_width=8, num_heads=2, seed=11),
        LTXLayerGatedMix((8, 14, 20), hidden_width=8, seed=11),
        LTXLayerGatedMix(
            (8, 14, 20),
            hidden_width=8,
            per_channel=True,
            seed=11,
            per_channel_chunk_tokens=2,
        ),
    ],
)
def test_new_mixers_initialize_close_to_uniform_normalized_mean(mixer):
    contexts = _independent_contexts()
    expected = sum(
        norm(contexts[layer]) for layer, norm in zip(mixer.layers, mixer.norms, strict=True)
    ) / len(mixer.layers)
    assert torch.allclose(mixer(contexts), expected, atol=5.0e-3, rtol=5.0e-3)


@pytest.mark.parametrize(
    "mixer",
    (
        *_small_mixers(seed=13),
        LTXLayerGatedMix(
            (8, 14, 20),
            hidden_width=8,
            per_channel=True,
            seed=13,
            per_channel_chunk_tokens=2,
        ),
    ),
)
def test_gradients_flow_to_every_trainable_mixer_parameter(mixer):
    output = mixer(_independent_contexts())
    output.square().mean().backward()
    for name, parameter in mixer.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum().item() > 0.0, name


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LTXLayerAttentionMix((8, 14, 20), hidden_width=8, attn_width=8, num_heads=2, seed=17),
        lambda: LTXLayerGatedMix((8, 14, 20), hidden_width=8, seed=17),
        lambda: LTXLayerGatedMix(
            (8, 14, 20),
            hidden_width=8,
            per_channel=True,
            seed=17,
            per_channel_chunk_tokens=2,
        ),
    ],
)
def test_new_mixer_initialization_and_outputs_are_deterministic(factory):
    first = factory()
    second = factory()
    assert all(
        torch.equal(first_value, second_value)
        for first_value, second_value in zip(
            first.state_dict().values(), second.state_dict().values(), strict=True
        )
    )
    assert torch.equal(first(_independent_contexts()), second(_independent_contexts()))


@pytest.mark.parametrize("mixer", _small_mixers(seed=23))
def test_mixer_diagnostics_have_comparable_keys_and_normalized_weights(mixer):
    diagnostics = mixer.diagnostics(_independent_contexts())
    assert {"weights", "mean_contribution_norms", "gain"}.issubset(diagnostics)
    assert set(diagnostics["weights"]) == {8, 14, 20}
    assert set(diagnostics["mean_contribution_norms"]) == {8, 14, 20}
    assert sum(diagnostics["weights"].values()) == pytest.approx(1.0, abs=1.0e-6)
    if not isinstance(mixer, LTXLayerScalarMix):
        assert diagnostics["weight_dispersion"] >= 0.0


def test_attention_diagnostics_aggregate_exactly_across_validation_batches():
    mixer = LTXLayerAttentionMix((8, 14, 20), hidden_width=8, attn_width=8, num_heads=2, seed=29)
    contexts = _independent_contexts()
    whole = mixer.diagnostics(contexts)
    accumulator = LTXAttentionDiagnosticsAccumulator(mixer)
    for index in range(contexts[8].shape[0]):
        accumulator.update({layer: value[index : index + 1] for layer, value in contexts.items()})
    split = accumulator.compute()

    assert sum(split["weights"].values()) == pytest.approx(1.0, abs=1.0e-6)
    assert split["weights"] == pytest.approx(whole["weights"], abs=1.0e-7)
    assert split["mean_contribution_norms"] == pytest.approx(whole["mean_contribution_norms"], abs=1.0e-6)
    assert split["weight_dispersion"] == pytest.approx(whole["weight_dispersion"], abs=1.0e-7)


def test_default_mixer_parameter_counts_have_expected_order_of_magnitude():
    attention = LTXLayerAttentionMix()
    gated = LTXLayerGatedMix()
    attention_count = sum(parameter.numel() for parameter in attention.parameters())
    gated_count = sum(parameter.numel() for parameter in gated.parameters())
    assert 20_000_000 < attention_count < 40_000_000
    assert 20_000 < gated_count < 30_000


def test_trainer_mixer_flags_preserve_scalar_default_and_parse_variants():
    defaults = trainer_parse_args([])
    assert defaults.mixer == "scalar"
    assert defaults.attn_width == 2048
    assert defaults.attn_heads == 8
    assert defaults.gate_per_channel is False

    attention = trainer_parse_args(["--mixer", "attention", "--attn-width", "1024", "--attn-heads", "4"])
    assert (attention.mixer, attention.attn_width, attention.attn_heads) == ("attention", 1024, 4)
    gated = trainer_parse_args(["--mixer", "gated", "--gate-per-channel"])
    assert gated.mixer == "gated"
    assert gated.gate_per_channel is True
