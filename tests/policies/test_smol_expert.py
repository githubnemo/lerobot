from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.vam.context_transform import apply_context_transform
from lerobot.policies.vam.cosmos_cache_dataset import CacheManifestEntry
from lerobot.policies.vam.ltx_layer_mix import LTXLayerAttentionMix
from lerobot.policies.vam.smol_expert import (
    CosmosPrefixAdapter,
    SmolExpertActionDecoder,
    SmolExpertPrefixKVCache,
    SmolVLANormalizer,
    _tensor_digest,
    checkpoint_tensor_sha256,
    cuda_graph_shape_eligible,
)
from lerobot.policies.vam.vam_split import _make_probe


class _TinyAttention(nn.Module):
    def __init__(self, hidden: int = 8, kv: int = 4) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, kv, bias=False)
        self.v_proj = nn.Linear(hidden, kv, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class _TinyLayer(nn.Module):
    def __init__(self, cross: bool) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(8)
        self.self_attn = _TinyAttention()
        if cross:
            self.self_attn.k_proj = nn.Linear(4, 4, bias=False)
            self.self_attn.v_proj = nn.Linear(4, 4, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(8)
        self.mlp = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 8))


class _TinyExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer(cross=index % 2 == 1) for index in range(2)])
        self.norm = nn.LayerNorm(8)


def _normalizer() -> SmolVLANormalizer:
    return SmolVLANormalizer(
        state_mean=torch.zeros(6),
        state_std=torch.ones(6),
        action_mean=torch.zeros(6),
        action_std=torch.ones(6),
    )


def _decoder() -> SmolExpertActionDecoder:
    return SmolExpertActionDecoder(
        _normalizer(),
        expert=_TinyExpert(),
        prefix_hidden_size=8,
        expert_hidden_size=8,
        kv_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_steps=2,
        input_channels=4,
        max_action_dim=8,
        max_state_dim=8,
    )


def _checkpoint_from_decoder(decoder: SmolExpertActionDecoder, path) -> None:
    tensors = {}
    modules = (
        (decoder.expert, "model.vlm_with_expert.lm_expert."),
        (decoder.state_proj, "model.state_proj."),
        (decoder.action_in_proj, "model.action_in_proj."),
        (decoder.action_out_proj, "model.action_out_proj."),
        (decoder.action_time_mlp_in, "model.action_time_mlp_in."),
        (decoder.action_time_mlp_out, "model.action_time_mlp_out."),
    )
    for module, prefix in modules:
        tensors.update({prefix + name: value.detach().clone() for name, value in module.state_dict().items()})
    save_file(tensors, str(path))


def _training_checkpoint_from_decoder(decoder: SmolExpertActionDecoder, path) -> None:
    save_file(
        {f"model.{name}": value.detach().clone() for name, value in decoder.state_dict().items()},
        str(path),
    )


def test_training_checkpoint_weights_load_with_trainer_format(tmp_path) -> None:
    source = _decoder()
    checkpoint = tmp_path / "best.safetensors"
    _training_checkpoint_from_decoder(source, checkpoint)
    target = _decoder()

    digests = target.load_training_checkpoint(checkpoint)

    key = "model.expert.layers.0.self_attn.q_proj.weight"
    assert digests[key] == checkpoint_tensor_sha256(checkpoint, key)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value)


def test_pool2_context_reduction_shape_is_shared_transform() -> None:
    context = torch.randn(2, 16 * 30 * 40, 4)
    reduced = apply_context_transform(context, "pool2")
    assert reduced.shape == (2, 4_800, 4)


def test_cosmos_adapter_maps_channels_to_prefix_hidden_width() -> None:
    adapter = CosmosPrefixAdapter(input_channels=4, prefix_hidden_size=8)
    output = adapter(torch.randn(2, 4_800, 4))
    assert output.shape == (2, 4_800, 8)
    assert adapter.projection.in_features == 4
    assert adapter.projection.out_features == 8


def test_pretrained_expert_weights_load_and_hash_against_checkpoint(tmp_path) -> None:
    source = _decoder()
    checkpoint = tmp_path / "model.safetensors"
    _checkpoint_from_decoder(source, checkpoint)
    target = _decoder()
    digests = target.load_pretrained_checkpoint(checkpoint)
    key = "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight"
    assert digests[key] == checkpoint_tensor_sha256(checkpoint, key)
    assert digests["model.action_in_proj.weight"] == checkpoint_tensor_sha256(
        checkpoint, "model.action_in_proj.weight"
    )
    assert torch.equal(
        target.expert.layers[0].self_attn.q_proj.weight, source.expert.layers[0].self_attn.q_proj.weight
    )
    assert torch.equal(target.action_in_proj.weight, source.action_in_proj.weight)
    assert _tensor_digest(target.action_in_proj.weight) == digests["model.action_in_proj.weight"]


def test_fixed_probe_generation_is_deterministic() -> None:
    entry = CacheManifestEntry(
        sample_id="episode-0000-frame-000004",
        episode_index=0,
        frame_index=4,
        window_indices=(0, 1, 2, 3, 4),
        noise_seed=7,
        safetensors="episode-0000-frame-000004.safetensors",
        sidecar="episode-0000-frame-000004.json",
        safetensors_sha256="0" * 64,
        sidecar_sha256="1" * 64,
        bytes=1,
    )
    assert _make_probe((entry,), 123) == _make_probe((entry,), 123)
    assert _make_probe((entry,), 123) != _make_probe((entry,), 124)


def test_smol_expert_loss_is_finite_and_backpropagates() -> None:
    torch.manual_seed(0)
    decoder = _decoder()
    state = torch.randn(2, 6)
    action = torch.randn(2, 30, 6)
    context = torch.randn(2, 4, 4)
    padding = torch.zeros(2, 30, dtype=torch.bool)
    t = torch.tensor([0.2, 0.8])
    epsilon = torch.randn(2, 30, 6)
    loss = decoder.flow_matching_loss(
        state,
        action,
        context,
        t=t,
        epsilon=epsilon,
        action_is_pad=padding,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert decoder.context_adapter.projection.weight.grad is not None
    assert torch.isfinite(decoder.context_adapter.projection.weight.grad).all()


def test_attention_mixer_and_smol_expert_receive_joint_gradients() -> None:
    torch.manual_seed(0)
    decoder = _decoder()
    mixer = LTXLayerAttentionMix((8, 14, 20), hidden_width=4, attn_width=4, num_heads=1, seed=0)
    contexts = {layer: torch.randn(2, 4, 4) + index for index, layer in enumerate(mixer.layers)}
    mixed = mixer(contexts)
    loss = decoder.flow_matching_loss(
        torch.randn(2, 6),
        torch.randn(2, 30, 6),
        mixed,
        action_is_pad=torch.zeros(2, 30, dtype=torch.bool),
    )
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in mixer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum().item() > 0, name
    assert decoder.context_adapter.projection.weight.grad is not None
    assert decoder.expert.layers[0].self_attn.q_proj.weight.grad is not None


def test_sampling_with_fixed_probe_seed_is_deterministic() -> None:
    decoder = _decoder().eval()
    state = torch.zeros(1, 6)
    context = torch.zeros(1, 4, 4)
    first = decoder.sample_actions(state, context, seed=31415)
    second = decoder.sample_actions(state, context, seed=31415)
    assert first.shape == (1, 30, 6)
    assert torch.equal(first, second)


def test_sampling_accepts_physical_rtc_prefix_and_guides_overlap() -> None:
    torch.manual_seed(0)
    decoder = _decoder().eval()
    state = torch.zeros(1, 6)
    context = torch.zeros(1, 4, 4)
    previous = torch.full((10, 6), 2.0)

    baseline = decoder.sample_actions(state, context, seed=7, use_cuda_graph=False)
    guided = decoder.sample_actions(
        state,
        context,
        seed=7,
        use_cuda_graph=False,
        rtc_processor=RTCProcessor(RTCConfig(execution_horizon=10)),
        inference_delay=2,
        prev_chunk_left_over=previous,
        execution_horizon=10,
    )

    baseline_overlap_rmse = (baseline[0, :10] - previous).square().mean().sqrt()
    guided_overlap_rmse = (guided[0, :10] - previous).square().mean().sqrt()
    assert guided.shape == baseline.shape == (1, 30, 6)
    assert torch.isfinite(guided).all()
    assert guided_overlap_rmse < baseline_overlap_rmse


def test_cached_prefix_kv_matches_uncached_sampling() -> None:
    torch.manual_seed(23)
    decoder = _decoder().eval()
    state = torch.randn(1, 6)
    context = torch.randn(1, 4, 4)

    uncached = decoder.sample_actions(state, context, seed=91, use_prefix_kv_cache=False)
    cached = decoder.sample_actions(state, context, seed=91)
    cache = decoder.prepare_prefix_kv(state, context)
    explicitly_cached = decoder.sample_actions(state, context, seed=91, prefix_kv_cache=cache)

    assert len(cache.keys) == len(decoder.expert.layers)
    assert len(cache.values) == len(cache.keys)
    torch.testing.assert_close(cached, uncached, rtol=0.0, atol=0.0)
    torch.testing.assert_close(explicitly_cached, uncached, rtol=0.0, atol=0.0)


def test_sampling_prepares_prefix_cache_once_per_call(monkeypatch) -> None:
    decoder = _decoder().eval()
    state = torch.randn(1, 6)
    context = torch.randn(1, 4, 4)
    prepared: list[SmolExpertPrefixKVCache] = []
    vector_fields: list[object] = []
    original_prepare = SmolExpertActionDecoder.prepare_prefix_kv
    original_vector_field = SmolExpertActionDecoder._vector_field

    def counted_prepare(self, state, context):
        cache = original_prepare(self, state, context)
        prepared.append(cache)
        return cache

    def counted_vector_field(self, prefix, noisy_action, time):
        vector_fields.append(prefix)
        return original_vector_field(self, prefix, noisy_action, time)

    monkeypatch.setattr(SmolExpertActionDecoder, "prepare_prefix_kv", counted_prepare)
    monkeypatch.setattr(SmolExpertActionDecoder, "_vector_field", counted_vector_field)
    decoder.sample_actions(state, context, seed=17, num_steps=4)

    assert len(prepared) == 1
    assert len(vector_fields) == 4
    assert all(isinstance(prefix, SmolExpertPrefixKVCache) for prefix in vector_fields)
    assert len({id(prefix) for prefix in vector_fields}) == 1

    prepared.clear()
    decoder.sample_actions(state, context, seed=17, num_steps=4, use_prefix_kv_cache=False)
    assert not prepared


def test_sampling_prepares_fresh_cache_for_each_observation(monkeypatch) -> None:
    decoder = _decoder().eval()
    state = torch.randn(1, 6)
    context = torch.randn(1, 4, 4)
    caches: list[SmolExpertPrefixKVCache] = []
    original_prepare = SmolExpertActionDecoder.prepare_prefix_kv

    def counted_prepare(self, state, context):
        cache = original_prepare(self, state, context)
        caches.append(cache)
        return cache

    monkeypatch.setattr(SmolExpertActionDecoder, "prepare_prefix_kv", counted_prepare)
    decoder.sample_actions(state, context, seed=3, num_steps=1)
    decoder.sample_actions(state + 1.0, context + 1.0, seed=3, num_steps=1)

    assert len(caches) == 2
    assert caches[0] is not caches[1]
    assert caches[0].keys[0].data_ptr() != caches[1].keys[0].data_ptr()
    assert not torch.equal(caches[0].keys[0], caches[1].keys[0])


def test_sampling_step_override_does_not_mutate_default() -> None:
    decoder = _decoder().eval()
    state = torch.zeros(1, 6)
    context = torch.zeros(1, 4, 4)
    noise = decoder._noise_for_seed(1, torch.device("cpu"), 7)

    decoder.sample_actions(state, context, noise=noise, num_steps=1)
    assert decoder.num_steps == 2


def _static_shape_decoder() -> SmolExpertActionDecoder:
    return SmolExpertActionDecoder(
        _normalizer(),
        expert=_TinyExpert(),
        prefix_hidden_size=8,
        expert_hidden_size=8,
        kv_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_steps=2,
        input_channels=2048,
        max_action_dim=8,
        max_state_dim=8,
    )


def test_cuda_graph_opt_in_falls_back_eager_on_cpu():
    decoder = _decoder().eval()
    state = torch.randn(1, 6)
    context = torch.randn(1, 4, 4)
    noise = decoder._noise_for_seed(1, torch.device("cpu"), 19)

    eager = decoder.sample_actions(state, context, noise=noise, use_cuda_graph=False)
    opted_in = decoder.sample_actions(state, context, noise=noise, use_cuda_graph=True)

    torch.testing.assert_close(opted_in, eager, rtol=0.0, atol=0.0)
    assert decoder.cuda_graph_capture_count == 0
    assert decoder.last_cuda_graph_capture_seconds is None


def test_cuda_graph_shape_predicate_accepts_only_production_cosmos_shape() -> None:
    common = {
        "state_shape": (1, 6),
        "noise_shape": (1, 30, 32),
        "max_action_dim": 32,
        "prefix_is_cached": True,
    }
    assert cuda_graph_shape_eligible(context_shape=(1, 4_800, 2_048), **common)
    assert not cuda_graph_shape_eligible(context_shape=(1, 640, 4_096), **common)
    assert not cuda_graph_shape_eligible(context_shape=(1, 2_400, 4_096), **common)
    assert not cuda_graph_shape_eligible(
        context_shape=(1, 4_800, 2_048),
        **{**common, "prefix_is_cached": False},
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph smoke requires CUDA")
def test_cuda_graph_replays_two_fresh_observations_without_stale_prefix():
    device = torch.device("cuda")
    decoder = _static_shape_decoder().to(device=device).eval()
    state_one = torch.randn(1, 6, device=device)
    context_one = torch.randn(1, 4_800, 2_048, device=device)
    noise_one = torch.randn(1, 30, 8, device=device)
    state_two = state_one + 0.5
    context_two = context_one + 0.125
    noise_two = -noise_one

    eager_one = decoder.sample_actions(
        state_one, context_one, noise=noise_one, use_cuda_graph=False, num_steps=2
    )
    graph_one = decoder.sample_actions(state_one, context_one, noise=noise_one, num_steps=2)
    eager_two = decoder.sample_actions(
        state_two, context_two, noise=noise_two, use_cuda_graph=False, num_steps=2
    )
    graph_two = decoder.sample_actions(state_two, context_two, noise=noise_two, num_steps=2)

    torch.testing.assert_close(graph_one, eager_one, rtol=0.0, atol=0.0)
    torch.testing.assert_close(graph_two, eager_two, rtol=0.0, atol=0.0)
    assert not torch.equal(eager_one, eager_two)
    assert decoder.cuda_graph_capture_count == 1
    assert decoder.last_cuda_graph_capture_seconds is not None

    eager_one_step = decoder.sample_actions(
        state_one, context_one, noise=noise_one, use_cuda_graph=False, num_steps=1
    )
    graph_one_step = decoder.sample_actions(state_one, context_one, noise=noise_one, num_steps=1)
    torch.testing.assert_close(graph_one_step, eager_one_step, rtol=0.0, atol=0.0)
    assert decoder.cuda_graph_capture_count == 2
