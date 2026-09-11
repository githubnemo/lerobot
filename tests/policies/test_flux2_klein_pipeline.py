"""Synthetic CPU dry-run tests for FLUX.2 [klein] (9B) feature extraction and training pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import nn

from lerobot.policies.vam import (
    Flux2KleinExtractionOutput,
    Flux2KleinExtractor,
    Flux2KleinExtractorConfig,
    Flux2KleinLoRAConfig,
    MultiReferenceInputs,
    build_flux2_klein_dummy_model,
    compute_multi_reference_rf_loss,
    inject_flux2_klein_lora,
    load_flux2_klein_lora,
    merge_flux2_klein_lora,
    prepare_multi_reference_conditioning,
    save_flux2_klein_lora,
)
from lerobot.policies.vam.smol_expert import (
    ACTION_DIM,
    ACTION_HORIZON,
    SmolExpertActionDecoder,
    SmolVLANormalizer,
)


def _make_tiny_expert(input_channels: int = 32) -> SmolExpertActionDecoder:
    class _TinyAttention(nn.Module):
        def __init__(self, hidden: int = 16, kv: int = 8) -> None:
            super().__init__()
            self.q_proj = nn.Linear(hidden, hidden, bias=False)
            self.k_proj = nn.Linear(hidden, kv, bias=False)
            self.v_proj = nn.Linear(hidden, kv, bias=False)
            self.o_proj = nn.Linear(hidden, hidden, bias=False)

    class _TinyLayer(nn.Module):
        def __init__(self, cross: bool) -> None:
            super().__init__()
            self.input_layernorm = nn.LayerNorm(16)
            self.self_attn = _TinyAttention()
            if cross:
                self.self_attn.k_proj = nn.Linear(8, 8, bias=False)
                self.self_attn.v_proj = nn.Linear(8, 8, bias=False)
            self.post_attention_layernorm = nn.LayerNorm(16)
            self.mlp = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 16))

    class _TinyExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_TinyLayer(cross=i % 2 == 1) for i in range(2)])
            self.norm = nn.LayerNorm(16)

    normalizer = SmolVLANormalizer(
        state_mean=torch.zeros(ACTION_DIM),
        state_std=torch.ones(ACTION_DIM),
        action_mean=torch.zeros(ACTION_DIM),
        action_std=torch.ones(ACTION_DIM),
    )

    return SmolExpertActionDecoder(
        normalizer,
        expert=_TinyExpert(),
        prefix_hidden_size=16,
        expert_hidden_size=16,
        kv_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_steps=2,
        input_channels=input_channels,
        max_action_dim=16,
        max_state_dim=16,
    )


def test_multi_reference_conditioning_formatting():
    target = torch.randn(1, 16, 4, 4)
    # 3 frames of history: t-2, t-1, t
    history = [torch.randn(1, 16, 4, 4) for _ in range(3)]
    goal = torch.randn(1, 16, 4, 4)

    cond: MultiReferenceInputs = prepare_multi_reference_conditioning(
        target_latent=target,
        observation_history=history,
        goal_latent=goal,
        joint_attention_dim=32,
        ref_time_scale=10,
        goal_time_coord=100,
    )

    # 4x4 = 16 tokens per frame
    # 1 target + 3 history + 1 goal = 5 frames = 80 tokens
    assert cond.hidden_states.shape == (1, 80, 16)
    assert cond.img_ids.shape == (1, 80, 4)
    assert cond.target_tokens_count == 16
    assert cond.history_tokens_count == 48
    assert cond.goal_tokens_count == 16
    assert "target" in cond.token_slices
    assert "history_t" in cond.token_slices
    assert "goal" in cond.token_slices

    # Verify time coordinates
    target_ids = cond.img_ids[0, :16, :]
    assert (target_ids[:, 0] == 0.0).all()

    hist_ids_0 = cond.img_ids[0, 16:32, :]
    assert (hist_ids_0[:, 0] == 10.0).all()

    goal_ids = cond.img_ids[0, 64:80, :]
    assert (goal_ids[:, 0] == 100.0).all()


def test_flux2_klein_tap_junction():
    dummy = build_flux2_klein_dummy_model(
        num_layers=2,
        num_single_layers=2,
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=16,
        joint_attention_dim=32,
        axes_dims_rope=(4, 4, 4, 4),
    )
    config = Flux2KleinExtractorConfig(
        num_layers=2,
        num_single_layers=2,
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_dim=32,
        in_channels=16,
        joint_attention_dim=32,
        axes_dims_rope=(4, 4, 4, 4),
        tap_location="junction",
    )
    extractor = Flux2KleinExtractor(config=config, model=dummy)

    cond = prepare_multi_reference_conditioning(
        target_latent=torch.randn(1, 16, 4, 4),
        observation_history=[torch.randn(1, 16, 4, 4) for _ in range(2)],
        goal_latent=None,
        joint_attention_dim=32,
    )
    # 3 frames total = 48 tokens
    out: Flux2KleinExtractionOutput = extractor.extract(cond, timestep=500.0)

    assert out.tap_location == "junction"
    assert out.features.shape == (1, 48, 32)
    assert out.visual_tokens.shape == (1, 48, 32)
    assert out.text_tokens is not None
    assert out.junction_features is not None
    assert out.junction_features.shape[1] == out.visual_tokens.shape[1] + out.text_tokens.shape[1]


def test_flux2_klein_tap_rectified_flow_trajectory():
    dummy = build_flux2_klein_dummy_model(
        num_layers=2,
        num_single_layers=2,
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=16,
        joint_attention_dim=32,
        axes_dims_rope=(4, 4, 4, 4),
    )
    config = Flux2KleinExtractorConfig(
        num_layers=2,
        num_single_layers=2,
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_dim=32,
        in_channels=16,
        joint_attention_dim=32,
        axes_dims_rope=(4, 4, 4, 4),
        tap_location="rectified_flow_trajectory",
        rf_num_steps=4,
        rf_tap_step=2,
    )
    extractor = Flux2KleinExtractor(config=config, model=dummy)

    cond = prepare_multi_reference_conditioning(
        target_latent=torch.randn(1, 16, 4, 4),
        observation_history=[torch.randn(1, 16, 4, 4)],
        goal_latent=None,
        joint_attention_dim=32,
    )
    out: Flux2KleinExtractionOutput = extractor.extract(cond)

    assert out.tap_location == "rectified_flow_trajectory"
    assert out.trajectory_step == 2
    assert out.features.shape == (1, 32, 32)
    assert "total_trajectory_steps" in out.provenance


def test_flux2_klein_lora_lifecycle():
    dummy = build_flux2_klein_dummy_model(
        num_layers=2,
        num_single_layers=2,
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=16,
        joint_attention_dim=32,
        axes_dims_rope=(4, 4, 4, 4),
    )
    lora_cfg = Flux2KleinLoRAConfig(
        rank=4,
        alpha=4.0,
        target_double_blocks=(0, 1),
        target_single_blocks=(0, 1),
    )
    injected = inject_flux2_klein_lora(dummy, lora_cfg)
    assert len(injected) > 0

    cond = prepare_multi_reference_conditioning(
        target_latent=torch.randn(1, 16, 4, 4),
        observation_history=[torch.randn(1, 16, 4, 4)],
        goal_latent=None,
        joint_attention_dim=32,
    )
    target_clean = cond.hidden_states[:, : cond.target_tokens_count]
    loss = compute_multi_reference_rf_loss(
        model=dummy,
        cond_inputs=cond,
        target_clean=target_clean,
    )
    assert torch.isfinite(loss)
    loss.backward()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "flux2_lora.safetensors"
        save_flux2_klein_lora(dummy, save_path)
        assert save_path.is_file()

        loaded = load_flux2_klein_lora(dummy, save_path)
        assert len(loaded) > 0

    merge_flux2_klein_lora(dummy)


def test_smolexpert_training_on_flux2_klein_features():
    feature_dim = 32
    decoder = _make_tiny_expert(input_channels=feature_dim)
    context = torch.randn(2, 48, feature_dim)
    state = torch.randn(2, ACTION_DIM)
    action = torch.randn(2, ACTION_HORIZON, ACTION_DIM)

    loss = decoder.flow_matching_loss(state=state, action=action, context=context)
    assert torch.isfinite(loss)
    assert loss.ndim == 0
    loss.backward()

    with torch.no_grad():
        actions = decoder.sample_actions(state=state[:1], context=context[:1], num_steps=2)
        assert actions.shape == (1, ACTION_HORIZON, ACTION_DIM)
