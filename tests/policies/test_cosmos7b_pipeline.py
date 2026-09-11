"""Synthetic CPU dry-run tests for Cosmos-1.0-Diffusion-7B feature extraction and training pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from lerobot.policies.vam import (
    Cosmos7BExtractionOutput,
    Cosmos7BExtractor,
    Cosmos7BExtractorConfig,
    Cosmos7BLoRAConfig,
    build_cosmos7b_dummy_model,
    compute_cosmos7b_rf_loss,
    inject_cosmos7b_lora,
    load_cosmos7b_lora,
    merge_cosmos7b_lora,
    save_cosmos7b_lora,
)
from lerobot.policies.vam.smol_expert import (
    ACTION_DIM,
    ACTION_HORIZON,
    SmolExpertActionDecoder,
    SmolVLANormalizer,
)


def _make_tiny_expert(input_channels: int = 64) -> SmolExpertActionDecoder:
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


def test_cosmos7b_extractor_config_validation():
    with pytest.raises(ValueError, match="hidden_layer.*out of range"):
        Cosmos7BExtractorConfig(hidden_layers=(30,), num_layers=28)

    with pytest.raises(ValueError, match="hidden_dim.*must equal"):
        Cosmos7BExtractorConfig(
            num_attention_heads=32,
            attention_head_dim=128,
            hidden_dim=2048,  # Mismatch: 32 * 128 = 4096 != 2048
        )


def test_cosmos7b_feature_extraction_layers_14_and_20():
    # Construct dummy model with 22 layers to include layers 14 and 20
    dummy = build_cosmos7b_dummy_model(
        num_layers=22,
        num_attention_heads=2,
        attention_head_dim=16,
        text_embed_dim=32,
        device="cpu",
        dtype=torch.float32,
    )
    config = Cosmos7BExtractorConfig(
        hidden_layers=(14, 20),
        patch_size=(1, 2, 2),
        num_layers=22,
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_dim=32,
        text_embed_dim=32,
        pool_spatial=2,
        concat_layers=True,
    )
    extractor = Cosmos7BExtractor(config=config, model=dummy)

    # Input video latents: [B=1, C=16, T=2, H=8, W=8]
    latents = torch.randn(1, 16, 2, 8, 8, device="cpu", dtype=torch.float32)
    text_cond = torch.randn(1, 4, 32, device="cpu", dtype=torch.float32)

    out: Cosmos7BExtractionOutput = extractor.extract(
        hidden_states=latents,
        text_conditioning=text_cond,
    )

    # Verify layers captured
    assert set(out.features_by_layer.keys()) == {14, 20}
    # Each layer has hidden_dim = 32. Concat layers = 64
    assert out.features.shape[-1] == 64
    # Spatiotemporal grid shape: T=2, H=8//2=4//2(pool)=2, W=2 -> 2*2*2 = 8 tokens
    assert out.features.shape == (1, 8, 64)
    assert out.grid_coords.shape == (1, 8, 3)
    assert out.grid_shape == (2, 2, 2)
    # Check 3D grid formatting
    assert out.grid_features_by_layer[14].shape == (1, 2, 2, 2, 32)
    assert out.grid_features_by_layer[20].shape == (1, 2, 2, 2, 32)


def test_cosmos7b_video_lora_lifecycle():
    dummy = build_cosmos7b_dummy_model(
        num_layers=4,
        num_attention_heads=2,
        attention_head_dim=16,
        text_embed_dim=32,
        device="cpu",
    )
    lora_cfg = Cosmos7BLoRAConfig(
        rank=4,
        alpha=4.0,
        target_blocks=(0, 1),
        target_modules=("to_q", "to_k", "to_v", "to_out.0"),
    )
    injected = inject_cosmos7b_lora(dummy, lora_cfg)
    assert len(injected) > 0

    # Video RF diffusion loss forward & backward
    clean_latents = torch.randn(1, 16, 2, 4, 4)
    text_embeds = torch.randn(1, 2, 32)
    loss = compute_cosmos7b_rf_loss(
        model=dummy,
        clean_latents=clean_latents,
        encoder_hidden_states=text_embeds,
    )
    assert torch.isfinite(loss)
    loss.backward()

    # Save and load LoRA weights
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "cosmos7b_lora.safetensors"
        save_cosmos7b_lora(dummy, save_path)
        assert save_path.is_file()

        loaded = load_cosmos7b_lora(dummy, save_path)
        assert len(loaded) > 0

    # Merge LoRA
    merge_cosmos7b_lora(dummy)


def test_smolexpert_training_on_cosmos7b_features():
    feature_dim = 64
    decoder = _make_tiny_expert(input_channels=feature_dim)
    context = torch.randn(2, 8, feature_dim)
    state = torch.randn(2, ACTION_DIM)
    action = torch.randn(2, ACTION_HORIZON, ACTION_DIM)

    # Compute flow matching loss
    loss = decoder.flow_matching_loss(state=state, action=action, context=context)
    assert torch.isfinite(loss)
    assert loss.ndim == 0
    loss.backward()

    # Verify action sampling
    with torch.no_grad():
        actions = decoder.sample_actions(state=state[:1], context=context[:1], num_steps=2)
        assert actions.shape == (1, ACTION_HORIZON, ACTION_DIM)
