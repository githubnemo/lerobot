"""Synthetic CPU dry-run tests for Cosmos-1.0-Diffusion-14B feature extraction with streaming and FP8."""

from __future__ import annotations

import torch

from lerobot.policies.vam import (
    Cosmos14BExtractionOutput,
    Cosmos14BExtractor,
    Cosmos14BExtractorConfig,
    build_cosmos14b_dummy_model,
)
from lerobot.policies.vam.cosmos14b_extractor import FP8Linear


def test_fp8_linear_module():
    """Test that FP8Linear properly stores weights and performs accurate linear forward pass."""
    lin = torch.nn.Linear(64, 32, bias=True)
    fp8_lin = FP8Linear.from_linear(lin)
    assert fp8_lin.weight.dtype == torch.float8_e4m3fn
    assert fp8_lin.weight.shape == (32, 64)
    assert fp8_lin.bias.shape == (32,)

    x = torch.randn(2, 8, 64, dtype=torch.bfloat16)
    out = fp8_lin(x)
    assert out.shape == (2, 8, 32)
    assert out.dtype == torch.bfloat16


def test_cosmos14b_extractor_cpu_synthetic():
    """Verify Cosmos 14B extractor with block streaming and FP8 quantization on CPU."""
    num_layers = 4
    dummy = build_cosmos14b_dummy_model(
        num_layers=num_layers,
        num_attention_heads=2,
        attention_head_dim=16,
        text_embed_dim=16,
        device="cpu",
        dtype=torch.float32,
    )

    cfg = Cosmos14BExtractorConfig(
        hidden_layers=(1, 3),
        patch_size=(1, 2, 2),
        in_channels=17,
        out_channels=16,
        num_layers=num_layers,
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_dim=32,
        text_embed_dim=16,
        device="cpu",
        dtype="float32",
        pool_spatial=2,
        concat_layers=True,
        block_streaming=True,
        fp8_linear=True,
    )
    extractor = Cosmos14BExtractor(config=cfg, model=dummy)

    # Input: [B=1, C=17, T=1, H=8, W=8]
    hidden = torch.randn(1, 17, 1, 8, 8, dtype=torch.float32)
    out: Cosmos14BExtractionOutput = extractor.extract(hidden)

    # Spatial 8x8 with patch (1, 2, 2) -> (4, 4), pooled by 2 -> (2, 2) = 4 tokens
    # 2 layers concatenated: 32 + 32 = 64 feature_dim
    assert out.features.shape == (1, 4, 64)
    assert out.num_tokens == 4
    assert out.feature_dim == 64
    assert 1 in out.features_by_layer
    assert 3 in out.features_by_layer
    assert out.grid_coords.shape == (1, 4, 3)
    assert out.provenance["block_streaming"] is True
    assert out.provenance["fp8_linear"] is True
