"""Unit tests for Cosmos-1.0-Diffusion-14B Quantized Video-LoRA implementation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from lerobot.policies.vam.cosmos14b_extractor import (
    build_cosmos14b_dummy_model,
)
from lerobot.policies.vam.cosmos14b_lora import (
    Cosmos14BLoRAConfig,
    Cosmos14BQuantizedLoRALinear,
    compute_rectified_flow_loss,
    get_cosmos14b_lora_parameters,
    inject_cosmos14b_quantized_lora,
    load_cosmos14b_lora,
    save_cosmos14b_lora,
)


def test_cosmos14b_lora_config_validation():
    """Verify config validation for rank, alpha, and quantization type."""
    cfg = Cosmos14BLoRAConfig(rank=16, alpha=16.0, quantization="fp8")
    assert cfg.rank == 16
    assert cfg.alpha == 16.0
    assert cfg.quantization == "fp8"

    with pytest.raises(ValueError, match="LoRA rank must be positive"):
        Cosmos14BLoRAConfig(rank=0)

    with pytest.raises(ValueError, match="LoRA alpha must be positive"):
        Cosmos14BLoRAConfig(alpha=-1.0)

    with pytest.raises(ValueError, match="Unsupported quantization"):
        Cosmos14BLoRAConfig(quantization="int4")


def test_cosmos14b_quantized_lora_linear_forward_backward():
    """Verify forward and backward execution through Cosmos14BQuantizedLoRALinear."""
    in_dim, out_dim, rank = 64, 32, 8
    lin = nn.Linear(in_dim, out_dim, bias=True)
    qlora = Cosmos14BQuantizedLoRALinear(lin, rank=rank, alpha=16.0, quantization="none", dtype=torch.float32)

    assert qlora.lora_A.shape == (rank, in_dim)
    assert qlora.lora_B.shape == (out_dim, rank)
    assert qlora.weight.requires_grad is False
    assert qlora.lora_A.requires_grad is True
    assert qlora.lora_B.requires_grad is True

    # Test forward
    x = torch.randn(2, 4, in_dim, requires_grad=True)
    out = qlora(x)
    assert out.shape == (2, 4, out_dim)

    # Test backward
    loss = out.sum()
    loss.backward()

    assert qlora.lora_B.grad is not None
    assert qlora.weight.grad is None
    assert x.grad is not None


def test_cosmos14b_lora_injection():
    """Test injection of Quantized LoRA into synthetic Cosmos 14B model."""
    num_layers = 2
    model = build_cosmos14b_dummy_model(
        num_layers=num_layers,
        num_attention_heads=2,
        attention_head_dim=16,
        text_embed_dim=16,
        device="cpu",
        dtype=torch.float32,
    )
    model.config.concat_padding_mask = False

    cfg = Cosmos14BLoRAConfig(
        rank=8,
        alpha=16.0,
        target_blocks=tuple(range(num_layers)),
        target_modules=("to_q", "to_k", "to_v", "to_out.0"),
        quantization="none",
    )
    injected = inject_cosmos14b_quantized_lora(model, cfg)
    assert len(injected) > 0

    lora_params = get_cosmos14b_lora_parameters(model)
    assert len(lora_params) == len(injected) * 2  # lora_A and lora_B for each module

    # All non-LoRA parameters must have requires_grad=False
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False


def test_cosmos14b_lora_save_and_load(tmp_path: Path):
    """Test checkpoint serialization and round-trip loading."""
    num_layers = 2
    model = build_cosmos14b_dummy_model(
        num_layers=num_layers,
        num_attention_heads=2,
        attention_head_dim=16,
        text_embed_dim=16,
        device="cpu",
        dtype=torch.float32,
    )
    cfg = Cosmos14BLoRAConfig(
        rank=4,
        alpha=8.0,
        target_blocks=(0,),
        target_modules=("to_q",),
        quantization="none",
    )
    inject_cosmos14b_quantized_lora(model, cfg)

    # Set non-zero values to test exact loading
    for p in get_cosmos14b_lora_parameters(model):
        p.data.fill_(0.42)

    save_path = tmp_path / "test_lora.safetensors"
    save_cosmos14b_lora(model, save_path)
    assert save_path.is_file()

    # Create fresh model and inject fresh LoRA
    model2 = build_cosmos14b_dummy_model(
        num_layers=num_layers,
        num_attention_heads=2,
        attention_head_dim=16,
        text_embed_dim=16,
        device="cpu",
        dtype=torch.float32,
    )
    inject_cosmos14b_quantized_lora(model2, cfg)
    load_cosmos14b_lora(model2, save_path)

    for p in get_cosmos14b_lora_parameters(model2):
        assert torch.allclose(p.data, torch.full_like(p.data, 0.42))


def test_cosmos14b_rectified_flow_loss():
    """Verify rectified flow loss calculation on dummy model."""
    num_layers = 2
    model = build_cosmos14b_dummy_model(
        num_layers=num_layers,
        num_attention_heads=2,
        attention_head_dim=16,
        text_embed_dim=16,
        device="cpu",
        dtype=torch.float32,
    )
    model.config.concat_padding_mask = False

    clean_latents = torch.randn(1, 17, 1, 8, 8)
    encoder_hidden_states = torch.randn(1, 4, 16)
    loss = compute_rectified_flow_loss(
        model=model,
        clean_latents=clean_latents,
        encoder_hidden_states=encoder_hidden_states,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
