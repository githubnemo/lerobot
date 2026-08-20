import sys
import types

import pytest
import torch

import lerobot.policies.vam._vendor.cosmos_predict2.models.text2image_dit as text2image_dit
import lerobot.policies.vam._vendor.cosmos_predict2.models.world2action_dit as world2action_dit

LOADER_MODULES = (world2action_dit, text2image_dit)


def install_fake_transformer_engine(monkeypatch, *, current_api):
    names = (
        "transformer_engine",
        "transformer_engine.pytorch",
        "transformer_engine.pytorch.attention",
        "transformer_engine.pytorch.attention.rope",
    )
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)

    te_module = types.ModuleType("transformer_engine")
    pytorch_module = types.ModuleType("transformer_engine.pytorch")
    attention_module = types.ModuleType("transformer_engine.pytorch.attention")
    rope_module = types.ModuleType("transformer_engine.pytorch.attention.rope")
    pytorch_module.__path__ = []
    attention_module.__path__ = []
    dot_product_attention = object()
    rotary_pos_emb = object()
    attention_module.DotProductAttention = dot_product_attention
    if current_api:
        rope_module.apply_rotary_pos_emb = rotary_pos_emb
    else:
        attention_module.apply_rotary_pos_emb = rotary_pos_emb
    te_module.pytorch = pytorch_module
    pytorch_module.attention = attention_module
    monkeypatch.setitem(sys.modules, "transformer_engine", te_module)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", pytorch_module)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.attention", attention_module)
    if current_api:
        monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.attention.rope", rope_module)
    return dot_product_attention, rotary_pos_emb


@pytest.mark.parametrize("loader_module", LOADER_MODULES)
def test_loader_supports_current_te_rope_api(monkeypatch, loader_module):
    dot_product_attention, rotary_pos_emb = install_fake_transformer_engine(monkeypatch, current_api=True)

    te_module, loaded_dot_product_attention, loaded_rotary_pos_emb = loader_module._load_transformer_engine()

    assert te_module.__name__ == "transformer_engine"
    assert loaded_dot_product_attention is dot_product_attention
    assert loaded_rotary_pos_emb is rotary_pos_emb


@pytest.mark.parametrize("loader_module", LOADER_MODULES)
def test_loader_falls_back_to_legacy_te_rotary_api(monkeypatch, loader_module):
    dot_product_attention, rotary_pos_emb = install_fake_transformer_engine(monkeypatch, current_api=False)

    _, loaded_dot_product_attention, loaded_rotary_pos_emb = loader_module._load_transformer_engine()

    assert loaded_dot_product_attention is dot_product_attention
    assert loaded_rotary_pos_emb is rotary_pos_emb


@pytest.mark.parametrize("loader_module", LOADER_MODULES)
def test_loader_reports_te_api_mismatch_separately(monkeypatch, loader_module):
    install_fake_transformer_engine(monkeypatch, current_api=False)
    attention_module = sys.modules["transformer_engine.pytorch.attention"]
    del attention_module.apply_rotary_pos_emb

    with pytest.raises(RuntimeError, match="API.*apply_rotary_pos_emb"):
        loader_module._load_transformer_engine()


def test_world_torch_attention_matches_sdpa_and_preserves_heads():
    functional = torch.nn.functional
    q = torch.randn(2, 3, 2, 4)
    k = torch.randn(2, 5, 2, 4)
    v = torch.randn(2, 5, 2, 4)
    expected = functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)

    actual = world2action_dit.torch_attention_op(q, k, v)

    assert actual.shape == (2, 3, 2, 4)
    assert torch.allclose(actual, expected)


class FakeRMSNorm(torch.nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, value):
        return value


def test_world_attention_wrapper_returns_flat_query_projection(monkeypatch):
    fake_te = types.SimpleNamespace(pytorch=types.SimpleNamespace(RMSNorm=FakeRMSNorm))
    monkeypatch.setattr(
        world2action_dit,
        "_load_transformer_engine",
        lambda: (fake_te, None, lambda value, rope, tensor_format, fused: value),
    )

    attention = world2action_dit.Attention(query_dim=8, n_heads=2, head_dim=4, backend="torch")
    output = attention(torch.randn(2, 3, 8))

    assert output.shape == (2, 3, 8)
