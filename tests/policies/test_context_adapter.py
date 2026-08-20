import torch

from lerobot.policies.vam.context_adapter import (
    DEFAULT_DECODER_CONTEXT_WIDTH,
    ContextAdapter,
    context_adapter_parameter_delta,
)


def test_ltx_adapter_shapes_and_parameter_delta():
    adapter = ContextAdapter(4096)
    context = torch.randn(2, 7, 4096, dtype=torch.bfloat16)
    output = adapter(context)
    assert output.shape == (2, 7, DEFAULT_DECODER_CONTEXT_WIDTH)
    assert output.dtype == torch.bfloat16
    assert adapter.spec.parameter_count == 8_398_848
    assert context_adapter_parameter_delta(4096) == 8_398_848
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 8_398_848
    assert all(parameter.requires_grad for parameter in adapter.parameters())


def test_matching_width_is_a_true_identity_without_parameters():
    adapter = ContextAdapter(2048)
    context = torch.randn(1, 3, 2048, dtype=torch.bfloat16)
    output = adapter(context)
    assert adapter.is_identity
    assert output is context
    assert torch.equal(output, context)
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 0


def test_matching_width_can_be_explicitly_normalized_when_requested():
    adapter = ContextAdapter(2048, identity_when_matching=False)
    output = adapter(torch.randn(1, 3, 2048, dtype=torch.float32))
    assert output.shape == (1, 3, 2048)
    assert not adapter.is_identity
