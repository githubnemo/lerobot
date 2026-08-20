"""Trainable adapters that normalize backbone context widths for World2Action.

The action decoder's attention width is intentionally fixed at 2048. A frozen
backbone may expose another width (LTX-2.5 exposes 4096), so the adapter is a
small trainable boundary: LayerNorm in the source width followed by one linear
projection. Matching widths use a true identity by default, preserving the
existing Cosmos-2B path exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

DEFAULT_DECODER_CONTEXT_WIDTH = 2048


@dataclass(frozen=True, slots=True)
class ContextAdapterSpec:
    """Serializable adapter dimensions and parameter accounting."""

    input_width: int
    output_width: int = DEFAULT_DECODER_CONTEXT_WIDTH
    identity: bool = False

    @property
    def parameter_count(self) -> int:
        if self.identity:
            return 0
        # LayerNorm affine weight+bias, then Linear weight+bias.
        return 2 * self.input_width + self.input_width * self.output_width + self.output_width


class ContextAdapter(nn.Module):
    """Map frozen backbone tokens to the decoder's fixed context width.

    For ``input_width == output_width`` the default is ``nn.Identity`` rather
    than a learned or normalizing layer. This is important: the adapter must
    not alter the established Cosmos-2B representation when it is unnecessary.
    For a mismatch, the trainable parameter order is LayerNorm then Linear.
    """

    def __init__(
        self,
        input_width: int,
        output_width: int = DEFAULT_DECODER_CONTEXT_WIDTH,
        *,
        identity_when_matching: bool = True,
    ) -> None:
        super().__init__()
        if type(input_width) is not int or input_width <= 0:
            raise ValueError("input_width must be a positive integer")
        if type(output_width) is not int or output_width <= 0:
            raise ValueError("output_width must be a positive integer")
        if input_width == output_width and identity_when_matching:
            self.norm: nn.Module = nn.Identity()
            self.projection: nn.Module = nn.Identity()
            self.spec = ContextAdapterSpec(input_width, output_width, identity=True)
        else:
            self.norm = nn.LayerNorm(input_width)
            self.projection = nn.Linear(input_width, output_width)
            self.spec = ContextAdapterSpec(input_width, output_width, identity=False)

    @property
    def input_width(self) -> int:
        return self.spec.input_width

    @property
    def output_width(self) -> int:
        return self.spec.output_width

    @property
    def is_identity(self) -> bool:
        return self.spec.identity

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if not isinstance(context, torch.Tensor) or context.ndim != 3:
            raise ValueError("context must have shape [B, tokens, width]")
        if context.shape[-1] != self.input_width:
            raise ValueError(f"context width must be {self.input_width}, got {context.shape[-1]}")
        if not torch.is_floating_point(context):
            raise TypeError("context must be floating point")
        if not torch.isfinite(context).all().item():
            raise ValueError("context must contain only finite values")
        if self.is_identity:
            return context
        # The backbone is frozen; only this module should receive gradients.
        normalized = self.norm(context.detach())
        projection_weight = next(self.projection.parameters())
        output = self.projection(normalized.to(dtype=projection_weight.dtype))
        return output.to(dtype=context.dtype)


def context_adapter_parameter_delta(
    input_width: int,
    output_width: int = DEFAULT_DECODER_CONTEXT_WIDTH,
    *,
    identity_when_matching: bool = True,
) -> int:
    """Return the number of trainable parameters introduced by an adapter."""

    return ContextAdapter(
        input_width,
        output_width,
        identity_when_matching=identity_when_matching,
    ).spec.parameter_count
