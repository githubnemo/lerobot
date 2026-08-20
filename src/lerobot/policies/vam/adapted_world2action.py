"""World2Action composition with a trainable frozen-backbone context adapter.

The existing decoder intentionally detaches its context input because Cosmos
features are frozen. This wrapper preserves that decoder while placing the
trainable adapter inside the denoiser boundary, after the detach. Gradients
reach the LayerNorm and projection but never reach a video backbone or cached
feature tensor.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .context_adapter import ContextAdapter
from .world2action import World2ActionConfig, World2ActionDecoder


class _AdaptedDenoiser(nn.Module):
    def __init__(self, denoiser: nn.Module | Any, adapter: ContextAdapter) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.adapter = adapter

    def forward(self, **kwargs: Any) -> Any:
        kwargs["crossattn_emb"] = self.adapter(kwargs["crossattn_emb"])
        return self.denoiser(**kwargs)


class AdaptedWorld2ActionDecoder(World2ActionDecoder):
    """Existing action decoder plus a trainable source-width context adapter."""

    def __init__(
        self,
        backbone_width: int,
        config: World2ActionConfig | None = None,
        *,
        denoiser: nn.Module | Any | None = None,
        normalizer: Any | None = None,
        scheduler: Any | None = None,
        identity_when_matching: bool = True,
    ) -> None:
        adapter = ContextAdapter(
            backbone_width,
            output_width=2048,
            identity_when_matching=identity_when_matching,
        )
        wrapped_denoiser = None if denoiser is None else _AdaptedDenoiser(denoiser, adapter)
        super().__init__(config, denoiser=wrapped_denoiser, normalizer=normalizer, scheduler=scheduler)
        if denoiser is None:
            # The base constructor built a native decoder; place it behind the
            # adapter without rebuilding or changing its weights.
            base_denoiser = self.denoiser
            self.denoiser = _AdaptedDenoiser(base_denoiser, adapter)

    @property
    def context_adapter(self) -> ContextAdapter:
        return self.denoiser.adapter

    def _validate_inputs(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        context_timestep: torch.Tensor | None,
        action: torch.Tensor | None = None,
    ) -> tuple[int, torch.Tensor | None]:
        if not isinstance(context, torch.Tensor) or context.ndim != 3:
            raise ValueError("backbone context must have shape [B, tokens, width]")
        if context.shape[-1] != self.context_adapter.input_width:
            raise ValueError(
                f"backbone context width must be {self.context_adapter.input_width}, got {context.shape[-1]}"
            )
        if context.dtype != torch.bfloat16:
            raise TypeError("backbone context must use frozen bfloat16 hidden tokens")
        # The parent validator still owns all state/action/scheduler shape
        # checks; this empty tensor only presents its fixed decoder width.
        decoder_context = context.new_zeros((*context.shape[:-1], 2048))
        return super()._validate_inputs(state, decoder_context, context_timestep, action)
