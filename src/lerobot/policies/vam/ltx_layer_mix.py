"""Learned scalar mixing for aligned multi-depth LTX-2.5 features."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional

from .ltx_action import (
    LTX_CHANNELS,
    LTX_CONTEXT_TRANSFORMS as LTX_CONTEXT_TRANSFORMS,
    apply_ltx_context_transform as apply_ltx_context_transform,
    ltx_context_grid,
    pool4_ltx_context as pool4_ltx_context,
)

LTX_LAYER_PROBE_DEPTHS = (8, 14, 20, 26, 34, 40)
LTX_HIDDEN_WIDTH = LTX_CHANNELS


def ltx_layer_probe_tokens(context_transform: str) -> int:
    """Return tokens stored per tapped layer for ``context_transform``."""
    return math.prod(ltx_context_grid(context_transform))


class LTXLayerScalarMix(nn.Module):
    """Mix aligned layer contexts without increasing World2Action token count."""

    def __init__(
        self,
        layers: tuple[int, ...] = LTX_LAYER_PROBE_DEPTHS,
        *,
        hidden_width: int = LTX_HIDDEN_WIDTH,
        seed: int = 0,
        logit_init_std: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if (
            not layers
            or any(type(layer) is not int or layer < 0 for layer in layers)
            or tuple(sorted(set(layers))) != layers
        ):
            raise ValueError("layers must be a non-empty sorted tuple of unique non-negative integers")
        if type(hidden_width) is not int or hidden_width <= 0:
            raise ValueError("hidden_width must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not 0.0 <= logit_init_std < 1.0:
            raise ValueError("logit_init_std must be in [0, 1)")
        self.layers = layers
        self.hidden_width = hidden_width
        self.seed = seed
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_width, elementwise_affine=False) for _layer in layers)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        logits = torch.empty(len(layers), dtype=torch.float32)
        logits.normal_(mean=0.0, std=logit_init_std, generator=generator)
        self.logits = nn.Parameter(logits)
        self.gain = nn.Parameter(torch.ones((), dtype=torch.float32))

    def _normalized_components(
        self, contexts: Mapping[int, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if set(contexts) != set(self.layers):
            raise ValueError(
                f"context layer keys must be exactly {list(self.layers)}, got {sorted(contexts)}"
            )
        reference_shape: tuple[int, ...] | None = None
        normalized: list[torch.Tensor] = []
        for layer, norm in zip(self.layers, self.norms, strict=True):
            context = contexts[layer]
            if (
                not isinstance(context, torch.Tensor)
                or context.ndim != 3
                or context.shape[-1] != self.hidden_width
            ):
                shape = tuple(context.shape) if isinstance(context, torch.Tensor) else type(context)
                raise ValueError(
                    f"layer {layer} context must have shape [B, N, {self.hidden_width}], got {shape}"
                )
            if not torch.is_floating_point(context) or not torch.isfinite(context).all().item():
                raise ValueError(f"layer {layer} context must be finite floating point")
            if reference_shape is None:
                reference_shape = tuple(context.shape)
            elif tuple(context.shape) != reference_shape:
                raise ValueError("all layer contexts must have identical shapes")
            normalized.append(norm(context.float()))
        weights = torch.softmax(self.logits.float(), dim=0)
        return torch.stack(normalized, dim=0), weights

    def forward(self, contexts: Mapping[int, torch.Tensor]) -> torch.Tensor:
        normalized, weights = self._normalized_components(contexts)
        weighted = normalized * weights.view(-1, 1, 1, 1)
        return self.gain.float() * weighted.sum(dim=0)

    @torch.no_grad()
    def diagnostics(self, contexts: Mapping[int, torch.Tensor]) -> dict[str, dict[int, float] | float]:
        """Return softmax weights and mean per-token contribution norms."""

        normalized, weights = self._normalized_components(contexts)
        contributions = normalized * weights.view(-1, 1, 1, 1)
        norms = torch.linalg.vector_norm(contributions, dim=-1).mean(dim=(1, 2))
        return {
            "weights": {
                layer: float(weight.detach().cpu())
                for layer, weight in zip(self.layers, weights, strict=True)
            },
            "mean_contribution_norms": {
                layer: float(norm.detach().cpu()) for layer, norm in zip(self.layers, norms, strict=True)
            },
            "gain": float(self.gain.detach().cpu()),
        }


def _validate_mixer_shape(layers: tuple[int, ...], hidden_width: int, seed: int) -> None:
    if (
        not layers
        or any(type(layer) is not int or layer < 0 for layer in layers)
        or tuple(sorted(set(layers))) != layers
    ):
        raise ValueError("layers must be a non-empty sorted tuple of unique non-negative integers")
    if type(hidden_width) is not int or hidden_width <= 0:
        raise ValueError("hidden_width must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _normalized_layer_contexts(
    layers: tuple[int, ...],
    hidden_width: int,
    norms: nn.ModuleList,
    contexts: Mapping[int, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    if set(contexts) != set(layers):
        raise ValueError(f"context layer keys must be exactly {list(layers)}, got {sorted(contexts)}")
    reference_shape: tuple[int, ...] | None = None
    normalized: list[torch.Tensor] = []
    for layer, norm in zip(layers, norms, strict=True):
        context = contexts[layer]
        if not isinstance(context, torch.Tensor) or context.ndim != 3 or context.shape[-1] != hidden_width:
            shape = tuple(context.shape) if isinstance(context, torch.Tensor) else type(context)
            raise ValueError(f"layer {layer} context must have shape [B, N, {hidden_width}], got {shape}")
        if not torch.is_floating_point(context) or not torch.isfinite(context).all().item():
            raise ValueError(f"layer {layer} context must be finite floating point")
        if reference_shape is None:
            reference_shape = tuple(context.shape)
        elif tuple(context.shape) != reference_shape:
            raise ValueError("all layer contexts must have identical shapes")
        normalized.append(norm(context.float()))
    return tuple(normalized)


class LTXLayerAttentionMix(nn.Module):
    """Collapse the layer axis with learned-query multi-head attention.

    Each token position treats its normalized layer vectors as a length-``L``
    sequence. A single learned query of shape ``(attn_width,)`` is broadcast to
    every batch/token position, so no tapped layer is privileged as the query
    and attention directly produces one vector per token rather than ``L``
    outputs requiring a second reduction.

    Q and K use small-std initialization, making initial attention weights
    nearly uniform. Because the default 2048-wide value bottleneck cannot be an
    identity map on 4096 channels, the projected attention is a small residual
    around the exact normalized-layer mean: V uses fan-in initialization and O
    uses a 1e-4 standard deviation. This starts close to scalar uniform mixing
    while retaining non-zero gradients through Q, K, V, and O.
    """

    def __init__(
        self,
        layers: tuple[int, ...] = LTX_LAYER_PROBE_DEPTHS,
        *,
        hidden_width: int = LTX_HIDDEN_WIDTH,
        attn_width: int = 2048,
        num_heads: int = 8,
        seed: int = 0,
        qk_init_std: float = 1.0e-3,
        output_init_std: float = 1.0e-4,
    ) -> None:
        super().__init__()
        _validate_mixer_shape(layers, hidden_width, seed)
        if type(attn_width) is not int or attn_width <= 0:
            raise ValueError("attn_width must be a positive integer")
        if type(num_heads) is not int or num_heads <= 0 or attn_width % num_heads:
            raise ValueError("num_heads must be positive and divide attn_width")
        if not 0.0 < qk_init_std < 1.0 or not 0.0 < output_init_std < 1.0:
            raise ValueError("initialization standard deviations must be in (0, 1)")
        self.layers = layers
        self.hidden_width = hidden_width
        self.attn_width = attn_width
        self.num_heads = num_heads
        self.head_width = attn_width // num_heads
        self.seed = seed
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_width, elementwise_affine=False) for _layer in layers)
        self.query = nn.Parameter(torch.empty(attn_width, dtype=torch.float32))
        self.q_proj = nn.Linear(attn_width, attn_width, bias=False)
        self.k_proj = nn.Linear(hidden_width, attn_width, bias=False)
        self.v_proj = nn.Linear(hidden_width, attn_width, bias=False)
        self.out_proj = nn.Linear(attn_width, hidden_width, bias=False)
        self.gain = nn.Parameter(torch.ones((), dtype=torch.float32))

        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.query.normal_(mean=0.0, std=2.0e-2, generator=generator)
            self.q_proj.weight.normal_(mean=0.0, std=qk_init_std, generator=generator)
            self.k_proj.weight.normal_(mean=0.0, std=qk_init_std, generator=generator)
            self.v_proj.weight.normal_(mean=0.0, std=hidden_width**-0.5, generator=generator)
            self.out_proj.weight.normal_(mean=0.0, std=output_init_std, generator=generator)

    def _attention_inputs(
        self, contexts: Mapping[int, torch.Tensor]
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
        components = _normalized_layer_contexts(self.layers, self.hidden_width, self.norms, contexts)
        stacked = torch.stack(components, dim=2)
        batch, tokens, layer_count, _channels = stacked.shape
        keys = self.k_proj(stacked)
        values = self.v_proj(stacked)
        query = self.q_proj(self.query)
        attention_dtype = torch.bfloat16 if stacked.device.type == "cuda" else keys.dtype
        keys = keys.to(dtype=attention_dtype)
        values = values.to(dtype=attention_dtype)
        query = query.to(dtype=attention_dtype)
        keys = (
            keys.reshape(batch, tokens, layer_count, self.num_heads, self.head_width)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch * tokens, self.num_heads, layer_count, self.head_width)
        )
        values = (
            values.reshape(batch, tokens, layer_count, self.num_heads, self.head_width)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch * tokens, self.num_heads, layer_count, self.head_width)
        )
        query = query.reshape(1, self.num_heads, 1, self.head_width).expand(batch * tokens, -1, -1, -1)
        return components, query, keys, values

    def forward(self, contexts: Mapping[int, torch.Tensor]) -> torch.Tensor:
        components, query, keys, values = self._attention_inputs(contexts)
        batch, tokens, _channels = components[0].shape
        attended = functional.scaled_dot_product_attention(
            query, keys, values, dropout_p=0.0, is_causal=False
        )
        attended = attended.squeeze(2).reshape(batch, tokens, self.attn_width)
        residual = self.out_proj(attended).float()
        uniform_mean = sum(components) / len(components)
        return self.gain.float() * (uniform_mean + residual)

    @torch.no_grad()
    def diagnostics(self, contexts: Mapping[int, torch.Tensor]) -> dict[str, object]:
        """Return mean layer weights, contribution norms, gain, and token dispersion."""
        accumulator = LTXAttentionDiagnosticsAccumulator(self)
        accumulator.update(contexts)
        return accumulator.compute()


class LTXAttentionDiagnosticsAccumulator:
    """Aggregate exact attention diagnostics across validation minibatches."""

    def __init__(self, mixer: LTXLayerAttentionMix) -> None:
        self.mixer = mixer
        shape = (mixer.num_heads, len(mixer.layers))
        self.weight_sums = torch.zeros(shape, dtype=torch.float64)
        self.weight_square_sums = torch.zeros(shape, dtype=torch.float64)
        self.contribution_norm_sums = torch.zeros(len(mixer.layers), dtype=torch.float64)
        self.positions = 0

    @torch.no_grad()
    def update(self, contexts: Mapping[int, torch.Tensor]) -> None:
        components, query, keys, values = self.mixer._attention_inputs(contexts)
        batch, tokens, _channels = components[0].shape
        layer_count = len(self.mixer.layers)
        scores = (query.float() * keys.float()).sum(dim=-1) / math.sqrt(self.mixer.head_width)
        attention_weights = torch.softmax(scores, dim=-1)
        token_weights = attention_weights.reshape(batch, tokens, self.mixer.num_heads, layer_count)
        token_weights64 = token_weights.double().cpu()
        self.weight_sums += token_weights64.sum(dim=(0, 1))
        self.weight_square_sums += token_weights64.square().sum(dim=(0, 1))
        for index, component in enumerate(components):
            weighted_value = attention_weights[..., index].unsqueeze(-1) * values[:, :, index]
            projected = self.mixer.out_proj(
                weighted_value.reshape(batch, tokens, self.mixer.attn_width)
            ).float()
            contribution = component / layer_count + projected
            self.contribution_norm_sums[index] += (
                torch.linalg.vector_norm(contribution, dim=-1).sum().double().cpu()
            )
        self.positions += batch * tokens

    def compute(self) -> dict[str, object]:
        if self.positions <= 0:
            raise ValueError("attention diagnostics require at least one validation position")
        head_layer_means = self.weight_sums / self.positions
        mean_weights = head_layer_means.mean(dim=0)
        variances = (self.weight_square_sums / self.positions - head_layer_means.square()).clamp_min(0.0)
        dispersion = variances.sqrt().mean()
        contribution_norms = self.contribution_norm_sums / self.positions
        weight_sum = float(mean_weights.sum().item())
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
            raise ValueError(f"mean attention weights must sum to one, got {weight_sum}")
        return {
            "weights": {
                layer: float(weight) for layer, weight in zip(self.mixer.layers, mean_weights, strict=True)
            },
            "mean_contribution_norms": {
                layer: float(norm) for layer, norm in zip(self.mixer.layers, contribution_norms, strict=True)
            },
            "gain": float(self.mixer.gain.detach().cpu()),
            "weight_dispersion": float(dispersion),
        }


class LTXLayerGatedMix(nn.Module):
    """Content-dependent token gating over normalized layer vectors.

    The gate consumes the mean normalized feature across layers and emits one
    layer logit per token. With ``per_channel=True``, learned ``(L, C)``
    log-weights are added to the broadcast content logits, then softmax is
    taken over L independently for every token and channel.

    A full BF16 ``(B, 640, L, 4096)`` tensor is about 250 MB at B=8 and L=6
    (about 500 MB in FP32). The per-channel path therefore processes token
    chunks and computes the layer softmax in streaming passes, keeping peak
    temporary storage proportional to ``(B, chunk_tokens, C)`` rather than L
    times that size.
    """

    def __init__(
        self,
        layers: tuple[int, ...] = LTX_LAYER_PROBE_DEPTHS,
        *,
        hidden_width: int = LTX_HIDDEN_WIDTH,
        per_channel: bool = False,
        seed: int = 0,
        gate_init_std: float = 1.0e-4,
        per_channel_chunk_tokens: int = 32,
    ) -> None:
        super().__init__()
        _validate_mixer_shape(layers, hidden_width, seed)
        if type(per_channel) is not bool:
            raise ValueError("per_channel must be a bool")
        if not 0.0 < gate_init_std < 1.0:
            raise ValueError("gate_init_std must be in (0, 1)")
        if type(per_channel_chunk_tokens) is not int or per_channel_chunk_tokens <= 0:
            raise ValueError("per_channel_chunk_tokens must be a positive integer")
        self.layers = layers
        self.hidden_width = hidden_width
        self.per_channel = per_channel
        self.seed = seed
        self.per_channel_chunk_tokens = per_channel_chunk_tokens
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_width, elementwise_affine=False) for _layer in layers)
        self.gate = nn.Linear(hidden_width, len(layers), bias=False)
        self.gain = nn.Parameter(torch.ones((), dtype=torch.float32))
        if per_channel:
            self.channel_logits = nn.Parameter(torch.zeros((len(layers), hidden_width), dtype=torch.float32))
        else:
            self.register_parameter("channel_logits", None)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.gate.weight.normal_(mean=0.0, std=gate_init_std, generator=generator)

    def _gate_inputs(
        self, contexts: Mapping[int, torch.Tensor]
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        components = _normalized_layer_contexts(self.layers, self.hidden_width, self.norms, contexts)
        gate_input = sum(components) / len(components)
        return components, self.gate(gate_input).float()

    def _channel_softmax_terms(self, gate_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.channel_logits is None:
            raise RuntimeError("per-channel softmax requested without channel logits")
        maximum: torch.Tensor | None = None
        for index in range(len(self.layers)):
            combined = gate_logits[..., index, None] + self.channel_logits[index]
            maximum = combined if maximum is None else torch.maximum(maximum, combined)
        if maximum is None:
            raise RuntimeError("layer list unexpectedly empty")
        denominator = torch.zeros_like(maximum)
        for index in range(len(self.layers)):
            combined = gate_logits[..., index, None] + self.channel_logits[index]
            denominator = denominator + torch.exp(combined - maximum)
        return maximum, denominator

    def forward(self, contexts: Mapping[int, torch.Tensor]) -> torch.Tensor:
        components, gate_logits = self._gate_inputs(contexts)
        if not self.per_channel:
            weights = torch.softmax(gate_logits, dim=-1)
            mixed = sum(component * weights[..., index, None] for index, component in enumerate(components))
            return self.gain.float() * mixed

        chunks: list[torch.Tensor] = []
        for start in range(0, gate_logits.shape[1], self.per_channel_chunk_tokens):
            stop = min(start + self.per_channel_chunk_tokens, gate_logits.shape[1])
            chunk_logits = gate_logits[:, start:stop]
            maximum, denominator = self._channel_softmax_terms(chunk_logits)
            mixed = torch.zeros_like(components[0][:, start:stop])
            for index, component in enumerate(components):
                combined = chunk_logits[..., index, None] + self.channel_logits[index]
                weight = torch.exp(combined - maximum) / denominator
                mixed = mixed + component[:, start:stop] * weight
            chunks.append(mixed)
        return self.gain.float() * torch.cat(chunks, dim=1)

    @torch.no_grad()
    def diagnostics(self, contexts: Mapping[int, torch.Tensor]) -> dict[str, object]:
        """Return mean layer weights, contribution norms, gain, and token dispersion."""

        components, gate_logits = self._gate_inputs(contexts)
        layer_count = len(self.layers)
        batch, tokens, _channels = components[0].shape
        if not self.per_channel:
            weights = torch.softmax(gate_logits, dim=-1)
            mean_weights = weights.mean(dim=(0, 1))
            flattened = weights.permute(2, 0, 1).reshape(layer_count, batch * tokens)
            dispersion = flattened.std(dim=-1, unbiased=False).mean()
            norms = [
                torch.linalg.vector_norm(component * weights[..., index, None], dim=-1).mean()
                for index, component in enumerate(components)
            ]
        else:
            weight_sums = torch.zeros(
                (layer_count, self.hidden_width), device=gate_logits.device, dtype=torch.float32
            )
            weight_square_sums = torch.zeros_like(weight_sums)
            norm_sums = torch.zeros(layer_count, device=gate_logits.device, dtype=torch.float32)
            for start in range(0, tokens, self.per_channel_chunk_tokens):
                stop = min(start + self.per_channel_chunk_tokens, tokens)
                chunk_logits = gate_logits[:, start:stop]
                maximum, denominator = self._channel_softmax_terms(chunk_logits)
                for index, component in enumerate(components):
                    combined = chunk_logits[..., index, None] + self.channel_logits[index]
                    weight = torch.exp(combined - maximum) / denominator
                    weight_sums[index] += weight.sum(dim=(0, 1))
                    weight_square_sums[index] += weight.square().sum(dim=(0, 1))
                    norm_sums[index] += torch.linalg.vector_norm(
                        component[:, start:stop] * weight, dim=-1
                    ).sum()
            position_count = batch * tokens
            channel_means = weight_sums / position_count
            channel_variances = (weight_square_sums / position_count - channel_means.square()).clamp_min(0.0)
            mean_weights = channel_means.mean(dim=-1)
            dispersion = channel_variances.sqrt().mean()
            norms = list(norm_sums / position_count)
        return {
            "weights": {
                layer: float(weight.detach().cpu())
                for layer, weight in zip(self.layers, mean_weights, strict=True)
            },
            "mean_contribution_norms": {
                layer: float(norm.detach().cpu()) for layer, norm in zip(self.layers, norms, strict=True)
            },
            "gain": float(self.gain.detach().cpu()),
            "weight_dispersion": float(dispersion.detach().cpu()),
        }
