import argparse
import json

import pytest
import torch
from torch import nn

from lerobot.policies.vam.ltx_action import LTXContextAdapter, frame_mean_ltx_context
from lerobot.policies.vam.ltx_prompt_embedding import (
    LTXPromptArtifact,
    LTXPromptEmbeddingError,
    build_provenance,
    load_ltx_prompt_artifact,
    save_ltx_prompt_artifact,
    validate_prompt_tensors,
)
from lerobot.policies.vam.world2action import World2ActionConfig, World2ActionDecoder
from scripts.video_vam.train_ltx_world2action_tiny import parse_windows, validate_normalizer_source


def prompt_tensors():
    embedding = torch.zeros((1, 1024, 4096), dtype=torch.bfloat16)
    embedding[:, :3] = 1
    mask = torch.zeros((1, 1024), dtype=torch.bool)
    mask[:, :3] = True
    return embedding, mask


def test_ltx_prompt_artifact_round_trip_and_provenance(tmp_path):
    embedding, mask = prompt_tensors()
    text_encoder = tmp_path / "gemma.safetensors"
    transformer = tmp_path / "transformer.safetensors"
    text_encoder.write_bytes(b"gemma")
    transformer.write_bytes(b"transformer")
    provenance = build_provenance(
        embedding,
        mask,
        text_encoder_path=text_encoder,
        transformer_path=transformer,
        generation_time_utc="2026-08-25T12:00:00Z",
    )
    output = tmp_path / "prompt.safetensors"
    save_ltx_prompt_artifact(LTXPromptArtifact(embedding, mask, provenance), output)

    loaded = load_ltx_prompt_artifact(output)

    assert torch.equal(loaded.embedding, embedding)
    assert torch.equal(loaded.attention_mask, mask)
    assert loaded.provenance.valid_tokens == 3
    assert loaded.provenance.prompt == "take cube out of box"
    assert loaded.provenance.gemma_version == "gemma4-12b-ltx-v1"


def test_ltx_prompt_artifact_never_accepts_zero_fallback():
    embedding, mask = prompt_tensors()
    with pytest.raises(LTXPromptEmbeddingError, match="zero prompt"):
        validate_prompt_tensors(torch.zeros_like(embedding), mask)


def test_ltx_prompt_artifact_detects_sidecar_tampering(tmp_path):
    embedding, mask = prompt_tensors()
    text_encoder = tmp_path / "gemma.safetensors"
    transformer = tmp_path / "transformer.safetensors"
    text_encoder.write_bytes(b"gemma")
    transformer.write_bytes(b"transformer")
    provenance = build_provenance(
        embedding,
        mask,
        text_encoder_path=text_encoder,
        transformer_path=transformer,
        generation_time_utc="2026-08-25T12:00:00Z",
    )
    output = tmp_path / "prompt.safetensors"
    save_ltx_prompt_artifact(LTXPromptArtifact(embedding, mask, provenance), output)
    sidecar = output.with_suffix(".json")
    payload = json.loads(sidecar.read_text())
    payload["prompt"] = ""
    sidecar.write_text(json.dumps(payload))

    with pytest.raises(LTXPromptEmbeddingError, match="prompt"):
        load_ltx_prompt_artifact(output)


def test_ltx_context_adapter_shapes_and_backpropagates():
    hidden = torch.randn((2, 8, 15, 20, 4096), dtype=torch.float32)
    frame_context = frame_mean_ltx_context(hidden)
    adapter = LTXContextAdapter()

    output = adapter(frame_context)
    output.square().mean().backward()

    assert frame_context.shape == (2, 8, 4096)
    assert output.shape == (2, 8, 2048)
    assert adapter.projection.weight.grad is not None
    assert torch.isfinite(adapter.projection.weight.grad).all()


class ConsumerDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, **kwargs):
        action = kwargs["xt_B_HA_A"]
        crossattn_emb = kwargs["crossattn_emb"]
        batch = action.shape[0]
        context_value = crossattn_emb.mean(dim=(1, 2))[:, None, None]
        native = torch.cat((torch.zeros_like(action[:, :1]), action), dim=1)
        return native * self.scale + context_value.expand(batch, 31, 6)


def test_ltx_adapter_is_real_world2action_loss_consumer():
    adapter = LTXContextAdapter()
    decoder = World2ActionDecoder(
        World2ActionConfig(device="cpu", dtype=torch.float32),
        denoiser=ConsumerDenoiser(),
    )
    state = torch.zeros((1, 1, 6), dtype=torch.float32)
    action = torch.ones((1, 30, 6), dtype=torch.float32)
    padding = torch.zeros((1, 30), dtype=torch.bool)
    context = adapter(torch.randn((1, 8, 4096), dtype=torch.float32)).to(torch.bfloat16)

    loss = decoder.flow_matching_loss(
        state,
        action,
        context,
        t=torch.tensor([0.5]),
        epsilon=torch.zeros_like(action),
        action_is_pad=padding,
        context_timestep=torch.ones((1, 1)),
        obs_dropout=0.0,
        detach_context=False,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert adapter.projection.weight.grad is not None
    assert decoder.denoiser.scale.grad is not None


def test_window_parser_requires_small_distinct_causal_set():
    assert parse_windows("0:4,1:8") == parse_windows("0:4,1:8")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_windows("0:4")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_windows("0:4,0:4")


def test_normalizer_source_must_match_established_cosmos_contract():
    payload = {
        "anchor_count": 4688,
        "derivation": "all_valid_training_episode_anchors",
        "episodes": list(range(32)),
        "padded_actions_excluded": True,
    }
    assert validate_normalizer_source(payload) == payload
    with pytest.raises(ValueError, match="normalizer provenance"):
        validate_normalizer_source({**payload, "episodes": [0]})
