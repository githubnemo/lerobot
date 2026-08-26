#!/usr/bin/env python

import pytest
import torch

from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.modeling_fastwam import _prompt_from_batch
from lerobot.policies.fastwam.text_context import (
    TextContextProvenance,
    encode_wan_text_context,
    format_task_prompts,
    load_text_context_artifact,
    save_text_context_artifact,
)


class TinyTokenizer:
    def __call__(self, prompts, *, return_mask, add_special_tokens):
        assert return_mask and add_special_tokens
        del add_special_tokens
        lengths = [2, 3, 1]
        ids = torch.zeros((len(prompts), 4), dtype=torch.long)
        mask = torch.zeros((len(prompts), 4), dtype=torch.bool)
        for row, prompt in enumerate(prompts):
            length = lengths[row % len(lengths)]
            ids[row, :length] = torch.arange(1, length + 1) + len(prompt)
            mask[row, :length] = True
        return ids, mask


class TinyEncoder:
    def __call__(self, ids, mask):
        del mask
        return ids.to(dtype=torch.float32).unsqueeze(-1).expand(*ids.shape, 3)


def _provenance(template="wrapped: {task}"):
    return TextContextProvenance(
        model_id="model",
        tokenizer_model_id="tokenizer",
        text_encoder_model_id="encoder",
        dtype="float32",
        prompt_template=template,
        tokenizer_max_len=4,
        context_len=4,
        context_dim=3,
    )


def test_cached_context_is_exact_online_context(tmp_path):
    prompts = ["wrapped: take cube out", "wrapped: put cube in"]
    online_context, online_mask = encode_wan_text_context(
        TinyTokenizer(), TinyEncoder(), prompts, device="cpu"
    )
    path = tmp_path / "context.safetensors"
    save_text_context_artifact(path, online_context, online_mask, prompts, _provenance())

    artifact = load_text_context_artifact(path, expected_provenance=_provenance())
    cached_context, cached_mask = artifact.lookup([prompts[1], prompts[0]])
    assert torch.equal(cached_context[0], online_context[1])
    assert torch.equal(cached_context[1], online_context[0])
    assert torch.equal(cached_mask[0], online_mask[1])
    assert torch.equal(cached_mask[1], online_mask[0])
    assert torch.equal(online_context[:, 3], torch.zeros_like(online_context[:, 3]))


def test_prompt_format_lookup_and_missing_prompt(tmp_path):
    prompts = format_task_prompts(["take cube out", "take cube out"], "wrapped: {task}")
    assert prompts == ["wrapped: take cube out"]
    context = torch.ones((1, 4, 3), dtype=torch.float32)
    mask = torch.ones((1, 4), dtype=torch.bool)
    path = tmp_path / "context.safetensors"
    save_text_context_artifact(path, context, mask, prompts, _provenance())
    artifact = load_text_context_artifact(path)
    with pytest.raises(KeyError, match="missing formatted prompt"):
        artifact.lookup(["wrapped: missing"])


def test_provenance_mismatch_fails_before_lookup(tmp_path):
    path = tmp_path / "context.safetensors"
    save_text_context_artifact(
        path,
        torch.ones((1, 4, 3), dtype=torch.float32),
        torch.ones((1, 4), dtype=torch.bool),
        ["wrapped: task"],
        _provenance(),
    )
    mismatched = TextContextProvenance(**{**_provenance().as_dict(), "tokenizer_max_len": 8})
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_text_context_artifact(path, expected_provenance=mismatched)


def test_default_online_configuration_compatibility():
    config = FastWAMConfig(base_model_id=None)
    assert config.load_text_encoder is True
    assert config.text_context_path is None
    assert _prompt_from_batch({"task": ["take cube out"]}, config) == [
        config.prompt_template.format(task="take cube out")
    ]


def test_disabled_text_encoder_requires_artifact_path():
    with pytest.raises(ValueError, match="requires `text_context_path`"):
        FastWAMConfig(load_text_encoder=False, base_model_id=None)
