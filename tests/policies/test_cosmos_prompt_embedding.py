import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.policies.vam.cosmos_prompt_embedding import (
    ATTENTION_MASK_KEY,
    DEFAULT_MAX_LENGTH,
    EMBEDDING_KEY,
    EXPECTED_EMBEDDING_SHAPE,
    PromptEmbeddingArtifact,
    PromptEmbeddingValidationError,
    build_provenance,
    generate_prompt_embedding,
    load_prompt_embedding,
    save_prompt_embedding,
    verify_t5_model,
)


class FakeTokenizer:
    def __init__(self, token_length: int = 7, use_batch_encode_plus: bool = True):
        self.token_length = token_length
        self.returned_length = DEFAULT_MAX_LENGTH
        self.calls = []
        if not use_batch_encode_plus:
            self.batch_encode_plus = None

    def _encode(self):
        input_ids = torch.arange(DEFAULT_MAX_LENGTH, dtype=torch.int64).reshape(1, DEFAULT_MAX_LENGTH)
        attention_mask = torch.zeros((1, DEFAULT_MAX_LENGTH), dtype=torch.int64)
        attention_mask[:, : self.token_length] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # Transformers 5 returns the padded sequence length here; production
            # code must derive content length from attention_mask instead.
            "length": torch.tensor([self.returned_length]),
        }

    def batch_encode_plus(self, prompts, **kwargs):
        self.calls.append(("batch_encode_plus", prompts, kwargs))
        return self._encode()

    def __call__(self, prompts, **kwargs):
        self.calls.append(("__call__", prompts, kwargs))
        return self._encode()


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embeddings = nn.Embedding(DEFAULT_MAX_LENGTH, 4)
        self.seen = None

    def get_input_embeddings(self):
        return self.input_embeddings

    def forward(self, *, input_ids, attention_mask):
        self.seen = (input_ids, attention_mask)
        hidden = torch.ones(EXPECTED_EMBEDDING_SHAPE, dtype=torch.float32, device=input_ids.device)
        return SimpleNamespace(last_hidden_state=hidden)


def generated_artifact(tmp_path):
    tokenizer = FakeTokenizer(token_length=7)
    model = FakeModel()
    generation = generate_prompt_embedding(tokenizer, model)
    provenance = build_provenance(
        generation,
        model_path=tmp_path / "t5-11b",
        transformers_version="5.5.4",
        torch_version="2.11.0+cu128",
        generation_time_utc="2026-08-17T12:00:00Z",
    )
    return (
        PromptEmbeddingArtifact(generation.embedding, generation.attention_mask, provenance),
        tokenizer,
        model,
    )


def test_both_tokenizer_api_branches_preserve_contract_and_hashes(tmp_path):
    tokenization_kwargs = {
        "return_tensors": "pt",
        "truncation": True,
        "padding": "max_length",
        "max_length": 512,
        "return_length": True,
    }
    generations = []
    provenances = []
    for use_batch_encode_plus in (True, False):
        tokenizer = FakeTokenizer(use_batch_encode_plus=use_batch_encode_plus)
        model = FakeModel()
        generation = generate_prompt_embedding(tokenizer, model)
        expected_method = "batch_encode_plus" if use_batch_encode_plus else "__call__"
        assert tokenizer.calls == [(expected_method, ["take cube out of box"], tokenization_kwargs)]
        assert model.seen[0].device == model.input_embeddings.weight.device
        assert model.seen[1].device == model.input_embeddings.weight.device
        assert tokenizer.returned_length == DEFAULT_MAX_LENGTH
        assert generation.attention_mask.sum().item() == 7
        assert torch.equal(generation.embedding[0, 7:], torch.zeros_like(generation.embedding[0, 7:]))
        generations.append(generation)
        provenances.append(
            build_provenance(
                generation,
                model_path=tmp_path / "t5-11b",
                transformers_version="5.5.4",
                torch_version="2.11.0+cu128",
                generation_time_utc="2026-08-17T12:00:00Z",
            )
        )

    assert torch.equal(generations[0].token_ids, generations[1].token_ids)
    assert torch.equal(generations[0].attention_mask, generations[1].attention_mask)
    assert torch.equal(generations[0].embedding, generations[1].embedding)
    assert provenances[0].token_ids_sha256 == provenances[1].token_ids_sha256
    assert provenances[0].output_sha256 == provenances[1].output_sha256
    assert provenances[0].to_dict() == provenances[1].to_dict()


def test_positions_after_attention_mask_length_are_exactly_zero(tmp_path):
    artifact, _, _ = generated_artifact(tmp_path)

    assert artifact.embedding.dtype == torch.bfloat16
    assert artifact.embedding.shape == EXPECTED_EMBEDDING_SHAPE
    assert torch.equal(artifact.embedding[0, 7:], torch.zeros_like(artifact.embedding[0, 7:]))
    assert torch.all(artifact.embedding[0, :7] == 1)


def test_save_load_and_provenance_round_trip(tmp_path):
    artifact, _, _ = generated_artifact(tmp_path)
    output = tmp_path / "cube.safetensors"

    save_prompt_embedding(artifact, output)
    reloaded = load_prompt_embedding(output)

    assert torch.equal(reloaded.embedding, artifact.embedding)
    assert torch.equal(reloaded.attention_mask, artifact.attention_mask)
    payload = json.loads(output.with_suffix(".json").read_text())
    assert payload["prompt"] == "take cube out of box"
    assert payload["max_length"] == 512
    assert payload["tokenization_api"] == "batch_encode_plus_or_tokenizer_call_same_kwargs"
    assert payload["model_repo"] == "jonpai/mimic-video"
    assert payload["model_revision"] == "f28339034831e3c2374be075e622e1ff38ebe0f8"
    assert payload["t5_sha256"] == "5fdc64177b14b0f72437fea171a752c626733f67755842528c75b90cacd1c807"
    assert payload["mimic_commit"] == "e3355dbc93132b576c02f920a59b4fc18a4f5906"
    assert payload["dtype"] == "bfloat16"
    assert payload["shape"] == [1, 512, 1024]


def test_malformed_shape_is_rejected_on_load(tmp_path):
    output = tmp_path / "malformed.safetensors"
    save_file(
        {
            EMBEDDING_KEY: torch.zeros((1, 512, 1023), dtype=torch.bfloat16),
            ATTENTION_MASK_KEY: torch.ones((1, 512), dtype=torch.bool),
        },
        str(output),
    )

    with pytest.raises(PromptEmbeddingValidationError, match="shape"):
        load_prompt_embedding(output)


def test_overwrite_requires_explicit_opt_in(tmp_path):
    artifact, _, _ = generated_artifact(tmp_path)
    output = tmp_path / "cube.safetensors"
    save_prompt_embedding(artifact, output)

    with pytest.raises(FileExistsError, match="overwrite"):
        save_prompt_embedding(artifact, output)


def test_model_hash_guard_streams_and_can_only_be_skipped_explicitly(tmp_path):
    model_dir = tmp_path / "t5-11b"
    model_dir.mkdir()
    model_file = model_dir / "pytorch_model.bin"
    model_file.write_bytes(b"abc")

    with pytest.raises(PromptEmbeddingValidationError, match="SHA256 mismatch"):
        verify_t5_model(model_dir, expected_size=3, expected_sha256="0" * 64)

    verification = verify_t5_model(
        model_dir,
        skip_hash=True,
        expected_size=3,
        expected_sha256="0" * 64,
    )
    assert verification.hash_verified is False
    assert verification.sha256 == "0" * 64
