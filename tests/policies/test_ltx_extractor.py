import hashlib

import pytest
import torch
from torch import nn

from lerobot.policies.vam.ltx_extractor import (
    LTXExtractor,
    LTXExtractorConfig,
    arch_invariant_rand,
    derive_window_seed,
    matched_ltx_layer,
    matched_ltx_noise_level,
)


class FakeEncoder(nn.Module):
    def forward(self, video):
        assert video.shape == (1, 3, 9, 480, 640)
        return torch.zeros((1, 128, 2, 15, 20), dtype=video.dtype)


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, *, latent, prompt_embedding, sigma, denoise_mask, positions, layer_index):
        self.calls.append(
            {
                "latent": latent,
                "prompt_embedding": prompt_embedding,
                "sigma": sigma,
                "denoise_mask": denoise_mask,
                "positions": positions,
                "layer_index": layer_index,
            }
        )
        assert latent.shape == (1, 2400, 128)
        assert prompt_embedding.shape == (1, 4, 4096)
        assert sigma.tolist() == [1.0]
        assert denoise_mask[:, : 2 * 15 * 20].eq(0).all()
        assert denoise_mask[:, 2 * 15 * 20 :].eq(1).all()
        assert positions.shape == (1, 3, 2400, 2)
        return torch.full((1, 2400, 4096), 0.5, dtype=latent.dtype)


def test_noise_matches_cosmos_numpy_random_state_and_window_formula():
    expected = int.from_bytes(
        hashlib.sha256(b"revision\0" + b"3" + b"\0" + b"4" + b"\0" + b"17").digest()[:8], "big"
    ) % (2**32 - 1)
    assert derive_window_seed("revision", 3, 4, 17) == expected
    actual = arch_invariant_rand((2, 3), expected)
    reference = torch.from_numpy(
        __import__("numpy").random.RandomState(expected).standard_normal((2, 3)).astype("float32")
    )
    assert torch.equal(actual, reference)
    assert derive_window_seed("revision", 3, 4, 17) != derive_window_seed("revision", 3, 5, 17)


def test_layer_and_noise_selection_match_relative_cosmos_point():
    assert matched_ltx_layer() == 34
    assert matched_ltx_layer(10, 28, 48) == 17
    assert matched_ltx_noise_level() == 1.0


def test_extractor_pads_causal_window_and_records_full_provenance():
    fake_backbone = FakeBackbone()
    config = LTXExtractorConfig(
        device="cpu",
        dtype="bfloat16",
        checkpoint_sha256="a" * 64,
        video_vae_sha256="b" * 64,
    )
    extractor = LTXExtractor(config, backbone=fake_backbone, latent_encoder=FakeEncoder())
    extraction = extractor.extract_window(
        torch.zeros((1, 3, 5, 480, 640), dtype=torch.uint8),
        torch.zeros((1, 4, 4096), dtype=torch.bfloat16),
        dataset_revision="dataset-rev",
        episode_index=2,
        frame_index=11,
        global_seed=7,
    )
    assert extraction.hidden_grid.shape == (1, 8, 15, 20, 4096)
    assert extraction.tokens.shape == (1, 2400, 4096)
    # The word "tokens" trips the gitleaks generic-api-key rule on this dtype assertion.
    assert extraction.tokens.dtype == torch.bfloat16  # gitleaks:allow
    assert not extraction.tokens.requires_grad
    assert extraction.provenance.noise_seed == derive_window_seed("dataset-rev", 2, 11, 7)
    assert extraction.provenance.padded_input_shape == (1, 3, 9, 480, 640)
    assert extraction.provenance.token_geometry == (8, 15, 20)
    assert extraction.provenance.token_count == 2400
    assert extraction.provenance.target_frame_count == 57
    assert extraction.provenance.hidden_width == 4096
    assert extraction.provenance.model_revision == "6c7e5e573ac1667efc83407806fe9b0b93730e60"
    assert extraction.provenance.quantization == "fp8-cast"
    assert extraction.provenance.offload_mode == "cpu"
    assert extraction.provenance.vae_compile_mode is None
    assert extraction.provenance.to_dict()["vae_compile_mode"] is None
    assert fake_backbone.calls[0]["layer_index"] == 34


def test_extractor_rejects_prompt_width_mismatch():
    extractor = LTXExtractor(
        LTXExtractorConfig(device="cpu"),
        backbone=FakeBackbone(),
        latent_encoder=FakeEncoder(),
    )
    with pytest.raises(ValueError, match="prompt_embedding"):
        extractor.extract(
            torch.zeros((1, 3, 5, 480, 640), dtype=torch.uint8),
            torch.zeros((1, 4, 1024), dtype=torch.bfloat16),
        )


class CloseableFakeBackbone(FakeBackbone):
    def __init__(self):
        super().__init__()
        self.persistent = False
        self.explicit_prefix = False
        self.closed = False

    def close(self):
        self.closed = True


def test_persistent_backend_is_default_and_context_manager_closes_it():
    backbone = CloseableFakeBackbone()
    config = LTXExtractorConfig(device="cpu")
    assert config.persistent_transformer is True
    assert config.explicit_prefix_execution is True
    assert config.vae_compile_mode is None

    with LTXExtractor(config, backbone=backbone, latent_encoder=FakeEncoder()) as extractor:
        assert extractor.backbone.persistent is True
        assert extractor.backbone.explicit_prefix is True
        assert backbone.closed is False

    assert backbone.closed is True


def test_vae_compile_mode_is_explicit_and_validated():
    config = LTXExtractorConfig(device="cpu", vae_compile_mode="reduce-overhead")
    assert config.vae_compile_mode == "reduce-overhead"

    with pytest.raises(ValueError, match="vae_compile_mode"):
        LTXExtractorConfig(device="cpu", vae_compile_mode="fastest")
