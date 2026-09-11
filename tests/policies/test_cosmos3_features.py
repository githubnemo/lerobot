"""Targeted unit tests for Cosmos 3 Edge feature extraction, VAE encoding, 3D mRoPE, and gen_seq routing."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import (
    get_3d_mrope_ids_text_tokens,
    get_3d_mrope_ids_vae_tokens,
)
from safetensors.torch import save_file
from torch import nn

from lerobot.policies.vam.cosmos3_features import (
    Cosmos3ExtractorConfig,
    Cosmos3FeatureExtractor,
    compute_cosmos3_cache_key,
    encode_cosmos3_video,
    patchify_and_pack_cosmos3_vision_latents,
    prepare_cosmos3_mrope_and_seq,
    sha256_file,
    tap_cosmos3_layers,
)


class _MockDiagonalGaussianDistribution:
    def __init__(self, mode_tensor: torch.Tensor, sample_tensor: torch.Tensor) -> None:
        self._mode = mode_tensor
        self._sample = sample_tensor

    def mode(self) -> torch.Tensor:
        return self._mode

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        return self._sample


class _MockAutoencoderOutput:
    def __init__(self, mode_tensor: torch.Tensor, sample_tensor: torch.Tensor) -> None:
        self.latent_dist = _MockDiagonalGaussianDistribution(mode_tensor, sample_tensor)


class _MockWanVAE(nn.Module):
    def __init__(self, z_dim: int = 48) -> None:
        super().__init__()
        self.dtype = torch.float32
        self.config = MagicMock()
        self.config.latents_mean = [0.1 * i for i in range(z_dim)]
        self.config.latents_std = [1.0 + 0.05 * i for i in range(z_dim)]
        self.config.scale_factor_spatial = 16
        self.config.scale_factor_temporal = 4
        self.param = nn.Parameter(torch.zeros(1))

    def encode(self, x: torch.Tensor) -> _MockAutoencoderOutput:
        b = x.shape[0]
        t = x.shape[2] if x.ndim == 5 else 1
        t_lat = (t - 1) // 4 + 1
        h_lat = x.shape[-2] // 16
        w_lat = x.shape[-1] // 16
        z_dim = len(self.config.latents_mean) if self.config.latents_mean is not None else 48
        mode_val = torch.ones((b, z_dim, t_lat, h_lat, w_lat), dtype=x.dtype, device=x.device) * 2.0
        sample_val = torch.ones((b, z_dim, t_lat, h_lat, w_lat), dtype=x.dtype, device=x.device) * 99.0
        return _MockAutoencoderOutput(mode_val, sample_val)


def build_tiny_real_cosmos3_transformer(
    num_layers: int = 4,
    hidden_size: int = 64,
    head_dim: int = 16,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    dtype: str = "float32",
) -> Cosmos3OmniTransformer:
    """Instantiate a real native diffusers Cosmos3OmniTransformer with compact dimensions for CPU tests."""
    return Cosmos3OmniTransformer(
        head_dim=head_dim,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        latent_channel=48,
        latent_patch_size=2,
        patch_latent_dim=192,
        vocab_size=1000,
        dtype=dtype,
    )


def test_encode_cosmos3_video_normalization_and_posterior_mode() -> None:
    vae = _MockWanVAE(z_dim=48)

    # 1. Test uint8-scaled input [0, 255]
    rgb_uint8 = torch.full((1, 3, 5, 480, 640), 255, dtype=torch.uint8)
    latents = encode_cosmos3_video(vae, rgb_uint8, sample_mode="argmax")

    # Shape check: T=5 -> t_lat = (5-1)//4 + 1 = 2; H=480//16=30; W=640//16=40
    assert latents.shape == (1, 48, 2, 30, 40)

    # Value check: raw_mu is 2.0.
    # latents = (2.0 - mean) / std
    for c in range(48):
        mean_c = vae.config.latents_mean[c]
        std_c = vae.config.latents_std[c]
        expected_val = (2.0 - mean_c) / std_c
        assert torch.allclose(latents[0, c], torch.tensor(expected_val, dtype=torch.float32), atol=1e-5)


def test_encode_cosmos3_video_missing_stats_fails_explicitly() -> None:
    vae = _MockWanVAE(z_dim=48)
    vae.config.latents_mean = None
    vae.config.latents_std = None

    rgb = torch.rand(1, 3, 5, 480, 640)
    with pytest.raises(ValueError, match="missing latents_mean or latents_std"):
        encode_cosmos3_video(vae, rgb)


def test_patchify_and_pack_cosmos3_vision_latents() -> None:
    latents = torch.randn(1, 48, 2, 30, 40)
    packed, grid_shape = patchify_and_pack_cosmos3_vision_latents(latents, patch_size=2)

    assert grid_shape == (2, 15, 20)
    assert packed.shape == (1, 600, 192)


def test_visual_route_invariant_gen_seq_and_modality_margin() -> None:
    transformer = build_tiny_real_cosmos3_transformer(num_layers=2, hidden_size=64)

    prompt_embeds = torch.randn(5, 64)
    vision_tokens = torch.randn(600, 64)
    grid_shape = (2, 15, 20)

    und_seq, gen_seq, rotary_emb = prepare_cosmos3_mrope_and_seq(
        transformer=transformer,
        text_token_embeds=prompt_embeds,
        packed_vision_tokens=vision_tokens,
        grid_shape=grid_shape,
        fps=10.0,
        base_fps=24.0,
        dtype=torch.float32,
    )

    # INVARIANT: und_seq contains strictly text tokens (5 tokens)
    assert und_seq.shape == (5, 64)
    # INVARIANT: gen_seq contains clean observation vision tokens (600 tokens)
    assert gen_seq.shape == (600, 64)
    # Rotary embedding sliced into und (5) and gen (600)
    cos_und, sin_und, cos_gen, sin_gen = rotary_emb
    assert cos_und.shape == (5, 16)
    assert cos_gen.shape == (600, 16)


def test_tap_cosmos3_layers_layer20_definition_and_safe_early_exit() -> None:
    """Layer N means representation after N blocks (block index N - 1). Layer 2 -> block index 1."""
    transformer = build_tiny_real_cosmos3_transformer(num_layers=4, hidden_size=64)

    prompt_embeds = torch.randn(5, 64)
    vision_tokens = torch.randn(8, 64)
    grid_shape = (2, 2, 2)

    und_seq, gen_seq, rotary_emb = prepare_cosmos3_mrope_and_seq(
        transformer=transformer,
        text_token_embeds=prompt_embeds,
        packed_vision_tokens=vision_tokens,
        grid_shape=grid_shape,
        fps=10.0,
        dtype=torch.float32,
    )

    tapped = tap_cosmos3_layers(
        transformer,
        und_seq=und_seq,
        gen_seq=gen_seq,
        rotary_emb=rotary_emb,
        target_layers=(2,),
        early_exit=True,
    )

    assert 2 in tapped
    assert tapped[2].shape == (1, 8, 64)

    for layer in transformer.layers:
        assert len(layer._forward_hooks) == 0


def test_native_full_forward_hook_parity_on_real_diffusers_model() -> None:
    """Compare intermediate representations from full native transformer.forward() hook vs tap_cosmos3_layers."""
    transformer = build_tiny_real_cosmos3_transformer(num_layers=3, hidden_size=64)
    transformer.eval()

    torch.manual_seed(42)
    prompt_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)
    und_len = len(prompt_ids)
    latent = torch.randn(1, 48, 2, 4, 4)
    packed_vision_raw, grid_shape = patchify_and_pack_cosmos3_vision_latents(latent, patch_size=2)
    t_lat, patch_h, patch_w = grid_shape
    num_vision_tokens = t_lat * patch_h * patch_w  # 2 * 2 * 2 = 8
    seq_len = und_len + num_vision_tokens

    # Build mRoPE position IDs matching native pipeline exactly
    text_mrope_ids, next_mrope_offset = get_3d_mrope_ids_text_tokens(
        num_tokens=und_len,
        temporal_offset=0,
        use_float_positions=True,
    )
    vision_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
        grid_t=t_lat,
        grid_h=patch_h,
        grid_w=patch_w,
        temporal_offset=next_mrope_offset + transformer.config.unified_3d_mrope_temporal_modality_margin,
        reset_spatial_indices=transformer.config.unified_3d_mrope_reset_spatial_ids,
        fps=10.0,
        base_fps=24.0,
        temporal_compression_factor=4,
    )
    pos_ids = torch.cat([text_mrope_ids, vision_mrope_ids], dim=1)

    # 1. Run full native transformer.forward with a hook on layer index 1 (Layer 2)
    native_hook_captured = {}

    def hook_fn(mod, inp, out):
        # out is (und_out, gen_out)
        native_hook_captured["gen"] = out[1].detach().clone()

    hook_handle = transformer.layers[1].register_forward_hook(hook_fn)
    try:
        transformer(
            input_ids=prompt_ids,
            text_indexes=torch.arange(und_len, dtype=torch.long),
            position_ids=pos_ids,
            und_len=und_len,
            sequence_length=seq_len,
            vision_tokens=[latent],
            vision_token_shapes=[grid_shape],
            vision_sequence_indexes=torch.arange(und_len, seq_len, dtype=torch.long),
            vision_mse_loss_indexes=torch.arange(und_len, seq_len, dtype=torch.long),
            vision_timesteps=torch.zeros(num_vision_tokens),
            vision_noisy_frame_indexes=[torch.tensor([], dtype=torch.long)],
            return_dict=False,
        )
    finally:
        hook_handle.remove()

    native_gen_layer2 = native_hook_captured["gen"].unsqueeze(0)

    # 2. Run extraction tapping Layer 2
    prompt_embeds = transformer.embed_tokens(prompt_ids)
    proj_vision = transformer.proj_in(packed_vision_raw)
    und_seq, gen_seq, rotary_emb = prepare_cosmos3_mrope_and_seq(
        transformer=transformer,
        text_token_embeds=prompt_embeds,
        packed_vision_tokens=proj_vision,
        grid_shape=grid_shape,
        fps=10.0,
        base_fps=24.0,
        dtype=torch.float32,
    )
    tapped = tap_cosmos3_layers(
        transformer,
        und_seq=und_seq,
        gen_seq=gen_seq,
        rotary_emb=rotary_emb,
        target_layers=(2,),
        early_exit=True,
    )
    extracted_gen_layer2 = tapped[2]

    # Must be bit-for-bit identical
    assert torch.allclose(native_gen_layer2, extracted_gen_layer2, atol=1e-6)


def test_compute_cosmos3_cache_key_determinism_and_staleness() -> None:
    key1 = compute_cosmos3_cache_key(
        dataset_repo="hubnemo/cube_out_of_box_dataset",
        dataset_revision="243370c3c08bcbd860133c4a0d658ea7c1d2e77e",
        episodes=range(32),
        stride=3,
        clip_frames=17,
        hidden_layer=20,
        fps=10.0,
        prompt="take cube out of box",
        checkpoint_sha256="abc123def456",
    )
    key2 = compute_cosmos3_cache_key(
        dataset_repo="hubnemo/cube_out_of_box_dataset",
        dataset_revision="243370c3c08bcbd860133c4a0d658ea7c1d2e77e",
        episodes=range(32),
        stride=3,
        clip_frames=17,
        hidden_layer=20,
        fps=10.0,
        prompt="take cube out of box",
        checkpoint_sha256="abc123def456",
    )
    assert key1 == key2

    # Different prompt -> different cache key
    key_prompt = compute_cosmos3_cache_key(
        dataset_repo="hubnemo/cube_out_of_box_dataset",
        dataset_revision="243370c3c08bcbd860133c4a0d658ea7c1d2e77e",
        episodes=range(32),
        stride=3,
        clip_frames=17,
        hidden_layer=20,
        fps=10.0,
        prompt="pick up red block",
        checkpoint_sha256="abc123def456",
    )
    assert key1 != key_prompt


def test_sha256_file() -> None:
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(b"cosmos3 test string")
        tmp.flush()
        digest = sha256_file(tmp.name)
        assert len(digest) == 64
        assert digest == "efa5d0506c464c03065fa3e39f48960d1607fa8f3b138520e41ac5cfbf16f3dd"


def test_cosmos3_extractor_end_to_end_and_unified_dataset_compat(tmp_path: Path) -> None:
    """Verify that Cosmos3FeatureExtractor extracts features compatible with UnifiedFeatureCacheDataset."""
    transformer = build_tiny_real_cosmos3_transformer(num_layers=4, hidden_size=64)
    vae = _MockWanVAE(z_dim=48)

    config = Cosmos3ExtractorConfig(
        backbone_name="cosmos3-edge",
        hidden_layers=(2,),
        device="cpu",
        dtype="float32",
        latent_patch_size=2,
        hidden_dim=64,
        num_layers=4,
        fps=10.0,
    )
    tokenizer = MagicMock(return_value=type("Tokens", (), {"input_ids": [1, 2, 3]})())
    extractor = Cosmos3FeatureExtractor(config, transformer=transformer, vae=vae, tokenizer=tokenizer)

    rgb = torch.rand(1, 3, 5, 480, 640)
    out = extractor.extract(rgb_frames=rgb)

    assert out.features.shape == (1, 600, 64)
    assert 2 in out.features_by_layer
    assert out.grid_shape == (2, 15, 20)

    # Save to disk in unified cache format
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    sample_file = cache_dir / "sample_0.safetensors"
    context_t = out.features.squeeze(0).contiguous()
    state_t = torch.randn(6)
    action_t = torch.randn(30, 6)
    pad_t = torch.zeros(30, dtype=torch.bool)

    save_file(
        {
            "context": context_t,
            "state": state_t,
            "target_action": action_t,
            "action_is_pad": pad_t,
        },
        str(sample_file),
    )

    manifest_file = cache_dir / "manifest.json"
    manifest_payload = {
        "schema_version": 1,
        "dataset": {
            "repo_id": "hubnemo/cube_out_of_box_dataset",
            "revision": "243370c3c08bcbd860133c4a0d658ea7c1d2e77e",
        },
        "subset": {"episodes": [0], "stride": 3},
        "entries": [
            {
                "sample_id": "episode-0000-frame-000004",
                "episode_index": 0,
                "frame_index": 4,
                "safetensors": sample_file.name,
                "bytes": sample_file.stat().st_size,
                "safetensors_sha256": sha256_file(sample_file),
            }
        ],
    }
    manifest_file.write_text(json.dumps(manifest_payload))

    from scripts.video_vam.train_smolexpert import UnifiedFeatureCacheDataset

    ds = UnifiedFeatureCacheDataset(manifest_file)
    assert len(ds) == 1
    item = ds[0]
    assert item.context.shape == (600, 64)
    assert item.state.shape == (6,)
    assert item.action.shape == (30, 6)
    assert item.action_is_pad.shape == (30,)
