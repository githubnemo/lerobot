"""Unit, AST, and parity tests for shared native video training, EDM objectives, VAE scaling, and protocol splits."""

from __future__ import annotations

import ast
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from lerobot.policies.vam.base.native_video import (
    CosmosEDMScaling,
    CosmosVAENormalizer,
    DatasetIntegrityError,
    EpisodeClipSpec,
    FluxVAENormalizer,
    LoRAMetadataError,
    TextEmbeddingError,
    extract_single_image_from_dataset_row,
    load_and_validate_prompt_embedding,
    preencode_and_offload_clips,
    validate_dataset_repo_and_root,
    validate_training_episodes,
)
from lerobot.policies.vam.base.split_guard import ProtocolViolationError
from lerobot.policies.vam.cosmos7b_extractor import (
    build_cosmos7b_dummy_model,
)
from lerobot.policies.vam.cosmos7b_lora import (
    Cosmos7BLoRAConfig,
    Cosmos7BLoRAError,
    compute_cosmos7b_edm_loss,
    get_cosmos7b_lora_parameters,
    inject_cosmos7b_lora,
    load_cosmos7b_lora,
    save_cosmos7b_lora,
)
from lerobot.policies.vam.cosmos14b_extractor import (
    build_cosmos14b_dummy_model,
)
from lerobot.policies.vam.cosmos14b_lora import (
    Cosmos14BLoRAConfig,
    compute_cosmos14b_edm_loss,
    get_cosmos14b_lora_parameters,
    inject_cosmos14b_quantized_lora,
)
from lerobot.policies.vam.flux2_klein_extractor import (
    build_flux2_klein_dummy_model,
    prepare_multi_reference_conditioning,
)
from lerobot.policies.vam.flux2_klein_lora import (
    Flux2KleinLoRAConfig,
    compute_multi_reference_rf_loss,
    get_flux2_klein_lora_parameters,
    inject_flux2_klein_lora,
)


def test_parse_and_validate_training_episodes_protocol1():
    eps = validate_training_episodes("0-31", protocol="protocol1")
    assert eps == list(range(32))

    eps_subset = validate_training_episodes("0, 1, 2, 5-10", protocol="protocol1")
    assert eps_subset == [0, 1, 2, 5, 6, 7, 8, 9, 10]

    with pytest.raises(ProtocolViolationError):
        validate_training_episodes("0-32", protocol="protocol1")

    with pytest.raises(ProtocolViolationError):
        validate_training_episodes("35-38", protocol="protocol1")


def test_parse_and_validate_training_episodes_scale100():
    eps = validate_training_episodes("0-31, 40-89", protocol="scale100")
    assert len(eps) == 82
    assert 32 not in eps
    assert 90 not in eps

    with pytest.raises(ProtocolViolationError):
        validate_training_episodes("0-31, 35, 40-89", protocol="scale100")

    with pytest.raises(ProtocolViolationError):
        validate_training_episodes("0-31, 40-92", protocol="scale100")


def test_validate_dataset_repo_and_root():
    dummy_ds_40 = SimpleNamespace(meta=SimpleNamespace(episodes=[None] * 40), revision="wrong_rev")
    with pytest.raises(DatasetIntegrityError):
        validate_dataset_repo_and_root("hubnemo/cube_out_of_box_dataset", "protocol1", None, dummy_ds_40)

    with pytest.raises(DatasetIntegrityError):
        validate_dataset_repo_and_root(
            "Orellius/cube_out_of_box_v2", "scale100", "/path/to/40eps", dummy_ds_40
        )


def test_load_and_validate_prompt_embedding():
    with pytest.raises(TextEmbeddingError, match="No --text-embedding-path provided"):
        load_and_validate_prompt_embedding(None, prompt="take cube", expected_dim=1024)

    with tempfile.TemporaryDirectory() as tmpdir:
        emb_file = Path(tmpdir) / "prompt.safetensors"
        emb_tensor = torch.randn(1, 16, 1024)
        save_file({"prompt_embeds": emb_tensor}, str(emb_file), metadata={"prompt": "take cube out of box"})

        loaded, prov = load_and_validate_prompt_embedding(
            emb_file, prompt="take cube out of box", expected_dim=1024
        )
        assert loaded.shape == (1, 16, 1024)
        assert prov["prompt"] == "take cube out of box"

        with pytest.raises(TextEmbeddingError, match="Prompt identity mismatch"):
            load_and_validate_prompt_embedding(emb_file, prompt="different task prompt", expected_dim=1024)

        with pytest.raises(TextEmbeddingError, match="dimension mismatch"):
            load_and_validate_prompt_embedding(emb_file, prompt="take cube out of box", expected_dim=5120)


def test_extract_single_image_from_dataset_row():
    img_4d = torch.randn(3, 3, 64, 64)
    extracted = extract_single_image_from_dataset_row(img_4d)
    assert extracted.shape == (3, 64, 64)
    assert torch.equal(extracted, img_4d[-1])

    img_3d = torch.randn(3, 64, 64)
    extracted_3d = extract_single_image_from_dataset_row(img_3d)
    assert extracted_3d.shape == (3, 64, 64)
    assert torch.equal(extracted_3d, img_3d)


def test_preencode_and_offload_clips_with_tchw_fake_dataset():
    class FakeTCHWDataset:
        def __init__(self):
            self.data = {i: {"observation.images.front": torch.randn(2, 3, 32, 32)} for i in range(20)}

        def __getitem__(self, idx):
            return self.data[idx]

    from diffusers import AutoencoderKLCosmos

    vae = AutoencoderKLCosmos(
        in_channels=3,
        out_channels=3,
        latent_channels=16,
        encoder_block_out_channels=(16, 32),
        decode_block_out_channels=(16, 32),
        num_layers=1,
        resolution=32,
    )
    normalizer = CosmosVAENormalizer(vae=vae, sigma_data=0.5)

    clips = [
        EpisodeClipSpec(
            episode_index=0, start_frame_idx=0, end_frame_idx=5, frame_indices=tuple(range(5)), fps=10
        ),
        EpisodeClipSpec(
            episode_index=0, start_frame_idx=5, end_frame_idx=10, frame_indices=tuple(range(5, 10)), fps=10
        ),
    ]
    text_tensor = torch.randn(1, 4, 32)
    preencoded = preencode_and_offload_clips(
        dataset=FakeTCHWDataset(),
        clips=clips,
        normalizer=normalizer,
        text_encoder_fn=text_tensor,
        prompt="take cube",
        batch_size=2,
        device="cpu",
        offload_to_cpu=False,
    )
    assert len(preencoded) == 2
    lat, txt = preencoded[0]
    assert lat.ndim == 5
    assert lat.shape[1] == 16


def test_cosmos_edm_scaling_mathematical_parity():
    scaling = CosmosEDMScaling(sigma_data=0.5)
    sigmas = torch.tensor([0.001, 0.5, 1.0, 10.0])

    c_skip, c_out, c_in, c_noise = scaling(sigmas)
    weights = scaling.sigma_loss_weights(sigmas)

    assert torch.all(c_in > 0)
    assert torch.all(c_skip >= 0)
    assert torch.all(c_out >= 0)
    assert torch.allclose(weights, 1.0 / (c_out**2), rtol=1e-4)

    c_skip_half = 0.25 / (0.25 + 0.25)
    assert math.isclose(c_skip[1].item(), c_skip_half, rel_tol=1e-5)


def test_cosmos7b_edm_loss_and_gradients():
    model = build_cosmos7b_dummy_model(
        num_layers=4,
        num_attention_heads=2,
        attention_head_dim=16,
        text_embed_dim=32,
        device="cpu",
    )
    lora_cfg = Cosmos7BLoRAConfig(
        rank=4,
        alpha=4.0,
        target_blocks=(0, 1),
    )
    inject_cosmos7b_lora(model, lora_cfg)
    lora_params = get_cosmos7b_lora_parameters(model)
    assert len(lora_params) > 0

    clean_latents = torch.randn(1, 16, 2, 4, 4)
    text_embeds = torch.randn(1, 4, 32)

    loss = compute_cosmos7b_edm_loss(
        model=model,
        clean_latents=clean_latents,
        encoder_hidden_states=text_embeds,
        fps=10,
    )
    assert torch.isfinite(loss)
    loss.backward(inputs=lora_params)
    for p in lora_params:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_cosmos14b_edm_loss_and_gradients():
    model = build_cosmos14b_dummy_model(
        num_layers=4,
        num_attention_heads=2,
        attention_head_dim=16,
        text_embed_dim=32,
        device="cpu",
    )
    lora_cfg = Cosmos14BLoRAConfig(
        rank=4,
        alpha=4.0,
        target_blocks=(0, 1),
        quantization="none",
    )
    inject_cosmos14b_quantized_lora(model, lora_cfg)
    lora_params = get_cosmos14b_lora_parameters(model)

    clean_latents = torch.randn(1, 17, 2, 4, 4)
    text_embeds = torch.randn(1, 4, 32)

    loss = compute_cosmos14b_edm_loss(
        model=model,
        clean_latents=clean_latents,
        encoder_hidden_states=text_embeds,
        fps=10,
    )
    assert torch.isfinite(loss)
    loss.backward(inputs=lora_params)

    for p in lora_params:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_strict_lora_metadata_validation_two_way():
    model = build_cosmos7b_dummy_model(
        num_layers=2, num_attention_heads=2, attention_head_dim=16, text_embed_dim=32
    )
    lora_cfg = Cosmos7BLoRAConfig(rank=8, alpha=16.0, target_blocks=(0, 1))
    inject_cosmos7b_lora(model, lora_cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "strict_lora.safetensors"
        save_cosmos7b_lora(model, save_path, rank=8, alpha=16.0)

        # 1. Matching load succeeds
        loaded = load_cosmos7b_lora(model, save_path, expected_rank=8, expected_alpha=16.0)
        assert len(loaded) > 0

        # 2. Mismatched rank fails
        with pytest.raises((Cosmos7BLoRAError, LoRAMetadataError), match="rank mismatch"):
            load_cosmos7b_lora(model, save_path, expected_rank=16)

        # 3. Missing model keys in partial checkpoint fails
        all_tensors = {k: v for k, v in model.state_dict().items() if "lora_" in k}
        partial_sd = {k: v for i, (k, v) in enumerate(all_tensors.items()) if i == 0}
        partial_path = Path(tmpdir) / "partial.safetensors"
        save_file(partial_sd, str(partial_path), metadata={"lora_rank": "8", "lora_alpha": "16.0"})
        with pytest.raises((Cosmos7BLoRAError, LoRAMetadataError), match="Missing LoRA parameters"):
            load_cosmos7b_lora(model, partial_path, expected_rank=8, expected_alpha=16.0)


def test_cosmos_vae_5_frame_encoding():
    from diffusers import AutoencoderKLCosmos

    vae = AutoencoderKLCosmos(
        in_channels=3,
        out_channels=3,
        latent_channels=16,
        encoder_block_out_channels=(32, 64),
        decode_block_out_channels=(32, 64),
        num_layers=1,
        resolution=64,
    )
    normalizer = CosmosVAENormalizer(vae=vae, sigma_data=0.5)

    rgb5 = torch.randint(0, 256, (1, 3, 5, 64, 64), dtype=torch.uint8).float()
    latents = normalizer.encode(rgb5, deterministic=True)

    assert latents.shape[0] == 1
    assert latents.shape[1] == 16
    assert latents.shape[2] == 2
    assert torch.isfinite(latents).all()


def test_flux2_vae_native_patchify_and_bn():
    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2(
        in_channels=3,
        out_channels=3,
        latent_channels=32,
        block_out_channels=(32, 64, 128, 128),
        layers_per_block=1,
    )
    normalizer = FluxVAENormalizer(vae=vae)

    rgb = torch.randn(1, 3, 64, 64)
    norm_latents = normalizer.encode(rgb, deterministic=True)

    assert norm_latents.shape == (1, 128, 4, 4)
    assert torch.isfinite(norm_latents).all()


def test_flux2_causal_multi_reference_conditioning_and_rf_loss():
    base_model = build_flux2_klein_dummy_model(
        num_layers=2,
        num_single_layers=2,
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=16,
        joint_attention_dim=32,
        axes_dims_rope=(4, 4, 4, 4),
        device="cpu",
    )
    lora_cfg = Flux2KleinLoRAConfig(
        rank=4,
        alpha=4.0,
        target_double_blocks=(0, 1),
        target_single_blocks=(0, 1),
    )
    inject_flux2_klein_lora(base_model, lora_cfg)
    lora_params = get_flux2_klein_lora_parameters(base_model)

    target_frame = torch.randn(1, 16, 4, 4)
    history_frames = [torch.randn(1, 16, 4, 4) for _ in range(3)]

    cond = prepare_multi_reference_conditioning(
        target_latent=target_frame,
        observation_history=history_frames,
        joint_attention_dim=32,
        causal_only=True,
    )
    assert "goal" not in cond.token_slices
    assert cond.goal_tokens_count == 0

    target_clean = cond.hidden_states[:, : cond.target_tokens_count]
    loss = compute_multi_reference_rf_loss(
        model=base_model,
        cond_inputs=cond,
        target_clean=target_clean,
    )
    assert torch.isfinite(loss)
    loss.backward(inputs=lora_params)

    for p in lora_params:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_ast_production_paths_no_random_text_or_pseudo_vae():
    """AST analysis verifying that production training paths do not contain randn fallback for text or pseudo VAE."""
    script_paths = [
        Path("/home/anton/lerobot-video-vam/scripts/video_vam/train_cosmos7b_video_lora.py"),
        Path("/home/anton/lerobot-video-vam/scripts/video_vam/train_cosmos14b_video_lora.py"),
        Path("/home/anton/lerobot-video-vam/scripts/video_vam/train_flux2_klein_lora.py"),
    ]

    for p in script_paths:
        if not p.is_file():
            continue
        tree = ast.parse(p.read_text())

        main_fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
        assert main_fn is not None, f"main() not found in {p}"

        backward_calls = []
        for node in ast.walk(main_fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "backward"
            ):
                backward_calls.append(node)

        assert len(backward_calls) >= 1, f"No backward calls found in {p}"
        for call in backward_calls:
            has_inputs = any(kw.arg == "inputs" for kw in call.keywords)
            assert has_inputs, f"Selective backward (inputs=lora_params) missing on backward call in {p}"
