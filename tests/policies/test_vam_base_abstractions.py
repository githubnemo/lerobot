"""Unit tests for VAM Base Abstractions and Protocol 1.0 Enforcement.

Tests:
1. BaseVAMExtractor, BaseExtractorConfig, and VAMExtractionOutput contract.
2. BaseLoRALinear, BaseVideoLoRAConfig, and LoRA injection/serialization/merging.
3. Protocol 1.0 split enforcement and data leakage prevention (split_guard.py).
4. Unified train_smolexpert pipeline on CPU (normalizer, action masking, scheduler, early stopping).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from torch import Tensor, nn

from lerobot.policies.vam.base import (
    PROTOCOL_1_0_TRAIN_EPISODES,
    PROTOCOL_1_0_VAL_EPISODES,
    BaseExtractorConfig,
    BaseLoRALinear,
    BaseVAMExtractor,
    BaseVideoLoRAConfig,
    ExtractorConfigError,
    ProtocolSplitGuard,
    ProtocolViolationError,
    VAEContractViolationError,
    VAMExtractionOutput,
    enforce_protocol_1_0_split,
    extract_episodes_from_manifest,
    extract_lora_state_dict,
    forbid_frame_level_random_split,
    get_lora_parameters,
    inject_base_lora,
    load_lora_weights,
    merge_all_lora_to_base,
    save_lora_weights,
    validate_manifests_for_protocol,
)
from lerobot.policies.vam.smol_expert import ACTION_DIM, ACTION_HORIZON
from scripts.video_vam.train_smolexpert import (
    EarlyStoppingTracker,
    UnifiedFeatureCacheDataset,
    build_scheduler,
    build_synthetic_tiny_decoder,
    compute_training_normalizer,
    evaluate_validation,
    main as train_smolexpert_main,
)

# ==============================================================================
# 1. Base Extractor Tests
# ==============================================================================


class ConcreteTestExtractor(BaseVAMExtractor):
    """Minimal concrete extractor implementing the BaseVAMExtractor contract for testing."""

    def __init__(self, config: BaseExtractorConfig) -> None:
        super().__init__(config)
        self.patch_size = (1, 2, 2)
        # Dummy linear block for hidden layers
        self.block = nn.Linear(16, 32)

    def encode_latents(self, rgb_frames: Tensor) -> Tensor:
        # Mock true VAE encoding contract: [B, 3, T, H, W] -> [B, 16, T, H//4, W//4]
        b, c, t, h, w = rgb_frames.shape
        return torch.zeros((b, 16, t, max(1, h // 4), max(1, w // 4)), dtype=self.dtype)

    def compute_grid_shape(self, latents: Tensor) -> tuple[int, int, int]:
        b, c, t, h, w = latents.shape
        p_t, p_h, p_w = self.patch_size
        return (t // p_t, h // p_h, w // p_w)

    def forward_transformer_blocks(
        self,
        latents: Tensor,
        text_conditioning: Tensor | str | None = None,
        timestep: float | Tensor = 0.0,
    ) -> dict[int, Tensor]:
        grid_shape = self.compute_grid_shape(latents)
        num_tokens = grid_shape[0] * grid_shape[1] * grid_shape[2]
        b = latents.shape[0]

        outputs = {}
        for lyr in self.config.hidden_layers:
            # Generate deterministic token embeddings [B, S, D]
            tokens = torch.randn((b, num_tokens, 16), dtype=self.dtype)
            outputs[lyr] = self.block(tokens)
        return outputs

    def get_provenance(self) -> dict[str, Any]:
        return {"extractor": "ConcreteTestExtractor", "version": "1.0"}


def test_base_extractor_config_validation():
    # Valid config
    cfg = BaseExtractorConfig(backbone_name="test_backbone", hidden_layers=(10, 20), pool_spatial=2)
    assert cfg.backbone_name == "test_backbone"
    assert cfg.hidden_layers == (10, 20)
    assert cfg.pool_spatial == 2

    # Invalid backbone name
    with pytest.raises(ExtractorConfigError, match="backbone_name"):
        BaseExtractorConfig(backbone_name="")

    # Invalid hidden layers
    with pytest.raises(ExtractorConfigError, match="hidden_layers"):
        BaseExtractorConfig(backbone_name="test", hidden_layers=())

    with pytest.raises(ExtractorConfigError, match="hidden_layers"):
        BaseExtractorConfig(backbone_name="test", hidden_layers=(-1,))

    # Invalid pool spatial
    with pytest.raises(ExtractorConfigError, match="pool_spatial"):
        BaseExtractorConfig(backbone_name="test", pool_spatial=0)

    # Invalid latent channels
    with pytest.raises(ExtractorConfigError, match="latent_channels"):
        BaseExtractorConfig(backbone_name="test", latent_channels=0)

    # Invalid dtype
    with pytest.raises(ExtractorConfigError, match="dtype"):
        BaseExtractorConfig(backbone_name="test", dtype="invalid_dtype")


def test_base_extractor_vae_contract_enforcement():
    cfg = BaseExtractorConfig(backbone_name="test", hidden_layers=(20,), latent_channels=16)
    extractor = ConcreteTestExtractor(cfg)

    # 1. Non-tensor input
    with pytest.raises(VAEContractViolationError, match="Latents must be a torch.Tensor"):
        extractor.validate_vae_contract([1, 2, 3])  # type: ignore

    # 2. Wrong dimensions (e.g. 4D instead of 5D)
    with pytest.raises(VAEContractViolationError, match="5 dimensions"):
        extractor.validate_vae_contract(torch.zeros(2, 16, 32, 32))

    # 3. Wrong channel count
    with pytest.raises(VAEContractViolationError, match="channels mismatch"):
        extractor.validate_vae_contract(torch.zeros(2, 12, 4, 32, 32))

    # 4. Non-floating point
    with pytest.raises(VAEContractViolationError, match="floating point"):
        extractor.validate_vae_contract(torch.zeros((2, 16, 4, 32, 32), dtype=torch.int32))

    # 5. Valid latent
    extractor.validate_vae_contract(torch.zeros(2, 16, 4, 32, 32, dtype=torch.float32))


def test_base_extractor_grid_and_pooling():
    cfg = BaseExtractorConfig(
        backbone_name="test",
        hidden_layers=(20,),
        pool_spatial=2,
        device="cpu",
    )
    extractor = ConcreteTestExtractor(cfg)

    # Grid shape: (T=2, H=4, W=4) -> 32 tokens
    grid_shape = (2, 4, 4)
    flat_tokens = torch.randn(2, 32, 16)

    # format_3d_grid
    grid_3d = extractor.format_3d_grid(flat_tokens, grid_shape)
    assert grid_3d.shape == (2, 2, 4, 4, 16)

    # Mismatched token count
    with pytest.raises(ValueError, match="does not match grid dimensions"):
        extractor.format_3d_grid(torch.randn(2, 20, 16), grid_shape)

    # pool_spatial_tokens
    pooled = extractor.pool_spatial_tokens(flat_tokens, grid_shape, factor=2)
    # After 2x2 pooling: H=2, W=2, T=2 -> 8 tokens
    assert pooled.shape == (2, 8, 16)

    # compute_3d_grid_coordinates
    coords = extractor.compute_3d_grid_coordinates(grid_shape=(2, 2, 2), batch_size=2)
    assert coords.shape == (2, 8, 3)
    # Verify coordinate bounds
    assert coords[:, :, 0].max().item() == 1  # T in [0, 1]
    assert coords[:, :, 1].max().item() == 1  # H in [0, 1]
    assert coords[:, :, 2].max().item() == 1  # W in [0, 1]


def test_base_extractor_end_to_end_extract():
    cfg = BaseExtractorConfig(
        backbone_name="test",
        hidden_layers=(10, 20),
        pool_spatial=2,
        concat_layers=True,
        device="cpu",
    )
    extractor = ConcreteTestExtractor(cfg)

    # Valid RGB video: [B=2, 3, T=2, H=16, W=16]
    rgb = torch.rand(2, 3, 2, 16, 16)
    out = extractor.extract(rgb_frames=rgb)

    assert isinstance(out, VAMExtractionOutput)
    assert out.batch_size == 2
    # Latents: [2, 16, 2, 4, 4] -> Patch (1,2,2) -> grid (2, 2, 2)
    # With pool_spatial=2: pooled grid (2, 1, 1) -> 2 tokens
    assert out.num_tokens == 2
    # Concat of two 32-dim layers -> 64 dim
    assert out.feature_dim == 64
    assert out.grid_coords.shape == (2, 2, 3)
    assert out.grid_shape == (2, 1, 1)
    assert 10 in out.features_by_layer and 20 in out.features_by_layer
    assert out.provenance["backbone_name"] == "test"


# ==============================================================================
# 2. Base LoRA Tests
# ==============================================================================


class DummyBlock(nn.Module):
    def __init__(self, in_dim: int = 16, out_dim: int = 16) -> None:
        super().__init__()
        self.to_q = nn.Linear(in_dim, out_dim)
        self.to_k = nn.Linear(in_dim, out_dim)
        self.to_v = nn.Linear(in_dim, out_dim)
        self.to_out = nn.ModuleList([nn.Linear(out_dim, out_dim)])
        self.unrelated = nn.Linear(in_dim, out_dim)


class DummyDiT(nn.Module):
    def __init__(self, num_blocks: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([DummyBlock() for _ in range(num_blocks)])


def test_base_lora_linear_forward_and_merge():
    base_linear = nn.Linear(16, 32)
    lora = BaseLoRALinear(base_linear, rank=4, alpha=8.0, dropout=0.0)

    assert lora.scaling == 2.0
    assert lora.in_features == 16
    assert lora.out_features == 32
    assert lora.lora_A.shape == (4, 16)
    assert lora.lora_B.shape == (32, 4)

    # Base parameters must be frozen
    for p in lora.base_layer.parameters():
        assert not p.requires_grad

    # LoRA parameters must be trainable
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad

    # At initialization, lora_B is all zeros, so output must exactly equal base layer output
    x = torch.randn(3, 16)
    out_lora = lora(x)
    out_base = base_linear(x)
    assert torch.allclose(out_lora, out_base, atol=1e-6)

    # Set non-zero LoRA B weights
    with torch.no_grad():
        lora.lora_B.fill_(0.5)

    out_modified = lora(x)
    assert not torch.allclose(out_modified, out_base, atol=1e-4)

    # Merge into base
    merged_linear = lora.merge_to_base()
    out_merged = merged_linear(x)
    assert torch.allclose(out_merged, out_modified, atol=1e-5)


def test_inject_base_lora_and_serialization(tmp_path: Path):
    model = DummyDiT(num_blocks=4)
    config = BaseVideoLoRAConfig(
        rank=8,
        alpha=16.0,
        target_modules=("to_q", "to_k", "to_v", "to_out.0"),
        target_blocks=(0, 1),
    )

    injected = inject_base_lora(model, config)
    # 4 targets in block 0 and 4 in block 1 = 8 injected layers
    assert len(injected) == 8
    assert "blocks.0.to_q" in injected
    assert "blocks.1.to_v" in injected
    # Block 2 should not be injected
    assert "blocks.2.to_q" not in injected

    # Verify parameters
    params = get_lora_parameters(model)
    assert len(params) == 16  # 8 layers * 2 (lora_A, lora_B)

    # State dict extraction
    sd = extract_lora_state_dict(model)
    assert len(sd) == 16

    # Save and reload
    save_path = tmp_path / "lora_test.safetensors"
    save_lora_weights(model, save_path, metadata={"test": "true"})
    assert save_path.is_file()
    assert save_path.with_suffix(".json").is_file()

    # Create fresh model with same config
    fresh_model = DummyDiT(num_blocks=4)
    inject_base_lora(fresh_model, config)

    # Modify weights before load to test change
    with torch.no_grad():
        for p in get_lora_parameters(model):
            p.add_(1.0)
    save_lora_weights(model, save_path)

    load_lora_weights(fresh_model, save_path)
    fresh_sd = extract_lora_state_dict(fresh_model)
    orig_sd = extract_lora_state_dict(model)

    for k in orig_sd:
        assert torch.allclose(orig_sd[k], fresh_sd[k])

    # Merge all
    merged_names = merge_all_lora_to_base(fresh_model)
    assert len(merged_names) == 8
    # After merging, no BaseLoRALinear instances remain
    for mod in fresh_model.modules():
        assert not isinstance(mod, BaseLoRALinear)


# ==============================================================================
# 3. Protocol 1.0 Split Guard Tests
# ==============================================================================


def test_split_guard_protocol_enforcement():
    # Valid canonical split
    enforce_protocol_1_0_split(PROTOCOL_1_0_TRAIN_EPISODES, PROTOCOL_1_0_VAL_EPISODES)

    # Valid subset split
    enforce_protocol_1_0_split([0, 1, 2], [32, 33], allow_subset=True)

    # 1. Overlapping train and val (data leakage)
    with pytest.raises(ProtocolViolationError, match="DATA LEAKAGE DETECTED"):
        enforce_protocol_1_0_split([0, 1, 2, 5], [5, 32, 33])

    # 2. Train set contains validation episodes (32-39)
    with pytest.raises(ProtocolViolationError, match="Training set contains held-out validation episodes"):
        enforce_protocol_1_0_split([0, 1, 32], [33, 34])

    # 3. Validation set contains train episodes (0-31)
    with pytest.raises(ProtocolViolationError, match="Validation set contains training episodes"):
        enforce_protocol_1_0_split([0, 1], [2, 32, 33])

    # 4. Empty set
    with pytest.raises(ProtocolViolationError, match="must not be empty"):
        enforce_protocol_1_0_split([], [32])

    with pytest.raises(ProtocolViolationError, match="must not be empty"):
        enforce_protocol_1_0_split([0], [])


def test_split_guard_manifest_validation():
    train_manifest = {
        "entries": [
            {"episode_index": 0, "frame_index": 0},
            {"episode_index": 1, "frame_index": 3},
        ]
    }
    val_manifest = {
        "entries": [
            {"episode_index": 32, "frame_index": 0},
            {"episode_index": 33, "frame_index": 20},
        ]
    }

    # Should pass
    train_eps, val_eps = validate_manifests_for_protocol(train_manifest, val_manifest)
    assert train_eps == (0, 1)
    assert val_eps == (32, 33)

    # Leaking manifest: episode 0 in val manifest
    leaking_val_manifest = {
        "entries": [
            {"episode_index": 0, "frame_index": 6},  # Leaked from train!
            {"episode_index": 32, "frame_index": 0},
        ]
    }
    with pytest.raises(ProtocolViolationError, match="DATA LEAKAGE DETECTED"):
        validate_manifests_for_protocol(train_manifest, leaking_val_manifest)


def test_forbid_frame_level_random_split():
    dummy_dataset = [1, 2, 3, 4]
    subset = torch.utils.data.Subset(dummy_dataset, [0, 2])
    with pytest.raises(ProtocolViolationError, match="Frame-level Subset detected"):
        forbid_frame_level_random_split(subset)


def test_protocol_split_guard_helper():
    guard = ProtocolSplitGuard.from_episodes([0, 1, 2], [32, 33])
    assert guard.train_episodes == (0, 1, 2)
    assert guard.val_episodes == (32, 33)

    train_m = {"entries": [{"episode_index": 5, "frame_index": 0}]}
    val_m = {"entries": [{"episode_index": 35, "frame_index": 0}]}
    manifest_guard = ProtocolSplitGuard.from_manifests(train_m, val_m)
    assert manifest_guard.train_episodes == (5,)
    assert manifest_guard.val_episodes == (35,)

    eps = extract_episodes_from_manifest(train_m)
    assert eps == {5}


# ==============================================================================
# 4. Unified SmolExpert Trainer Tests
# ==============================================================================


def _create_mock_cache_entry(
    root: Path,
    episode: int,
    frame: int,
    feature_dim: int = 16,
    num_tokens: int = 4,
    has_padding: bool = False,
) -> dict[str, Any]:
    file_name = f"ep{episode}_frame{frame}.safetensors"
    file_path = root / file_name

    context = torch.randn(num_tokens, feature_dim, dtype=torch.float32)
    state = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32)
    action = torch.ones(ACTION_HORIZON, ACTION_DIM, dtype=torch.float32) * (episode + 1.0)
    action_is_pad = torch.zeros(ACTION_HORIZON, dtype=torch.bool)
    if has_padding:
        # Pad the last 10 action steps
        action_is_pad[-10:] = True
        action[-10:] = 999.0  # Outlier value that must be masked out!

    save_file(
        {
            "context": context,
            "state": state,
            "action": action,
            "action_is_pad": action_is_pad,
        },
        str(file_path),
    )

    return {
        "sample_id": f"sample_ep{episode}_f{frame}",
        "episode_index": episode,
        "frame_index": frame,
        "safetensors": file_name,
    }


def test_unified_dataset_and_normalizer_padding_masking(tmp_path: Path):
    train_dir = tmp_path / "train_cache"
    train_dir.mkdir()

    # Create 2 train samples in episode 0 and 1 with deliberate extreme outlier values in padded steps
    entries = [
        _create_mock_cache_entry(train_dir, episode=0, frame=0, has_padding=True),
        _create_mock_cache_entry(train_dir, episode=1, frame=3, has_padding=False),
    ]
    manifest = {"entries": entries}
    manifest_path = train_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    dataset = UnifiedFeatureCacheDataset(manifest_path)
    assert len(dataset) == 2
    item0 = dataset[0]
    assert item0.context.shape == (4, 16)
    assert item0.action_is_pad[-1].item() is True
    assert item0.action[-1, 0].item() == 999.0

    # Compute normalizer
    norm_dir = tmp_path / "normalizer_out"
    normalizer = compute_training_normalizer(dataset, train_episodes=[0, 1], output_dir=norm_dir)

    # Padded actions (with value 999.0) must be completely excluded from statistics
    # Episode 0 valid actions are 1.0, Episode 1 valid actions are 2.0
    # Mean must be around 1.0 ~ 2.0, certainly NOT >= 100!
    assert normalizer.action_mean[0].item() < 10.0
    assert (norm_dir / "normalizer.safetensors").is_file()
    assert (norm_dir / "normalizer.json").is_file()


def test_evaluate_validation_action_masking(tmp_path: Path):
    val_dir = tmp_path / "val_cache"
    val_dir.mkdir()

    # Validation sample with padding and 999.0 outlier in padded area
    entries = [_create_mock_cache_entry(val_dir, episode=32, frame=0, has_padding=True)]
    manifest_path = val_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"entries": entries}))

    val_dataset = UnifiedFeatureCacheDataset(manifest_path)
    decoder = build_synthetic_tiny_decoder(input_channels=16, device="cpu")

    metrics = evaluate_validation(decoder, val_dataset, device=torch.device("cpu"), batch_size=1, num_steps=2)
    assert "val_rmse" in metrics
    assert "val_h1" in metrics
    assert "val_first5" in metrics
    assert "val_flow_loss" in metrics

    # Since padded actions are masked out, the 999.0 outlier does not blow up RMSE
    assert metrics["val_rmse"] < 100.0


def test_unified_scheduler_factors():
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    # Cosine scheduler with 10 warmup steps and 100 max steps
    cosine_sched = build_scheduler(optimizer, "cosine", warmup_steps=10, max_steps=100)

    # Warmup ramp
    assert cosine_sched.get_last_lr()[0] == pytest.approx(0.1, rel=1e-2)  # step 0: 1/10
    for _ in range(10):
        optimizer.step()
        cosine_sched.step()
    # At warmup end: lr factor = 1.0
    assert cosine_sched.get_last_lr()[0] == pytest.approx(1.0, rel=1e-2)

    # Constant scheduler
    const_sched = build_scheduler(optimizer, "constant", warmup_steps=5, max_steps=100)
    for _ in range(10):
        optimizer.step()
        const_sched.step()
    assert const_sched.get_last_lr()[0] == pytest.approx(1.0, rel=1e-2)


def test_early_stopping_tracker():
    tracker = EarlyStoppingTracker(patience=3, min_delta=0.01)

    # First eval: improvement
    assert tracker.update(15.0, step=100) is True
    assert tracker.best_metric == 15.0
    assert not tracker.should_stop

    # Improvement
    assert tracker.update(14.5, step=200) is True
    assert tracker.best_metric == 14.5
    assert tracker.bad_evaluations == 0

    # Stagnation (eval 1)
    assert tracker.update(14.5, step=300) is False
    assert tracker.bad_evaluations == 1
    assert not tracker.should_stop

    # Stagnation (eval 2)
    assert tracker.update(14.6, step=400) is False
    assert tracker.bad_evaluations == 2
    assert not tracker.should_stop

    # Stagnation (eval 3 -> patience 3 exhausted)
    assert tracker.update(14.7, step=500) is False
    assert tracker.bad_evaluations == 3
    assert tracker.should_stop is True

    # Test min_steps constraint
    tracker_min = EarlyStoppingTracker(patience=2, min_steps=1000)
    tracker_min.update(15.0, step=100)
    assert tracker_min.update(15.5, step=200) is False
    assert tracker_min.update(16.0, step=300) is False
    # Patience exhausted but step < min_steps
    assert tracker_min.bad_evaluations == 2
    assert not tracker_min.should_stop
    # Step >= min_steps triggers stopping
    assert tracker_min.update(16.5, step=1000) is False
    assert tracker_min.should_stop is True


def test_train_smolexpert_cli_cpu_dry_run(tmp_path: Path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    # Protocol 1.0 compliant: Train (ep 0, 1), Val (ep 32)
    train_entries = [
        _create_mock_cache_entry(train_dir, episode=0, frame=0),
        _create_mock_cache_entry(train_dir, episode=1, frame=0),
    ]
    (train_dir / "manifest.json").write_text(json.dumps({"entries": train_entries}))

    val_entries = [
        _create_mock_cache_entry(val_dir, episode=32, frame=0),
    ]
    (val_dir / "manifest.json").write_text(json.dumps({"entries": val_entries}))

    out_dir = tmp_path / "run_out"

    ret = train_smolexpert_main(
        [
            "--train-manifest",
            str(train_dir / "manifest.json"),
            "--val-manifest",
            str(val_dir / "manifest.json"),
            "--output-dir",
            str(out_dir),
            "--max-steps",
            "4",
            "--val-every",
            "2",
            "--batch-size",
            "2",
            "--dry-run",
            "--device",
            "cpu",
            "--no-wandb",
        ]
    )
    assert ret == 0
    assert (out_dir / "normalizer.safetensors").is_file()
    assert (out_dir / "best.safetensors").is_file()
    assert (out_dir / "best_metrics.json").is_file()
    assert (out_dir / "training_summary.json").is_file()


def test_train_smolexpert_cli_dual_eval(tmp_path: Path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    eval2_dir = tmp_path / "eval2"
    train_dir.mkdir()
    val_dir.mkdir()
    eval2_dir.mkdir()

    # Train (ep 0, 1, 40), Val/Eval1 (ep 32), Eval2 (ep 90)
    train_entries = [
        _create_mock_cache_entry(train_dir, episode=0, frame=0),
        _create_mock_cache_entry(train_dir, episode=1, frame=0),
        _create_mock_cache_entry(train_dir, episode=40, frame=0),
    ]
    (train_dir / "manifest.json").write_text(json.dumps({"entries": train_entries}))

    val_entries = [
        _create_mock_cache_entry(val_dir, episode=32, frame=0),
    ]
    (val_dir / "manifest.json").write_text(json.dumps({"entries": val_entries}))

    eval2_entries = [
        _create_mock_cache_entry(eval2_dir, episode=90, frame=0),
    ]
    (eval2_dir / "manifest.json").write_text(json.dumps({"entries": eval2_entries}))

    out_dir = tmp_path / "run_dual_out"

    ret = train_smolexpert_main(
        [
            "--train-manifest",
            str(train_dir / "manifest.json"),
            "--val-manifest",
            str(val_dir / "manifest.json"),
            "--eval2-manifest",
            str(eval2_dir / "manifest.json"),
            "--output-dir",
            str(out_dir),
            "--max-steps",
            "4",
            "--val-every",
            "2",
            "--batch-size",
            "2",
            "--dry-run",
            "--device",
            "cpu",
            "--no-wandb",
        ]
    )
    assert ret == 0
    assert (out_dir / "normalizer.safetensors").is_file()
    assert (out_dir / "best.safetensors").is_file()
    assert (out_dir / "best_metrics.json").is_file()
    assert (out_dir / "final_dual_eval_metrics.json").is_file()
    assert (out_dir / "training_summary.json").is_file()

    with open(out_dir / "final_dual_eval_metrics.json") as f:
        metrics = json.load(f)
        assert "eval1_historical" in metrics
        assert "eval2_new_benchmark" in metrics
        assert metrics["eval1_historical"]["val_rmse"] > 0
        assert metrics["eval2_new_benchmark"]["val_rmse"] > 0


def test_train_smolexpert_cli_blocks_data_leakage(tmp_path: Path):
    train_dir = tmp_path / "train_leak"
    val_dir = tmp_path / "val_leak"
    train_dir.mkdir()
    val_dir.mkdir()

    # Data leakage: episode 0 in BOTH train and val!
    train_entries = [_create_mock_cache_entry(train_dir, episode=0, frame=0)]
    (train_dir / "manifest.json").write_text(json.dumps({"entries": train_entries}))

    val_entries = [_create_mock_cache_entry(val_dir, episode=0, frame=3)]
    (val_dir / "manifest.json").write_text(json.dumps({"entries": val_entries}))

    out_dir = tmp_path / "run_leak_out"

    with pytest.raises(ProtocolViolationError, match="DATA LEAKAGE DETECTED"):
        train_smolexpert_main(
            [
                "--train-manifest",
                str(train_dir / "manifest.json"),
                "--val-manifest",
                str(val_dir / "manifest.json"),
                "--output-dir",
                str(out_dir),
                "--dry-run",
                "--device",
                "cpu",
                "--no-wandb",
            ]
        )
