import logging

import numpy as np
import pytest
import torch
from torch import nn

import lerobot.policies.vam.cosmos_predict2_extractor as extractor_module
import lerobot.policies.vam.download_cosmos_checkpoints as checkpoints
from lerobot.policies.vam._vendor.cosmos_predict2._compat import log as cosmos_log
from lerobot.policies.vam.cosmos_predict2_extractor import (
    CosmosPredict2BackendError,
    CosmosPredict2CheckpointError,
    CosmosPredict2Extractor,
    CosmosPredict2ExtractorConfig,
    _backbone_autocast_context,
    arch_invariant_rand,
    load_checkpoint_strict,
)
from lerobot.policies.vam.download_cosmos_checkpoints import build_plan


class FakeTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.seen_shape = None

    @torch.no_grad()
    def encode(self, images):
        self.seen_shape = tuple(images.shape)
        return torch.full((images.shape[0], 16, 2, 60, 80), 0.25, dtype=images.dtype, device=images.device)


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.seen = None
        self.grad_enabled = None

    def forward(self, **kwargs):
        self.grad_enabled = torch.is_grad_enabled()
        self.seen = kwargs
        x = kwargs["x_B_C_T_H_W"]
        hidden = torch.zeros(x.shape[0], 16, 30, 40, 2048, device=x.device, dtype=x.dtype)
        hidden.requires_grad_()
        return None, [hidden for _ in range(21)]


def test_compat_logger_success_delegates_to_info(caplog):
    with caplog.at_level(logging.INFO, logger=cosmos_log.name):
        cosmos_log.success("loaded %s", "Cosmos")
        cosmos_log.warning("warning remains standard")

    assert [record.levelno for record in caplog.records] == [logging.INFO, logging.WARNING]
    assert caplog.records[0].getMessage() == "loaded Cosmos"
    assert caplog.records[1].getMessage() == "warning remains standard"


def test_arch_invariant_rand_matches_upstream_numpy():
    shape = (2, 3)
    expected = np.random.RandomState(17).standard_normal(shape).astype(np.float32)
    actual = arch_invariant_rand(shape, 17)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, torch.from_numpy(expected))


def test_backbone_autocast_selection_without_cuda(monkeypatch):
    calls = []

    class FakeAutocast:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(extractor_module.torch, "autocast", FakeAutocast)

    with _backbone_autocast_context(torch.device("cuda"), torch.bfloat16):
        pass
    with _backbone_autocast_context(torch.device("cuda"), torch.float16):
        pass
    with _backbone_autocast_context(torch.device("cpu"), torch.bfloat16):
        pass
    with _backbone_autocast_context(torch.device("cpu"), torch.float32):
        pass

    assert calls == [
        {"device_type": "cuda", "dtype": torch.bfloat16},
        {"device_type": "cuda", "dtype": torch.float16},
    ]


def make_extractor():
    config = CosmosPredict2ExtractorConfig(
        checkpoint_path="/not-loaded/backbone.pt",
        tokenizer_path="/not-loaded/tokenizer.pth",
        device="cpu",
        dtype=torch.float32,
        seed=17,
    )
    return CosmosPredict2Extractor(config, backbone=FakeBackbone(), tokenizer=FakeTokenizer())


def test_extract_contract_layout_and_metadata():
    extractor = make_extractor()
    images = torch.full((1, 3, 5, 480, 640), 128, dtype=torch.uint8)
    prompt = torch.zeros(1, 512, 1024)

    output = extractor.extract(images, prompt)

    assert output.hidden_grid.shape == (1, 16, 30, 40, 2048)
    assert output.tokens.shape == (1, 19200, 2048)
    assert output.sigma.shape == (1,)
    assert output.sigma.tolist() == [10.0]
    assert output.grid_shape == (16, 30, 40)
    assert output.layer == 20
    assert output.provenance.upstream_commit == "e3355dbc93132b576c02f920a59b4fc18a4f5906"
    assert output.provenance.backend == "minimal_a2a"
    assert output.provenance.checkpoint_ignored_metadata_keys == ()
    assert output.provenance.checkpoint_ignored_metadata_count == 0
    assert not output.hidden_grid.requires_grad
    assert torch.equal(output.tokens[:, 0], output.hidden_grid[:, 0, 0, 0])

    tokenizer = extractor.tokenizer
    backbone = extractor.backbone
    assert tokenizer.seen_shape == (1, 3, 61, 480, 640)
    assert backbone.seen["x_B_C_T_H_W"].shape == (1, 16, 16, 60, 80)
    assert backbone.seen["x_B_C_T_H_W"][:, :, :2].eq(0.25).all()
    assert backbone.seen["condition_video_input_mask_B_C_T_H_W"][:, :, :2].eq(1).all()
    assert backbone.seen["condition_video_input_mask_B_C_T_H_W"][:, :, 2:].eq(0).all()
    timesteps = backbone.seen["timesteps_B_T"]
    assert timesteps.shape == (1, 16)
    assert torch.allclose(timesteps[:, :2], torch.full((1, 2), 0.0001 / 1.0001))
    assert torch.allclose(timesteps[:, 2:], torch.full((1, 14), 10.0 / 11.0))
    assert backbone.grad_enabled is False


def test_parameters_are_frozen():
    extractor = make_extractor()
    assert all(not parameter.requires_grad for parameter in extractor.backbone.parameters())
    assert all(not parameter.requires_grad for parameter in extractor.tokenizer.parameters())


def test_invalid_shapes_and_ranges():
    extractor = make_extractor()
    prompt = torch.zeros(1, 512, 1024)
    with pytest.raises(ValueError, match=r"\[B, 3, T, 480, 640\]"):
        extractor.extract(torch.zeros(1, 3, 4, 480, 640), prompt)
    with pytest.raises(ValueError, match="prompt_embedding"):
        extractor.extract(torch.zeros(1, 3, 5, 480, 640), torch.zeros(1, 512, 1023))
    with pytest.raises(ValueError, match="finite"):
        extractor.extract(torch.zeros(1, 3, 5, 480, 640), torch.full((1, 512, 1024), float("nan")))
    with pytest.raises(ValueError, match=r"\[0, 1\] or \[-1, 1\]"):
        extractor.extract(torch.full((1, 3, 5, 480, 640), 2.0), prompt)


def test_strict_checkpoint_key_errors(tmp_path):
    module = nn.Linear(3, 2)
    state = module.state_dict()
    state.pop("bias")
    checkpoint = tmp_path / "partial.pt"
    torch.save(state, checkpoint)

    with pytest.raises(CosmosPredict2CheckpointError, match="missing"):
        load_checkpoint_strict(module, checkpoint)


def _checkpoint_with_te_metadata(tmp_path, *, missing=False, extra_key=None, metadata_dtype=torch.uint8):
    module = nn.Linear(3, 2)
    state = module.state_dict()
    if missing:
        state.pop("bias")
    for block in range(28):
        state[f"blocks.{block}.cross_attn.attn_op._extra_state"] = torch.tensor(
            [128, 4, 78, 46], dtype=metadata_dtype
        )
    if extra_key is not None:
        state[extra_key] = torch.ones(1)
    checkpoint = tmp_path / "v2w_pretrained_cosmos.pt"
    torch.save(state, checkpoint)
    return module, checkpoint


def test_exact_official_te_metadata_is_accepted_and_reported(tmp_path):
    module, checkpoint = _checkpoint_with_te_metadata(tmp_path)
    ignored = load_checkpoint_strict(module, checkpoint, allow_official_te_extra_state=True)
    expected = tuple(sorted(f"blocks.{block}.cross_attn.attn_op._extra_state" for block in range(28)))
    assert ignored == expected


def test_te_metadata_allowlist_does_not_hide_near_match_or_weight_key(tmp_path):
    module, checkpoint = _checkpoint_with_te_metadata(
        tmp_path,
        extra_key="blocks.0.cross_attn.attn_op._extra_state.weight",
    )
    with pytest.raises(CosmosPredict2CheckpointError, match="_extra_state.weight"):
        load_checkpoint_strict(module, checkpoint, allow_official_te_extra_state=True)


def test_te_metadata_allowlist_still_rejects_missing_weights(tmp_path):
    module, checkpoint = _checkpoint_with_te_metadata(tmp_path, missing=True)
    with pytest.raises(CosmosPredict2CheckpointError, match="missing"):
        load_checkpoint_strict(module, checkpoint, allow_official_te_extra_state=True)


def test_te_metadata_payload_must_be_uint8_tensor(tmp_path):
    module, checkpoint = _checkpoint_with_te_metadata(tmp_path, metadata_dtype=torch.int64)
    with pytest.raises(CosmosPredict2CheckpointError, match="1-D uint8 tensor"):
        load_checkpoint_strict(module, checkpoint, allow_official_te_extra_state=True)


def test_te_metadata_is_rejected_without_explicit_official_opt_in(tmp_path):
    module, checkpoint = _checkpoint_with_te_metadata(tmp_path)
    with pytest.raises(CosmosPredict2CheckpointError, match="unexpected"):
        load_checkpoint_strict(module, checkpoint)


def test_te_metadata_opt_in_rejects_nonofficial_checkpoint_name(tmp_path):
    module, checkpoint = _checkpoint_with_te_metadata(tmp_path)
    renamed = checkpoint.with_name("other_checkpoint.pt")
    checkpoint.rename(renamed)
    with pytest.raises(CosmosPredict2CheckpointError, match="pinned generic checkpoint"):
        load_checkpoint_strict(module, renamed, allow_official_te_extra_state=True)


def test_minimal_a2a_requires_cuda_for_official_execution():
    config = CosmosPredict2ExtractorConfig(
        checkpoint_path="/not-loaded/backbone.pt",
        tokenizer_path="/not-loaded/tokenizer.pth",
        device="cpu",
    )
    with pytest.raises(CosmosPredict2BackendError, match="minimal_a2a.*CUDA"):
        CosmosPredict2Extractor(config)


def test_official_minimal_a2a_reports_missing_checkpoint():
    config = CosmosPredict2ExtractorConfig(
        checkpoint_path="/not-loaded/backbone.pt",
        tokenizer_path="/not-loaded/tokenizer.pth",
        device="cuda",
    )
    with pytest.raises(FileNotFoundError, match="backbone checkpoint"):
        CosmosPredict2Extractor(config)


def test_official_minimal_a2a_reports_missing_transformer_engine(tmp_path, monkeypatch):
    backbone = tmp_path / "backbone.pt"
    tokenizer = tmp_path / "tokenizer.pth"
    backbone.touch()
    tokenizer.touch()

    def missing_transformer_engine(name):
        if name == "transformer_engine":
            raise ModuleNotFoundError("test missing Transformer Engine")
        return __import__(name)

    monkeypatch.setattr(extractor_module, "import_module", missing_transformer_engine)
    config = CosmosPredict2ExtractorConfig(
        checkpoint_path=backbone,
        tokenizer_path=tokenizer,
        device="cuda",
    )
    with pytest.raises(CosmosPredict2BackendError, match="TE.*minimal_a2a|Transformer Engine.*normalization"):
        CosmosPredict2Extractor(config)


def test_checkpoint_plan_is_pinned_and_does_not_include_t5():
    plan = build_plan(include_bridge_lora=True)
    assert plan.repo_id == "jonpai/mimic-video"
    assert plan.revision == "f28339034831e3c2374be075e622e1ff38ebe0f8"
    assert plan.allow_patterns == (
        "video_backbone/v2w_pretrained_cosmos.pt",
        "video_backbone/tokenizer/*",
        "video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt",
    )
    assert not any("text" in pattern.lower() for pattern in plan.allow_patterns)


def test_checkpoint_plan_selects_t5_and_accounts_for_known_bytes():
    plan = build_plan(include_t5=True)
    assert plan.allow_patterns == (
        "video_backbone/v2w_pretrained_cosmos.pt",
        "video_backbone/tokenizer/*",
        "text_encoder/t5-11b/*",
    )
    assert plan.known_bytes == 3913017214 + 507609880 + 45229452544
    assert plan.required_bytes_with_buffer > plan.known_bytes


def test_disk_preflight_refuses_insufficient_space(tmp_path):
    plan = build_plan()
    with pytest.raises(RuntimeError, match="Insufficient free space"):
        checkpoints.check_free_space(
            tmp_path / "nested" / "output",
            plan,
            disk_usage=lambda _: type("Usage", (), {"free": 0})(),
        )


def test_verification_reports_size_or_hash_mismatch(tmp_path):
    artifact = checkpoints.Artifact("artifact.bin", "artifact.bin", 3, "0" * 64)
    plan = checkpoints.CosmosCheckpointDownloadPlan("repo", "rev", (artifact.pattern,), (artifact,))
    (tmp_path / artifact.relative_path).write_bytes(b"bad")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        checkpoints.verify_download(tmp_path, plan)


def test_t5_deletion_requires_execute_exact_hash_and_embedding_confirmation(tmp_path):
    with pytest.raises(ValueError, match="exactly match"):
        checkpoints.delete_t5(tmp_path, execute=True, expected_sha256="wrong", prompt_embedding_verified=True)
    with pytest.raises(ValueError, match="prompt-embedding-verified"):
        checkpoints.delete_t5(
            tmp_path,
            execute=True,
            expected_sha256=checkpoints.ARTIFACTS["t5"].sha256,
            prompt_embedding_verified=False,
        )


def test_per_call_noise_seed_is_deterministic_and_recorded():
    extractor = make_extractor()
    images = torch.full((1, 3, 5, 480, 640), 128, dtype=torch.uint8)
    prompt = torch.zeros(1, 512, 1024)
    first = extractor.extract(images, prompt, noise_seed=123)
    first_input = extractor.backbone.seen["x_B_C_T_H_W"].clone()
    extractor.extract(images, prompt, noise_seed=123)
    second_input = extractor.backbone.seen["x_B_C_T_H_W"].clone()
    extractor.extract(images, prompt, noise_seed=124)
    third_input = extractor.backbone.seen["x_B_C_T_H_W"].clone()
    assert first.provenance.noise_seed == 123
    assert torch.equal(first_input, second_input)
    assert not torch.equal(first_input, third_input)
    with pytest.raises(ValueError, match="noise_seed"):
        extractor.extract(images, prompt, noise_seed=2**32 - 1)
