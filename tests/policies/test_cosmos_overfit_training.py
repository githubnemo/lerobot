import pytest
import torch

from scripts.video_vam.train_cosmos_world2action_overfit import (
    DEFAULT_LR,
    EXPECTED_NATIVE_DECODER_PARAMETERS,
    _masked_physical_mse,
    build_optimizer,
    optimizer_settings,
    parse_args,
    select_autocast_context,
    validate_checkpoint_metadata,
    validate_train_args,
)


def test_diagnostic_cli_requires_acknowledgement_and_approved_values():
    args = parse_args(
        [
            "--manifest",
            "manifest.json",
            "--output-dir",
            "out",
            "--i-understand-diagnostic",
        ]
    )
    validate_train_args(args)
    assert args.steps == 300
    assert args.batch_size == 1
    assert args.lr == DEFAULT_LR
    without_ack = parse_args(["--manifest", "manifest.json", "--output-dir", "out"])
    with pytest.raises(ValueError, match="diagnostic"):
        validate_train_args(without_ack)


def test_optimizer_settings_request_fused_adamw_without_changing_hyperparameters():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer, settings = build_optimizer([parameter])
    assert settings["name"] == "AdamW"
    assert settings["lr"] == DEFAULT_LR
    assert settings["weight_decay"] == 0.0
    assert settings["betas"] == [0.9, 0.99]
    assert settings["eps"] == 1e-8
    assert settings["fused_requested"] is True
    assert optimizer.defaults["betas"] == (0.9, 0.99)
    assert optimizer.defaults["eps"] == 1e-8
    assert optimizer_settings()["fused_requested"] is True


def test_autocast_selection_is_noop_without_cuda():
    with select_autocast_context(torch.device("cpu")):
        value = torch.ones(())
    assert value.dtype == torch.float32


def test_final_physical_mse_excludes_padded_positions():
    prediction = torch.zeros(1, 30, 6)
    target = torch.zeros_like(prediction)
    target[:, 0] = 2.0
    target[:, 1:] = 100.0
    padding = torch.zeros(1, 30, dtype=torch.bool)
    padding[:, 1:] = True
    assert _masked_physical_mse(prediction, target, padding).item() == pytest.approx(4.0)


def test_checkpoint_metadata_rejects_non_full_or_rollout_status():
    keys = {
        "schema_version": 1,
        "artifact": "world2action_decoder_weights",
        "status": "diagnostic_only_non_rollout",
        "robot_ready": False,
        "weights_only": True,
        "optimizer_resume_supported": False,
        "step": 300,
        "manifest": "manifest.json",
        "manifest_sha256": "a" * 64,
        "normalizer": "normalizer.safetensors",
        "decoder_config": {},
        "parameter_count": {
            "decoder": EXPECTED_NATIVE_DECODER_PARAMETERS,
            "trainable_decoder": EXPECTED_NATIVE_DECODER_PARAMETERS,
        },
        "optimizer": {},
        "autocast": "cuda_bfloat16",
        "parameter_dtype": "float32",
        "frozen_context_dtype": "bfloat16",
        "action_padding_semantics": "action_is_pad=true excluded from loss and train statistics",
        "provenance": {},
    }
    validate_checkpoint_metadata(keys)
    keys["robot_ready"] = True
    with pytest.raises(ValueError, match="diagnostic-only"):
        validate_checkpoint_metadata(keys)
