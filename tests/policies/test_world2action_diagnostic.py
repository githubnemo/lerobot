import copy
import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

import lerobot.policies.vam.world2action as module
from lerobot.policies.vam.world2action import World2ActionConfig, World2ActionDecoder


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, **kwargs):
        return torch.zeros_like(kwargs["xt_B_HA_A"]) + self.weight


def make_artifacts(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"entries": []}')
    normalizer = tmp_path / "normalizer.safetensors"
    save_file({"state_min": torch.zeros(1, 6)}, str(normalizer))
    checkpoint = tmp_path / "checkpoint.safetensors"
    source = TinyDenoiser().state_dict()
    save_file(source, str(checkpoint))
    metadata = {
        "schema_version": 1,
        "artifact": "world2action_decoder_weights",
        "status": "diagnostic_only_non_rollout",
        "robot_ready": False,
        "weights_only": True,
        "optimizer_resume_supported": False,
        "step": 1000,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "normalizer": str(normalizer.resolve()),
        "decoder_config": copy.deepcopy(module.EXPECTED_DIAGNOSTIC_DECODER_CONFIG),
        "parameter_count": {"decoder": 1, "trainable_decoder": 1},
        "optimizer": {},
        "autocast": "cuda_bfloat16",
        "parameter_dtype": "float32",
        "frozen_context_dtype": "bfloat16",
        "action_padding_semantics": "action_is_pad=true excluded from loss and train statistics",
        "provenance": {"test": True},
    }
    metadata_path = checkpoint.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata))
    return checkpoint, metadata_path, manifest, normalizer, metadata


def make_decoder():
    return World2ActionDecoder(World2ActionConfig(device="cpu"), denoiser=TinyDenoiser())


def test_diagnostic_reopen_validates_provenance_and_assigns_weights(tmp_path):
    checkpoint, metadata_path, manifest, normalizer, _ = make_artifacts(tmp_path)
    decoder = make_decoder()

    metadata = module.load_diagnostic_checkpoint(
        decoder,
        checkpoint,
        metadata_path=metadata_path,
        manifest_path=manifest,
        normalizer_path=normalizer,
        expected_step=1000,
        expected_parameter_count=1,
    )

    assert metadata["status"] == "diagnostic_only_non_rollout"
    assert decoder.denoiser.weight.dtype == torch.bfloat16
    assert decoder.denoiser.weight.item() == pytest.approx(2.0)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_key",
        "extra_key",
        "bad_shape",
        "bad_dtype",
        "bad_manifest",
        "bad_status",
        "bad_count",
        "metadata_extra",
    ],
)
def test_diagnostic_reopen_rejects_tampering(tmp_path, mutation):
    checkpoint, metadata_path, manifest, normalizer, metadata = make_artifacts(tmp_path)
    if mutation == "missing_key":
        save_file({}, str(checkpoint))
    elif mutation == "extra_key":
        save_file({"weight": torch.tensor(2.0), "unexpected": torch.tensor(1.0)}, str(checkpoint))
    elif mutation == "bad_shape":
        save_file({"weight": torch.zeros(2)}, str(checkpoint))
    elif mutation == "bad_dtype":
        save_file({"weight": torch.tensor(2.0, dtype=torch.float64)}, str(checkpoint))
    elif mutation == "bad_manifest":
        metadata["manifest_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata))
    elif mutation == "bad_status":
        metadata["status"] = "rollout_ready"
        metadata_path.write_text(json.dumps(metadata))
    elif mutation == "metadata_extra":
        metadata["unexpected"] = True
        metadata_path.write_text(json.dumps(metadata))
    else:
        metadata["parameter_count"]["decoder"] = 499171958
        metadata_path.write_text(json.dumps(metadata))

    with pytest.raises((module.World2ActionValidationError, RuntimeError)):
        module.load_diagnostic_checkpoint(
            make_decoder(),
            checkpoint,
            metadata_path=metadata_path,
            manifest_path=manifest,
            normalizer_path=normalizer,
            expected_parameter_count=1,
        )
