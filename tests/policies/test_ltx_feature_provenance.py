import pytest

from lerobot.policies.vam.ltx_feature_provenance import BackboneFeatureProvenance, assert_same_backbone


def provenance(**overrides):
    value = {
        "backbone": "LTX-2.5-22B-distilled",
        "checkpoint_identity": "checkpoint.safetensors",
        "checkpoint_sha256": "a" * 64,
        "hidden_width": 4096,
        "layer": 34,
        "noise_parameterization": "normalized_rectified_flow_sigma",
        "noise_level": 1.0,
        "token_geometry": (16, 15, 20),
        "token_count": 4800,
        "dtype": "bfloat16",
    }
    value.update(overrides)
    return BackboneFeatureProvenance(**value)


def test_provenance_round_trip_is_json_safe():
    original = provenance()
    restored = BackboneFeatureProvenance.from_dict(original.to_dict())
    assert restored == original
    assert restored.to_dict()["token_geometry"] == [16, 15, 20]


def test_backbone_mismatch_is_rejected():
    with pytest.raises(ValueError, match="incompatible"):
        assert_same_backbone(provenance(), provenance(backbone="Cosmos-Predict2-2B"))
