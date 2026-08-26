import json

import pytest
import torch

from lerobot.policies.vam.cosmos_video_prediction import (
    OfficialRectifiedFlowAB2Scheduler,
    build_prediction_report,
    compute_baseline_metrics,
    compute_frame_metrics,
    cosmos_2b_backbone_spec,
    temporal_alignment,
    validate_report_schema,
)


def test_temporal_alignment_uses_causal_four_to_one_mapping(tmp_path):
    spec = cosmos_2b_backbone_spec(tmp_path / "cosmos.pt", tmp_path / "tokenizer.pth")
    alignment = temporal_alignment(
        current_frame_index=14,
        input_frames=spec.input_frames,
        state_t=spec.state_t,
        latent_conditional_frames=spec.latent_conditional_frames,
        temporal_compression_factor=spec.temporal_compression_factor,
    )

    assert spec.pixel_frames == 61
    assert spec.vae_input_mode == "observed_prefix"
    assert spec.latent_chunk_duration == 21
    assert alignment.conditioning_indices == (10, 11, 12, 13, 14)
    assert alignment.output_indices[:5] == alignment.conditioning_indices
    assert alignment.predicted_future_indices == tuple(range(15, 71))


def test_backbone_spec_accepts_legacy_padded_vae_mode(tmp_path):
    spec = cosmos_2b_backbone_spec(
        tmp_path / "cosmos.pt", tmp_path / "tokenizer.pth", vae_input_mode="legacy_padded_vae"
    )
    assert spec.vae_input_mode == "legacy_padded_vae"


def test_temporal_alignment_rejects_inconsistent_prefix_geometry():
    with pytest.raises(ValueError, match="input_frames"):
        temporal_alignment(
            current_frame_index=10,
            input_frames=4,
            state_t=16,
            latent_conditional_frames=2,
            temporal_compression_factor=4,
        )


def test_metrics_are_per_frame_and_deterministic():
    predicted = torch.full((2, 3, 16, 16), 0.5)
    target = torch.zeros_like(predicted)
    result = compute_frame_metrics(predicted, target, frame_indices=(21, 22))

    assert [row["frame_index"] for row in result["per_frame"]] == [21, 22]
    assert result["per_frame"][0]["psnr_db"] == pytest.approx(6.0205999, rel=1e-6)
    assert result["mean_psnr_db"] == pytest.approx(6.0205999, rel=1e-6)
    assert result["mean_ssim"] < 1.0
    identical = compute_frame_metrics(target, target)
    assert identical["per_frame"][0]["psnr_db"] is None


def test_baselines_use_the_same_metric_implementation():
    conditioning = torch.zeros((5, 3, 16, 16))
    target = torch.zeros((4, 3, 16, 16))
    baselines = compute_baseline_metrics(conditioning, target, frame_indices=(5, 6, 7, 8))

    assert baselines["ground_truth_copy"]["mean_psnr_db"] is None
    assert baselines["ground_truth_copy"]["mean_ssim"] == pytest.approx(1.0)
    assert baselines["static"]["mean_psnr_db"] is None


def test_official_scheduler_has_35_step_schedule_and_exact_endpoints():
    scheduler = OfficialRectifiedFlowAB2Scheduler(0.002, 80.0, 7.0, 35)
    scheduler.set_timesteps(torch.device("cpu"))

    assert scheduler.sigmas is not None
    assert scheduler.sigmas.shape == (36,)
    assert scheduler.sigmas[0].item() == pytest.approx(80.0)
    assert scheduler.sigmas[-1].item() == pytest.approx(0.002)


def test_report_schema_is_strict_and_json_safe(tmp_path):
    spec = cosmos_2b_backbone_spec(tmp_path / "cosmos.pt", tmp_path / "tokenizer.pth")
    alignment = temporal_alignment(
        current_frame_index=14,
        input_frames=5,
        state_t=16,
        latent_conditional_frames=2,
        temporal_compression_factor=4,
    )
    report = build_prediction_report(
        spec=spec,
        seed=17,
        checkpoint={"path": str(spec.checkpoint_path), "size_bytes": 3, "sha256": "0" * 64},
        episode_index=2,
        current_frame_index=14,
        alignment=alignment,
        sampler={"name": "RectifiedFlowAB2Scheduler", "steps": 35},
        metrics={"per_frame": [], "mean_psnr_db": None, "mean_ssim": None, "lpips": {"status": "skipped"}},
        runtime={"clip_seconds": 1.0},
        artifacts={"report": str(tmp_path / "report.json")},
        alignment_probe={"decoded_frame_count": 61},
    )
    validate_report_schema(report)
    json.dumps(report, allow_nan=False)

    invalid = dict(report)
    invalid["unexpected"] = True
    with pytest.raises(ValueError, match="exactly"):
        validate_report_schema(invalid)
