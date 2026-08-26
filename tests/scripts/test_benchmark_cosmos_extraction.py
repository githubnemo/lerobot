import json

import pytest
import torch
from safetensors.torch import save_file

import scripts.video_vam.benchmark_cosmos_extraction as benchmark_module
from scripts.video_vam.benchmark_cosmos_extraction import (
    ALL_ARMS,
    COMPILE_ARMS,
    DECODER_STEP_SWEEP,
    EXPERT_COMPILE_VARIANTS,
    _assemble_full_condition,
    arms_for_request,
    build_output_payload,
    output_drift,
    parse_args,
    render_markdown,
    summarize_samples,
    summarize_tensor_comparisons,
    tensor_comparison,
    validate_args,
    write_outputs,
)


def _timing(median_ms: float, p90_ms: float) -> dict[str, float | int]:
    return {"median_ms": median_ms, "p90_ms": p90_ms, "samples": 20}


def _measured_arm() -> dict:
    return {
        "arm": "baseline",
        "effective_arm": "baseline",
        "status": "ok",
        "message": None,
        "optimization": {"implemented": True},
        "extractor_load_seconds": 1.0,
        "decoder_load_seconds": 2.0,
        "dtypes": {"extractor": "bfloat16", "decoder": "bfloat16"},
        "stage_timings_ms": {
            "preprocessing": _timing(1.0, 1.2),
            "vae_encode": _timing(10.0, 12.0),
            "dit_forward_layer20": _timing(100.0, 120.0),
            "feature_reduction_pool2": _timing(2.0, 2.4),
        },
        "extraction_end_to_end_ms": _timing(113.0, 136.0),
        "decoder_included": True,
        "decoder_inference_ms": _timing(50.0, 60.0),
        "full_end_to_end_ms": _timing(163.0, 196.0),
        "latency": {
            "extraction_hz": 1000.0 / 113.0,
            "decoder_hz": 20.0,
            "chunk_hz_30_steps": 30_000.0 / 163.0,
            "chunk_hz_30_steps_p90": 30_000.0 / 196.0,
        },
    }


def _payload(arm: dict | None = None) -> dict:
    return build_output_payload(
        arm_results=[arm or _measured_arm()],
        hardware={
            "gpu_name": "test GPU",
            "torch_version": "test",
            "cuda_version": "test",
        },
        configuration={
            "iterations": 20,
            "warmups": 3,
            "cudnn_benchmark": False,
            "channels_last": False,
        },
        input_metadata={
            "rgb_shape": [1, 3, 5, 480, 640],
            "pool2_shape": [1, 4800, 2048],
            "extractor_dtype": "bfloat16",
        },
        optimization_inventory_data={"fp8": {}, "int8": {}},
        generated_at_utc="2026-08-24T10:00:00+00:00",
    )


def test_parse_args_exposes_arms_and_safe_sampling_defaults():
    args = parse_args(["--arm", "compile", "--output-dir", "reports"])
    validate_args(args)
    assert args.arm == "compile"
    assert args.attention_backend == "minimal_a2a"
    assert args.iterations == 20
    assert args.warmups == 3
    assert "train-converge" in str(args.decoder_checkpoint)
    assert "train-converge" in str(args.normalizer)
    assert args.expert_checkpoint == "lerobot/smolvla_base"
    assert args.vlm_config == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def test_load_decoder_uses_trainer_constructor_and_custom_artifact(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.safetensors"
    checkpoint.write_bytes(b"placeholder")
    normalizer = tmp_path / "normalizer.safetensors"
    save_file(
        {
            "state_mean": torch.zeros(6),
            "state_std": torch.ones(6),
            "action_mean": torch.zeros(6),
            "action_std": torch.ones(6),
        },
        str(normalizer),
    )
    calls = {}

    class FakeDecoder:
        @classmethod
        def from_training_checkpoint(cls, path, **kwargs):
            calls["path"] = path
            calls.update(kwargs)
            return cls()

        def eval(self):
            return self

    monkeypatch.setattr(benchmark_module, "SmolExpertActionDecoder", FakeDecoder)
    args = parse_args(
        [
            "--decoder-checkpoint",
            str(checkpoint),
            "--normalizer",
            str(normalizer),
            "--expert-checkpoint",
            "base-checkpoint",
            "--vlm-config",
            "expert-config",
        ]
    )

    decoder, elapsed = benchmark_module._load_decoder(args, torch.device("cpu"))

    assert isinstance(decoder, FakeDecoder)
    assert calls["path"] == checkpoint
    assert tuple(calls["normalizer"].state_mean.shape) == (6,)
    assert calls["expert_checkpoint"] == "base-checkpoint"
    assert calls["vlm_config_name"] == "expert-config"
    assert calls["device"] == torch.device("cpu")
    assert calls["num_steps"] == 10
    assert elapsed >= 0.0


def test_all_contains_only_stable_implemented_arms():
    assert (
        arms_for_request("all")
        == ALL_ARMS
        == (
            "baseline",
            "compile",
            "compile_regional",
            "fp8",
            "vae_prefix",
        )
    )
    assert COMPILE_ARMS == ("compile", "compile_max", "compile_regional", "compile_regional_max")
    assert "vae_prefix" in arms_for_request("all")
    assert "cuda_graph" not in arms_for_request("all")
    assert "decoder_sweep" not in arms_for_request("all")
    assert "context_cache" not in arms_for_request("all")
    assert DECODER_STEP_SWEEP == (10, 5, 3, 2, 1)


def test_assemble_full_condition_expands_only_valid_prefix_latents():
    class Config:
        latent_channels = 16
        latent_conditional_frames = 2
        state_t = 16
        latent_height = 60
        latent_width = 80

    class Extractor:
        config = Config()

    prefix = torch.ones(1, 16, 2, 60, 80)
    full = _assemble_full_condition(Extractor(), prefix)
    assert full.shape == (1, 16, 16, 60, 80)
    assert torch.equal(full[:, :, :2], prefix)
    assert torch.count_nonzero(full[:, :, 2:]) == 0
    with pytest.raises(ValueError, match="tokenizer result"):
        _assemble_full_condition(Extractor(), torch.zeros(1, 16, 3, 60, 80))


def test_tensor_comparison_reports_shape_cosine_and_strict_summary():
    reference = torch.tensor([1.0, 0.0, -2.0])
    exact = tensor_comparison(reference, reference)
    assert exact["shape"] == [3]
    assert exact["max_abs"] == 0.0
    assert exact["cosine"] == pytest.approx(1.0)
    summary = summarize_tensor_comparisons([exact, exact])
    assert summary["samples"] == 2
    assert summary["min_cosine"] == pytest.approx(1.0)


def test_tensor_comparison_rejects_shape_drift():
    with pytest.raises(ValueError, match="equal shapes"):
        tensor_comparison(torch.zeros(2), torch.zeros(3))


def test_output_drift_reports_exact_zero_and_nonzero_metrics():
    reference = torch.tensor([[1.0, -2.0]])
    assert output_drift(reference, reference) == {"max_abs": 0.0, "mean_abs": 0.0, "rmse": 0.0}
    metrics = output_drift(reference, torch.tensor([[2.0, -1.0]]))
    assert metrics == {"max_abs": 1.0, "mean_abs": 1.0, "rmse": 1.0}


def test_validate_args_requires_stable_sample_count():
    args = parse_args(["--iterations", "19"])
    with pytest.raises(ValueError, match="at least 20"):
        validate_args(args)


def test_summarize_samples_returns_milliseconds_and_p90():
    summary = summarize_samples([0.001, 0.002, 0.003, 0.004])
    assert summary["median_ms"] == pytest.approx(2.5)
    assert summary["p90_ms"] == pytest.approx(3.7)
    assert summary["samples"] == 4


def test_write_outputs_preserves_json_schema_and_summary(tmp_path):
    payload = _payload()
    json_path, markdown_path = write_outputs(tmp_path, payload)
    written = json.loads(json_path.read_text())
    assert written["schema_version"] == 1
    assert written["arms"][0]["stage_timings_ms"]["vae_encode"]["samples"] == 20
    summary = markdown_path.read_text()
    assert "| baseline | ok |" in summary
    assert "30-step Hz" in summary
    assert render_markdown(payload) == summary


def test_unavailable_arm_is_renderable_without_fake_timings():
    arm = {
        "arm": "int8",
        "effective_arm": None,
        "status": "rejected",
        "message": "no validated CUDA int8 path",
        "optimization": {"implemented": False},
        "extractor_load_seconds": None,
        "decoder_load_seconds": None,
        "dtypes": {},
    }
    payload = _payload(arm)
    assert "rejected" in render_markdown(payload)
    assert "no validated CUDA int8 path" in render_markdown(payload)


def test_expert_compile_arm_has_fixed_safe_construction_defaults():
    args = parse_args(["--arm", "expert_compile"])
    validate_args(args)

    assert arms_for_request("expert_compile") == ("expert_compile",)
    assert args.decoder_steps == 10
    assert EXPERT_COMPILE_VARIANTS == (
        "eager_kv_cache",
        "compile_reduce_overhead",
        "compile_max_autotune",
        "cuda_graph",
    )


def test_expert_compile_arm_rejects_decoder_free_or_non_ten_step_requests():
    no_decoder = parse_args(["--arm", "expert_compile", "--no-decoder"])
    with pytest.raises(ValueError, match="requires the SmolVLA decoder"):
        validate_args(no_decoder)

    wrong_steps = parse_args(["--arm", "expert_compile", "--decoder-steps", "5"])
    with pytest.raises(ValueError, match="requires --decoder-steps 10"):
        validate_args(wrong_steps)


def test_expert_compile_result_renders_nested_variant_timings():
    arm = _measured_arm()
    arm.update(
        {
            "arm": "expert_compile",
            "stage_timings_ms": {},
            "extraction_end_to_end_ms": None,
            "decoder_inference_ms": _timing(8.0, 9.0),
            "full_end_to_end_ms": _timing(8.0, 9.0),
            "latency": {
                "extraction_hz": None,
                "decoder_hz": 125.0,
                "chunk_hz_30_steps": 3750.0,
                "chunk_hz_30_steps_p90": 3333.3,
            },
            "expert_compile": {
                "cache_preparation_ms": 2.0,
                "variants": [
                    {
                        "name": "eager_kv_cache",
                        "status": "ok",
                        "timing_ms": _timing(8.0, 9.0),
                        "correctness_vs_eager_kv_cache": {
                            "max_abs": 0.0,
                            "passed": True,
                        },
                    }
                ],
                "winner_variant": "eager_kv_cache",
                "five_step_winner": {"status": "ok"},
                "cuda_graph_decision": {"decision": "captured", "reason": "static shapes"},
            },
        }
    )
    summary = render_markdown(_payload(arm))
    assert "cache preparation: 2.00 ms" in summary
    assert "eager_kv_cache" in summary
    assert "CUDA graph decision: `captured`" in summary
