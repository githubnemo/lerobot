import pytest
import torch

from scripts.video_vam.benchmark_ltx_extraction import (
    CORRECTNESS_COSINE_THRESHOLD,
    _percentile,
    _summary,
    _tensor_metrics,
    parse_args,
)


def test_percentiles_and_summary_are_deterministic():
    values = [0.1, 0.2, 0.3, 0.4]
    assert _percentile(values, 0.5) == pytest.approx(0.25)
    summary = _summary(values)
    assert summary["count"] == 4
    assert summary["p50_seconds"] == pytest.approx(0.25)
    assert summary["p90_seconds"] == pytest.approx(0.37)


def test_tensor_metrics_gate_identical_and_drifting_features():
    reference = torch.tensor([1.0, 2.0, 3.0])
    identical = _tensor_metrics(reference, reference.clone())
    assert identical["cosine"] == pytest.approx(1.0)
    assert identical["max_abs"] == pytest.approx(0.0)
    assert identical["rmse"] == pytest.approx(0.0)
    assert identical["relative_rmse"] == pytest.approx(0.0)

    candidate = torch.tensor([1.0, 2.0, 3.2])
    drift = _tensor_metrics(reference, candidate)
    assert drift["cosine"] < 1.0
    assert drift["max_abs"] == pytest.approx(0.2, abs=1e-6)
    assert drift["cosine"] >= CORRECTNESS_COSINE_THRESHOLD
    assert drift["reference_rms"] > 0
    assert drift["relative_rmse"] > 0


def test_parse_args_keeps_zero_prompt_explicit():
    args = parse_args(["--allow-zero-prompt", "--iterations", "3"])
    assert args.allow_zero_prompt is True
    assert args.iterations == 3
    assert args.with_expert is False
    assert args.vae_iterations == 10
    assert args.vae_compile_mode is None


def test_parse_args_accepts_opt_in_vae_compile_mode():
    args = parse_args(["--vae-compile-mode", "reduce-overhead"])
    assert args.vae_compile_mode == "reduce-overhead"
