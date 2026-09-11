"""Unit tests for Cosmos-1.0-Diffusion-14B video rollout generation."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.video_vam.render_cosmos14b_rollout import (
    compute_metrics,
)


def test_compute_metrics_identical():
    """Identical frames should yield MSE 0, PSNR 50, and SSIM 1.0."""
    frames = np.random.randint(0, 256, size=(5, 64, 64, 3), dtype=np.uint8)
    metrics = compute_metrics(frames, frames)

    assert metrics["mean_mse"] == 0.0
    assert metrics["mean_psnr_db"] == 50.0
    assert pytest.approx(metrics["mean_ssim"], abs=1e-4) == 1.0
    assert len(metrics["per_frame_psnr"]) == 5
    assert len(metrics["per_frame_ssim"]) == 5
    assert len(metrics["per_frame_mse"]) == 5


def test_compute_metrics_delta():
    """Perturbed frames should yield positive MSE, valid PSNR < 50, and SSIM < 1.0."""
    target = np.full((3, 32, 32, 3), 100, dtype=np.uint8)
    pred = target.copy()
    pred[:, :, :, 0] += 10  # Perturb red channel

    metrics = compute_metrics(pred, target)

    assert metrics["mean_mse"] > 0.0
    assert 0.0 < metrics["mean_psnr_db"] < 50.0
    assert 0.0 <= metrics["mean_ssim"] <= 1.0
