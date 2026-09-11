"""Targeted unit tests for train_cosmos_t2_distillation.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lerobot.policies.vam.base.split_guard import ProtocolViolationError
from scripts.video_vam.train_cosmos_t2_distillation import (
    main as distill_main,
)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _create_mock_cosmos_manifest(
    root: Path,
    file_name: str,
    episodes: list[int],
) -> Path:
    entries = []
    total_bytes = 0
    for ep in episodes:
        frame = 4
        f_name = f"episode_{ep:04d}_frame_{frame:06d}.safetensors"
        sidecar_name = f"episode_{ep:04d}_frame_{frame:06d}.json"
        f_path = root / f_name
        sidecar_path = root / sidecar_name

        # Non-constant distinct target tokens
        torch.manual_seed(ep * 100 + frame)
        context = torch.randn(2400, 2048, dtype=torch.float32)
        save_file({"context": context}, str(f_path))
        sidecar_path.write_text(json.dumps({"episode_index": ep, "frame_index": frame}))

        entry_bytes = f_path.stat().st_size + sidecar_path.stat().st_size
        total_bytes += entry_bytes

        entries.append(
            {
                "sample_id": f"episode-{ep:04d}-frame-{frame:06d}",
                "episode_index": ep,
                "frame_index": frame,
                "window_indices": list(range(frame - 4, frame + 1)),
                "safetensors": f_name,
                "sidecar": sidecar_name,
                "safetensors_sha256": _sha256_file(f_path),
                "sidecar_sha256": _sha256_file(sidecar_path),
                "noise_seed": 42 + ep,
                "bytes": entry_bytes,
            }
        )

    payload = {
        "schema_version": 1,
        "cache_schema_version": 2,
        "dataset": {
            "repo_id": "hubnemo/cube_out_of_box_dataset",
            "revision": "243370c3c08bcbd860133c4a0d658ea7c1d2e77e",
        },
        "subset": {
            "episodes": episodes,
            "frame_start": 4,
            "frame_end": 100,
            "max_samples": 100,
            "stride": 3,
        },
        "provenance": {"backbone": "cosmos2b", "test": True},
        "global_seed": 42,
        "entries": entries,
        "total_bytes": total_bytes,
        "runtime": {"elapsed_seconds": 1.0, "samples_per_second": 10.0},
    }

    manifest_path = root / file_name
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def test_distillation_splits_and_leakage_block(tmp_path: Path):
    train_dir = tmp_path / "teacher_train_leak"
    val_dir = tmp_path / "teacher_val_leak"
    train_dir.mkdir()
    val_dir.mkdir()

    train_m = _create_mock_cosmos_manifest(train_dir, "manifest.json", episodes=[0])
    # Leaked: episode 0 in val manifest
    val_m = _create_mock_cosmos_manifest(val_dir, "manifest.json", episodes=[0])

    out_dir = tmp_path / "distill_out"

    with pytest.raises(ProtocolViolationError, match="DATA LEAKAGE DETECTED"):
        distill_main(
            [
                "--train-manifest",
                str(train_m),
                "--val-manifest",
                str(val_m),
                "--output-dir",
                str(out_dir),
                "--dry-run",
                "--device",
                "cpu",
            ]
        )


def test_distillation_dry_run_and_artifacts(tmp_path: Path):
    train_dir = tmp_path / "teacher_train"
    val_dir = tmp_path / "teacher_val"
    train_dir.mkdir()
    val_dir.mkdir()

    train_m = _create_mock_cosmos_manifest(train_dir, "manifest.json", episodes=[0, 1])
    val_m = _create_mock_cosmos_manifest(val_dir, "manifest.json", episodes=[32])

    out_dir = tmp_path / "distill_run"

    ret = distill_main(
        [
            "--train-manifest",
            str(train_m),
            "--val-manifest",
            str(val_m),
            "--output-dir",
            str(out_dir),
            "--max-steps",
            "4",
            "--val-every",
            "2",
            "--batch-size",
            "1",
            "--dry-run",
            "--device",
            "cpu",
            "--no-wandb",
        ]
    )
    assert ret == 0
    assert (out_dir / "init.json").is_file()
    assert (out_dir / "best.json").is_file()
    assert (out_dir / "last.json").is_file()
    assert (out_dir / "checkpoint.pt").is_file()
    assert (out_dir / "final_distill_summary.json").is_file()
    assert (out_dir / "metrics.jsonl").is_file()


def test_distillation_resume_and_output_safety(tmp_path: Path):
    train_dir = tmp_path / "teacher_train"
    val_dir = tmp_path / "teacher_val"
    train_dir.mkdir()
    val_dir.mkdir()

    train_m = _create_mock_cosmos_manifest(train_dir, "manifest.json", episodes=[0])
    val_m = _create_mock_cosmos_manifest(val_dir, "manifest.json", episodes=[32])

    out_dir = tmp_path / "distill_resume_test"

    # Step 1: Run with total schedule 4, interrupted at step 2 via --stop-after-steps 2
    distill_main(
        [
            "--train-manifest",
            str(train_m),
            "--val-manifest",
            str(val_m),
            "--output-dir",
            str(out_dir),
            "--max-steps",
            "4",
            "--stop-after-steps",
            "2",
            "--val-every",
            "2",
            "--batch-size",
            "1",
            "--dry-run",
            "--device",
            "cpu",
            "--no-wandb",
        ]
    )

    # Step 2: Try running without --resume or --overwrite on non-empty -> FileExistsError
    with pytest.raises(FileExistsError, match="is not empty"):
        distill_main(
            [
                "--train-manifest",
                str(train_m),
                "--val-manifest",
                str(val_m),
                "--output-dir",
                str(out_dir),
                "--max-steps",
                "4",
                "--val-every",
                "2",
                "--batch-size",
                "1",
                "--dry-run",
                "--device",
                "cpu",
                "--no-wandb",
            ]
        )

    # Step 3: Run with --resume to complete full schedule of 4 steps
    ret = distill_main(
        [
            "--train-manifest",
            str(train_m),
            "--val-manifest",
            str(val_m),
            "--output-dir",
            str(out_dir),
            "--max-steps",
            "4",
            "--val-every",
            "2",
            "--batch-size",
            "1",
            "--dry-run",
            "--device",
            "cpu",
            "--resume",
            "--no-wandb",
        ]
    )
    assert ret == 0

    with open(out_dir / "final_distill_summary.json") as f:
        summary = json.load(f)
        assert summary["total_steps"] == 4


def test_distillation_uninterrupted_vs_resumed_equivalence(tmp_path: Path):
    """Verify exact numerical equivalence between uninterrupted run and run resumed midway."""
    train_dir = tmp_path / "teacher_train_eq"
    val_dir = tmp_path / "teacher_val_eq"
    train_dir.mkdir()
    val_dir.mkdir()

    train_m = _create_mock_cosmos_manifest(train_dir, "manifest.json", episodes=[0, 1])
    val_m = _create_mock_cosmos_manifest(val_dir, "manifest.json", episodes=[32])

    uninterrupted_dir = tmp_path / "out_uninterrupted"
    resumed_dir = tmp_path / "out_resumed"

    # 1. Run 4 steps uninterrupted
    distill_main(
        [
            "--train-manifest",
            str(train_m),
            "--val-manifest",
            str(val_m),
            "--output-dir",
            str(uninterrupted_dir),
            "--max-steps",
            "4",
            "--val-every",
            "2",
            "--batch-size",
            "1",
            "--seed",
            "123",
            "--dry-run",
            "--device",
            "cpu",
            "--no-wandb",
        ]
    )

    # 2. Run with immutable schedule 4, interrupted at step 2 via --stop-after-steps 2
    distill_main(
        [
            "--train-manifest",
            str(train_m),
            "--val-manifest",
            str(val_m),
            "--output-dir",
            str(resumed_dir),
            "--max-steps",
            "4",
            "--stop-after-steps",
            "2",
            "--val-every",
            "2",
            "--batch-size",
            "1",
            "--seed",
            "123",
            "--dry-run",
            "--device",
            "cpu",
            "--no-wandb",
        ]
    )

    # 3. Resume with immutable schedule 4 to completion
    distill_main(
        [
            "--train-manifest",
            str(train_m),
            "--val-manifest",
            str(val_m),
            "--output-dir",
            str(resumed_dir),
            "--max-steps",
            "4",
            "--val-every",
            "2",
            "--batch-size",
            "1",
            "--seed",
            "123",
            "--dry-run",
            "--device",
            "cpu",
            "--resume",
            "--no-wandb",
        ]
    )

    # Check metrics.jsonl exact match
    uninterrupted_metrics = [
        json.loads(line) for line in (uninterrupted_dir / "metrics.jsonl").read_text().splitlines() if line
    ]
    resumed_metrics = [
        json.loads(line) for line in (resumed_dir / "metrics.jsonl").read_text().splitlines() if line
    ]

    assert len(uninterrupted_metrics) == len(resumed_metrics)
    for m_un, m_res in zip(uninterrupted_metrics, resumed_metrics, strict=True):
        assert m_un["step"] == m_res["step"]
        assert m_un["val_loss"] == pytest.approx(m_res["val_loss"], abs=1e-6)
        assert m_un["train_loss"] == pytest.approx(m_res["train_loss"], abs=1e-6)

    # Check final checkpoint weights match exactly
    ckpt_un = torch.load(uninterrupted_dir / "checkpoint.pt", map_location="cpu")
    ckpt_res = torch.load(resumed_dir / "checkpoint.pt", map_location="cpu")

    for k in ckpt_un["lora_state_dict"]:
        assert torch.allclose(ckpt_un["lora_state_dict"][k], ckpt_res["lora_state_dict"][k], atol=1e-6)
