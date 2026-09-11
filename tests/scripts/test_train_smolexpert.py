"""Targeted unit tests for train_smolexpert.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lerobot.policies.vam.smol_expert import ACTION_DIM, ACTION_HORIZON, SmolVLANormalizer
from scripts.video_vam.train_smolexpert import (
    UnifiedFeatureCacheDataset,
    build_synthetic_tiny_decoder,
    evaluate_validation,
    main as train_smolexpert_main,
)


def _create_mock_cache_item(
    root: Path,
    file_name: str,
    *,
    context: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
    action: torch.Tensor | None = None,
    action_is_pad: torch.Tensor | None = None,
) -> Path:
    tensors: dict[str, torch.Tensor] = {}
    if context is not None:
        tensors["context"] = context
    if state is not None:
        tensors["state"] = state
    if action is not None:
        tensors["action"] = action
    if action_is_pad is not None:
        tensors["action_is_pad"] = action_is_pad

    file_path = root / file_name
    save_file(tensors, str(file_path))
    return file_path


def test_strict_cache_required_keys_and_no_zero_fallback(tmp_path: Path):
    cache_dir = tmp_path / "cache_strict"
    cache_dir.mkdir()

    # 1. Missing state -> KeyError
    f1 = _create_mock_cache_item(
        cache_dir,
        "missing_state.safetensors",
        context=torch.randn(4, 16),
        action=torch.zeros(ACTION_HORIZON, ACTION_DIM),
        action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
    )
    manifest1 = {
        "entries": [{"safetensors": f1.name, "episode_index": 0, "frame_index": 0, "sample_id": "s1"}]
    }
    m_path1 = cache_dir / "m1.json"
    m_path1.write_text(json.dumps(manifest1))
    ds1 = UnifiedFeatureCacheDataset(m_path1)
    with pytest.raises(KeyError, match="required 'state' tensor"):
        _ = ds1[0]

    # 2. Missing action -> KeyError
    f2 = _create_mock_cache_item(
        cache_dir,
        "missing_action.safetensors",
        context=torch.randn(4, 16),
        state=torch.zeros(ACTION_DIM),
        action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
    )
    manifest2 = {
        "entries": [{"safetensors": f2.name, "episode_index": 0, "frame_index": 0, "sample_id": "s2"}]
    }
    m_path2 = cache_dir / "m2.json"
    m_path2.write_text(json.dumps(manifest2))
    ds2 = UnifiedFeatureCacheDataset(m_path2)
    with pytest.raises(KeyError, match="required 'target_action' or 'action' tensor"):
        _ = ds2[0]

    # 3. Missing action_is_pad -> KeyError
    f3 = _create_mock_cache_item(
        cache_dir,
        "missing_pad.safetensors",
        context=torch.randn(4, 16),
        state=torch.zeros(ACTION_DIM),
        action=torch.zeros(ACTION_HORIZON, ACTION_DIM),
    )
    manifest3 = {
        "entries": [{"safetensors": f3.name, "episode_index": 0, "frame_index": 0, "sample_id": "s3"}]
    }
    m_path3 = cache_dir / "m3.json"
    m_path3.write_text(json.dumps(manifest3))
    ds3 = UnifiedFeatureCacheDataset(m_path3)
    with pytest.raises(KeyError, match="required 'action_is_pad' tensor"):
        _ = ds3[0]

    # 4. Missing context -> KeyError
    f4 = _create_mock_cache_item(
        cache_dir,
        "missing_context.safetensors",
        state=torch.zeros(ACTION_DIM),
        action=torch.zeros(ACTION_HORIZON, ACTION_DIM),
        action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
    )
    manifest4 = {
        "entries": [{"safetensors": f4.name, "episode_index": 0, "frame_index": 0, "sample_id": "s4"}]
    }
    m_path4 = cache_dir / "m4.json"
    m_path4.write_text(json.dumps(manifest4))
    ds4 = UnifiedFeatureCacheDataset(m_path4)
    with pytest.raises(KeyError, match="required 'context' or 'features' tensor"):
        _ = ds4[0]


def test_strict_cache_legacy_state_shapes(tmp_path: Path):
    cache_dir = tmp_path / "cache_shapes"
    cache_dir.mkdir()

    # Shape [6]
    f1 = _create_mock_cache_item(
        cache_dir,
        "shape_6.safetensors",
        context=torch.randn(4, 16),
        state=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        action=torch.zeros(ACTION_HORIZON, ACTION_DIM),
        action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
    )

    # Shape [1, 6]
    f2 = _create_mock_cache_item(
        cache_dir,
        "shape_1_6.safetensors",
        context=torch.randn(4, 16),
        state=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        action=torch.zeros(ACTION_HORIZON, ACTION_DIM),
        action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
    )

    # Legacy shape [1, 1, 6]
    f3 = _create_mock_cache_item(
        cache_dir,
        "shape_1_1_6.safetensors",
        context=torch.randn(4, 16),
        state=torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]),
        action=torch.zeros(ACTION_HORIZON, ACTION_DIM),
        action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
    )

    manifest = {
        "entries": [
            {"safetensors": f1.name, "episode_index": 0, "frame_index": 0, "sample_id": "s1"},
            {"safetensors": f2.name, "episode_index": 0, "frame_index": 1, "sample_id": "s2"},
            {"safetensors": f3.name, "episode_index": 0, "frame_index": 2, "sample_id": "s3"},
        ]
    }
    m_path = cache_dir / "m.json"
    m_path.write_text(json.dumps(manifest))
    ds = UnifiedFeatureCacheDataset(m_path)

    for i in range(3):
        item = ds[i]
        assert item.state.shape == (ACTION_DIM,)
        assert item.state[0].item() == 1.0


def test_strict_cache_non_finite_rejection(tmp_path: Path):
    cache_dir = tmp_path / "cache_nan"
    cache_dir.mkdir()

    # NaN in state
    f1 = _create_mock_cache_item(
        cache_dir,
        "nan_state.safetensors",
        context=torch.randn(4, 16),
        state=torch.tensor([float("nan"), 2.0, 3.0, 4.0, 5.0, 6.0]),
        action=torch.zeros(ACTION_HORIZON, ACTION_DIM),
        action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
    )
    m_path1 = cache_dir / "m1.json"
    m_path1.write_text(
        json.dumps(
            {"entries": [{"safetensors": f1.name, "episode_index": 0, "frame_index": 0, "sample_id": "s1"}]}
        )
    )
    ds1 = UnifiedFeatureCacheDataset(m_path1)
    with pytest.raises(ValueError, match="non-finite values"):
        _ = ds1[0]

    # Inf in action
    f2 = _create_mock_cache_item(
        cache_dir,
        "inf_action.safetensors",
        context=torch.randn(4, 16),
        state=torch.zeros(ACTION_DIM),
        action=torch.full((ACTION_HORIZON, ACTION_DIM), float("inf")),
        action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
    )
    m_path2 = cache_dir / "m2.json"
    m_path2.write_text(
        json.dumps(
            {"entries": [{"safetensors": f2.name, "episode_index": 0, "frame_index": 0, "sample_id": "s2"}]}
        )
    )
    ds2 = UnifiedFeatureCacheDataset(m_path2)
    with pytest.raises(ValueError, match="non-finite values"):
        _ = ds2[0]


def test_strict_cache_hash_verification(tmp_path: Path):
    cache_dir = tmp_path / "cache_hash"
    cache_dir.mkdir()

    import hashlib

    f1 = _create_mock_cache_item(
        cache_dir,
        "valid_hash.safetensors",
        context=torch.randn(4, 16),
        state=torch.zeros(ACTION_DIM),
        action=torch.zeros(ACTION_HORIZON, ACTION_DIM),
        action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
    )
    true_hash = hashlib.sha256(f1.read_bytes()).hexdigest()

    # Correct hash -> succeeds
    m_path1 = cache_dir / "m1.json"
    m_path1.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "safetensors": f1.name,
                        "episode_index": 0,
                        "frame_index": 0,
                        "sample_id": "s1",
                        "sha256": true_hash,
                    }
                ]
            }
        )
    )
    ds1 = UnifiedFeatureCacheDataset(m_path1, verify_hashes=True)
    assert ds1[0].sample_id == "s1"

    # Bad hash -> raises ValueError on construction
    m_path2 = cache_dir / "m2.json"
    m_path2.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "safetensors": f1.name,
                        "episode_index": 0,
                        "frame_index": 0,
                        "sample_id": "s1",
                        "sha256": "corrupted_hash",
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _ = UnifiedFeatureCacheDataset(m_path2, verify_hashes=True)


def test_eval_noise_per_sample_seeded_and_batch_and_order_invariant(tmp_path: Path):
    cache_dir = tmp_path / "cache_eval"
    cache_dir.mkdir()

    entries = []
    for i in range(4):
        f = _create_mock_cache_item(
            cache_dir,
            f"val_{i}.safetensors",
            context=torch.randn(4, 16),
            state=torch.full((ACTION_DIM,), float(i + 1)),
            action=torch.full((ACTION_HORIZON, ACTION_DIM), float(i + 1)),
            action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
        )
        entries.append(
            {"safetensors": f.name, "episode_index": 32, "frame_index": i, "sample_id": f"sample_{i}"}
        )

    m_path = cache_dir / "m.json"
    m_path.write_text(json.dumps({"entries": entries}))
    val_dataset = UnifiedFeatureCacheDataset(m_path)

    # Reordered dataset (reversed order)
    m_rev_path = cache_dir / "m_rev.json"
    m_rev_path.write_text(json.dumps({"entries": list(reversed(entries))}))
    val_dataset_rev = UnifiedFeatureCacheDataset(m_rev_path)

    decoder = build_synthetic_tiny_decoder(input_channels=16, device="cpu")

    # Evaluate with batch_size = 1 vs batch_size = 2 vs batch_size = 4
    res_b1 = evaluate_validation(
        decoder, val_dataset, device=torch.device("cpu"), batch_size=1, num_steps=2, seed=123
    )
    res_b2 = evaluate_validation(
        decoder, val_dataset, device=torch.device("cpu"), batch_size=2, num_steps=2, seed=123
    )
    res_b4 = evaluate_validation(
        decoder, val_dataset, device=torch.device("cpu"), batch_size=4, num_steps=2, seed=123
    )
    res_rev = evaluate_validation(
        decoder, val_dataset_rev, device=torch.device("cpu"), batch_size=2, num_steps=2, seed=123
    )

    # Batch invariance
    assert res_b1["val_rmse"] == pytest.approx(res_b2["val_rmse"], abs=1e-6)
    assert res_b1["val_rmse"] == pytest.approx(res_b4["val_rmse"], abs=1e-6)
    assert res_b1["arm_rmse_deg"] == pytest.approx(res_b4["arm_rmse_deg"], abs=1e-6)
    assert res_b1["gripper_rmse"] == pytest.approx(res_b4["gripper_rmse"], abs=1e-6)

    # Order invariance (Reversed dataset gives exact same aggregate and per-joint metrics)
    assert res_b1["val_rmse"] == pytest.approx(res_rev["val_rmse"], abs=1e-6)
    assert res_b1["arm_rmse_deg"] == pytest.approx(res_rev["arm_rmse_deg"], abs=1e-6)
    assert res_b1["gripper_rmse"] == pytest.approx(res_rev["gripper_rmse"], abs=1e-6)
    assert res_b1["val_first5"] == pytest.approx(res_rev["val_first5"], abs=1e-6)
    assert res_b1["val_first5_pooled_rmse"] == pytest.approx(res_rev["val_first5_pooled_rmse"], abs=1e-6)


def test_eval_independent_of_train_rng(tmp_path: Path):
    cache_dir = tmp_path / "cache_rng"
    cache_dir.mkdir()

    entries = []
    for i in range(2):
        f = _create_mock_cache_item(
            cache_dir,
            f"sample_{i}.safetensors",
            context=torch.randn(4, 16),
            state=torch.zeros(ACTION_DIM),
            action=torch.zeros(ACTION_HORIZON, ACTION_DIM),
            action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
        )
        entries.append({"safetensors": f.name, "episode_index": 32, "frame_index": i, "sample_id": f"s_{i}"})

    m_path = cache_dir / "m.json"
    m_path.write_text(json.dumps({"entries": entries}))
    val_dataset = UnifiedFeatureCacheDataset(m_path)
    decoder = build_synthetic_tiny_decoder(input_channels=16, device="cpu")

    # Set torch RNG state
    torch.manual_seed(999)
    rng_before = torch.get_rng_state()

    _ = evaluate_validation(
        decoder, val_dataset, device=torch.device("cpu"), batch_size=1, num_steps=2, seed=42
    )

    rng_after = torch.get_rng_state()
    # Global RNG state must not be mutated by evaluate_validation
    assert torch.equal(rng_before, rng_after)


def test_eval_only_requires_checkpoint_normalizer_and_no_recompute_overwrite(tmp_path: Path):
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    f = _create_mock_cache_item(
        val_dir,
        "val_0.safetensors",
        context=torch.randn(4, 16),
        state=torch.zeros(ACTION_DIM),
        action=torch.zeros(ACTION_HORIZON, ACTION_DIM),
        action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
    )
    val_m = val_dir / "manifest.json"
    val_m.write_text(
        json.dumps(
            {"entries": [{"safetensors": f.name, "episode_index": 32, "frame_index": 0, "sample_id": "v0"}]}
        )
    )

    out_dir = tmp_path / "eval_out"
    out_dir.mkdir()

    # Without normalizer in out_dir -> FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Normalizer not found"):
        train_smolexpert_main(
            [
                "--val-manifest",
                str(val_m),
                "--output-dir",
                str(out_dir),
                "--eval-only",
                "--dry-run",
                "--device",
                "cpu",
            ]
        )

    # Save mock normalizer
    norm = SmolVLANormalizer(
        state_mean=torch.zeros(ACTION_DIM),
        state_std=torch.ones(ACTION_DIM),
        action_mean=torch.zeros(ACTION_DIM),
        action_std=torch.ones(ACTION_DIM),
    )
    norm_path = out_dir / "normalizer.safetensors"
    save_file(norm.state_dict(), str(norm_path))
    orig_mtime = norm_path.stat().st_mtime
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text("existing training manifest")

    ret = train_smolexpert_main(
        [
            "--val-manifest",
            str(val_m),
            "--output-dir",
            str(out_dir),
            "--eval-only",
            "--dry-run",
            "--device",
            "cpu",
        ]
    )
    assert ret == 0
    assert (out_dir / "eval_metrics.json").is_file()
    # Normalizer must NOT be overwritten in eval-only mode
    assert norm_path.stat().st_mtime == orig_mtime
    assert manifest_path.read_text() == "existing training manifest"


def test_cli_output_safety_non_empty_and_overwrite(tmp_path: Path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    f_tr = _create_mock_cache_item(
        train_dir,
        "tr.safetensors",
        context=torch.randn(4, 16),
        state=torch.zeros(ACTION_DIM),
        action=torch.zeros(ACTION_HORIZON, ACTION_DIM),
        action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
    )
    train_m = train_dir / "manifest.json"
    train_m.write_text(
        json.dumps(
            {
                "entries": [
                    {"safetensors": f_tr.name, "episode_index": 0, "frame_index": 0, "sample_id": "tr0"}
                ]
            }
        )
    )

    f_va = _create_mock_cache_item(
        val_dir,
        "va.safetensors",
        context=torch.randn(4, 16),
        state=torch.zeros(ACTION_DIM),
        action=torch.zeros(ACTION_HORIZON, ACTION_DIM),
        action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
    )
    val_m = val_dir / "manifest.json"
    val_m.write_text(
        json.dumps(
            {
                "entries": [
                    {"safetensors": f_va.name, "episode_index": 32, "frame_index": 0, "sample_id": "va0"}
                ]
            }
        )
    )

    out_dir = tmp_path / "out_safety"
    out_dir.mkdir()
    (out_dir / "stray_file.txt").write_text("existing content")

    # Non-empty directory without --overwrite -> FileExistsError
    with pytest.raises(FileExistsError, match="is not empty"):
        train_smolexpert_main(
            [
                "--train-manifest",
                str(train_m),
                "--val-manifest",
                str(val_m),
                "--output-dir",
                str(out_dir),
                "--max-steps",
                "2",
                "--dry-run",
                "--device",
                "cpu",
            ]
        )

    # With --overwrite -> succeeds
    ret = train_smolexpert_main(
        [
            "--train-manifest",
            str(train_m),
            "--val-manifest",
            str(val_m),
            "--output-dir",
            str(out_dir),
            "--max-steps",
            "2",
            "--dry-run",
            "--device",
            "cpu",
            "--overwrite",
            "--no-wandb",
        ]
    )
    assert ret == 0
    assert (out_dir / "best.safetensors").is_file()
    # Unrelated files are preserved
    assert (out_dir / "stray_file.txt").is_file()


@pytest.mark.parametrize("save_every", [0, 1])
def test_storage_periodic_last_and_final_before_best_reload(tmp_path, monkeypatch, save_every):
    from safetensors.torch import load_file

    from scripts.video_vam import train_smolexpert as trainer

    original_parse = trainer.parse_args
    original_eval = trainer.evaluate_validation
    original_save = trainer.save_weights_artifact
    saves = []
    eval_count = 0

    def parse(argv):
        args = original_parse(argv)
        args.val_every = 1
        args.save_last_every = save_every
        args.max_steps = 4
        args.patience = 1
        return args

    def evaluate(*args, **kwargs):
        nonlocal eval_count
        metrics = original_eval(*args, **kwargs)
        eval_count += 1
        metrics["val_rmse"] = [3.0, 1.0, 2.0, 1.0][eval_count - 1]
        return metrics

    def save(output_dir, role, state, step, manifest, **kwargs):
        saves.append((role, step, eval_count))
        return original_save(output_dir, role, state, step, manifest, **kwargs)

    monkeypatch.setattr(trainer, "parse_args", parse)
    monkeypatch.setattr(trainer, "evaluate_validation", evaluate)
    monkeypatch.setattr(trainer, "save_weights_artifact", save)
    test_cli_output_safety_non_empty_and_overwrite(tmp_path)
    out = tmp_path / "out_safety"
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["checkpoints"]["best"]["optimizer_step"] == 1
    assert manifest["checkpoints"]["last"]["optimizer_step"] == 2
    expected = [("last", 1, 1), ("best", 1, 2), ("last", 2, 2), ("last", 2, 3)]
    assert saves == (expected if save_every else [("best", 1, 2), ("last", 2, 3)])
    best = load_file(str(out / "best.safetensors"))
    last = load_file(str(out / "last.safetensors"))
    assert any(not torch.equal(best[key], last[key]) for key in best)
    assert manifest["normalizer"]["path"] == "normalizer.safetensors"
    assert manifest["datasets"]["train"]["episodes"] == [0]
    assert manifest["policy"]["pretrained_revision"] is None


def test_output_default_and_explicit_path(monkeypatch, tmp_path):
    from scripts.video_vam import train_smolexpert as trainer

    monkeypatch.chdir(tmp_path)
    args = trainer.parse_args(["--val-manifest", "val.json"])
    assert args.output_dir.parent == Path(trainer.__file__).resolve().parents[2] / "outputs/train"
    explicit = trainer.parse_args(["--val-manifest", "val.json", "--output-dir", "custom"])
    assert explicit.output_dir == Path("custom")
    with pytest.raises(SystemExit):
        trainer.parse_args(["--val-manifest", "val.json", "--eval-only"])
