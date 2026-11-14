"""Targeted unit tests for train_smolexpert.py."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace

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


def test_augment_temporal_window_gpu():
    from scripts.video_vam.train_smolexpert import (
        augment_temporal_window_gpu,
        unaugmented_temporal_window_gpu,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = torch.randint(0, 256, (2, 5, 3, 480, 640), dtype=torch.uint8, device=device)
    unaug = unaugmented_temporal_window_gpu(batch)
    assert unaug.shape == (2, 3, 5, 480, 640)
    assert 0.0 <= unaug.min().item() <= unaug.max().item() <= 1.0

    aug = augment_temporal_window_gpu(batch, p_blur=0.5)
    assert aug.shape == (2, 3, 5, 480, 640)
    assert 0.0 <= aug.min().item() <= aug.max().item() <= 1.0


def test_online_backbone_dry_run(cosmos3_assets, fake_online_dataset):
    from scripts.video_vam import train_smolexpert as trainer

    out_dir = cosmos3_assets.root / "online_out"
    ret = trainer.main(
        cosmos3_assets.argv
        + [
            "--output-dir",
            str(out_dir),
            "--dry-run",
            "--device",
            "cpu",
            "--augment",
            "--max-steps",
            "2",
            "--val-every",
            "1",
            "--batch-size",
            "2",
            "--protocol",
            "scale100",
            "--no-wandb",
        ]
    )
    assert ret == 0
    assert (out_dir / "best.safetensors").is_file()
    assert (out_dir / "normalizer.safetensors").is_file()
    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg["online_trained"] is True
    assert cfg["augmented"] is True
    manifest = json.loads((out_dir / "run_manifest.json").read_text())
    assert manifest["online_feature_identity"] == cosmos3_assets.identity


def test_online_cosmos2b_dry_run(tmp_path: Path, fake_online_dataset):
    from scripts.video_vam.train_smolexpert import main as train_smolexpert_main

    out_dir = tmp_path / "online_c2b_out"
    val_dir = tmp_path / "val_cache_c2b"
    val_dir.mkdir()
    f_val = _create_mock_cache_item(
        val_dir,
        "val_0.safetensors",
        context=torch.randn(2400, 2048),
        state=torch.zeros(ACTION_DIM),
        action=torch.zeros(ACTION_HORIZON, ACTION_DIM),
        action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
    )
    val_m = val_dir / "manifest.json"
    val_m.write_text(
        json.dumps(
            {
                "entries": [
                    {"safetensors": f_val.name, "episode_index": 32, "frame_index": 0, "sample_id": "v0"}
                ]
            }
        )
    )

    ret = train_smolexpert_main(
        [
            "--online-backbone",
            "cosmos2b",
            "--val-manifest",
            str(val_m),
            "--eval2-manifest",
            str(_synthetic_eval2_manifest(val_m)),
            "--output-dir",
            str(out_dir),
            "--dry-run",
            "--device",
            "cpu",
            "--augment",
            "--max-steps",
            "2",
            "--val-every",
            "1",
            "--batch-size",
            "2",
            "--protocol",
            "scale100",
        ]
    )
    assert ret == 0
    assert (out_dir / "best.safetensors").is_file()
    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg["backend"] == "cosmos"
    assert cfg["online_trained"] is True
    assert cfg["augmented"] is True


def test_online_flux2_klein_dry_run(tmp_path: Path, fake_online_dataset):
    from scripts.video_vam.train_smolexpert import main as train_smolexpert_main

    out_dir = tmp_path / "online_flux2_out"
    val_dir = tmp_path / "val_cache_flux2"
    val_dir.mkdir()
    f_val = _create_mock_cache_item(
        val_dir,
        "val_0.safetensors",
        context=torch.randn(256, 6144),
        state=torch.zeros(ACTION_DIM),
        action=torch.zeros(ACTION_HORIZON, ACTION_DIM),
        action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
    )
    val_m = val_dir / "manifest.json"
    val_m.write_text(
        json.dumps(
            {
                "entries": [
                    {"safetensors": f_val.name, "episode_index": 32, "frame_index": 0, "sample_id": "v0"}
                ]
            }
        )
    )

    ret = train_smolexpert_main(
        [
            "--online-backbone",
            "flux2_klein",
            "--val-manifest",
            str(val_m),
            "--eval2-manifest",
            str(_synthetic_eval2_manifest(val_m)),
            "--output-dir",
            str(out_dir),
            "--dry-run",
            "--device",
            "cpu",
            "--augment",
            "--max-steps",
            "2",
            "--val-every",
            "1",
            "--batch-size",
            "2",
            "--protocol",
            "scale100",
        ]
    )
    assert ret == 0
    assert (out_dir / "best.safetensors").is_file()
    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg["backend"] == "flux2_klein"
    assert cfg["online_trained"] is True
    assert cfg["augmented"] is True


def _synthetic_eval2_manifest(val_manifest: Path) -> Path:
    payload = json.loads(val_manifest.read_text())
    for entry in payload["entries"]:
        entry["episode_index"] = 90
        entry["sample_id"] = "eval2-" + entry["sample_id"]
    path = val_manifest.with_name("eval2.json")
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def fake_online_dataset(monkeypatch):
    from scripts.video_vam import train_smolexpert as trainer

    def initialize(self, repo_id, *, root=None, episodes, stride=1, contract=None):
        self.repo_id = repo_id
        self.episodes = tuple(episodes)
        self.payload = {"source": "online_video_dataset", "episodes": list(episodes)}
        self.manifest_path = Path("online:/synthetic")
        self.items = [
            trainer.OnlineVideoItem(
                rgb=torch.zeros(5, 3, 4, 4, dtype=torch.uint8),
                state=torch.full((ACTION_DIM,), float(i)),
                action=torch.full((ACTION_HORIZON, ACTION_DIM), float(i)),
                action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
                sample_id=f"online-{i}",
                episode_index=episodes[0],
                frame_index=4 + i,
            )
            for i in range(2)
        ]

    monkeypatch.setattr(trainer.OnlineVideoDataset, "__init__", initialize)
    monkeypatch.setattr(trainer.OnlineVideoDataset, "__len__", lambda self: len(self.items))
    monkeypatch.setattr(trainer.OnlineVideoDataset, "__getitem__", lambda self, index: self.items[index])


@pytest.fixture
def cosmos3_assets(tmp_path):
    from scripts.video_vam import train_smolexpert as trainer

    checkpoint = tmp_path / "backbone"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"test_checkpoint": true}')
    for component in ("transformer", "vae"):
        directory = checkpoint / component
        directory.mkdir()
        save_file({"weight": torch.ones(1)}, str(directory / "diffusion_pytorch_model.safetensors"))
    lora = tmp_path / "lora.safetensors"
    save_file({"lora_A": torch.zeros(2, 2)}, str(lora), metadata={"rank": "2", "alpha": "4.0"})
    argv = [
        "--online-backbone",
        "cosmos3_edge",
        "--backbone-checkpoint",
        str(checkpoint),
        "--backbone-lora-weights",
        str(lora),
        "--backbone-layer",
        "7",
        "--backbone-prompt",
        "saved non-default prompt",
        "--device",
        "cpu",
    ]
    args = trainer.parse_args(argv)
    identity = trainer._cosmos3_identity_from_args(args)
    paths = {}
    for split, episode, frame in (("val", 32, 4820), ("eval2", 90, 11174)):
        root = tmp_path / split
        root.mkdir()
        artifact = _create_mock_cache_item(
            root,
            "sample.safetensors",
            context=torch.zeros(600, 2048, dtype=torch.bfloat16),
            state=torch.ones(ACTION_DIM),
            action=torch.ones(ACTION_HORIZON, ACTION_DIM),
            action_is_pad=torch.tensor([False] * 28 + [True] * 2),
        )
        payload = {
            "schema_version": 1,
            "dataset": {"repo_id": "Orellius/cube_out_of_box_v2", "revision": "legacy-unverified-revision"},
            "provenance": {
                **identity,
                "builder": trainer.COSMOS3_CACHE_BUILDER,
                "lora_checkpoint": str(lora),
            },
            "runtime": {"dtype": "bfloat16"},
            "entries": [
                {
                    "sample_id": f"episode-{episode:04d}-frame-{frame:06d}",
                    "episode_index": episode,
                    "frame_index": frame,
                    "window_indices": list(range(frame - 4, frame + 1)),
                    "safetensors": artifact.name,
                    "safetensors_sha256": trainer.sha256_file(artifact),
                }
            ],
        }
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(payload))
        paths[split] = manifest
    argv += ["--val-manifest", str(paths["val"]), "--eval2-manifest", str(paths["eval2"])]
    return SimpleNamespace(
        root=tmp_path,
        checkpoint=checkpoint,
        lora=lora,
        identity=identity,
        argv=argv,
        args=trainer.parse_args(argv),
        paths=paths,
    )


def test_cosmos3_cache_matching_effective_adapter_metadata(cosmos3_assets):
    from scripts.video_vam import train_smolexpert as trainer

    identity, datasets = trainer.preflight_cosmos3_eval_caches(cosmos3_assets.args)
    # CLI rank/alpha are resolved from the adapter metadata, never left at unrelated defaults.
    assert cosmos3_assets.args.backbone_lora_rank == 2
    assert cosmos3_assets.args.backbone_lora_alpha == 4.0
    assert identity["lora_rank"] == 2
    assert identity["lora_alpha"] == 4.0
    assert set(datasets) == {"val", "eval2"}
    assert datasets["val"][0].sample_id == "episode-0032-frame-004820"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lora_sha256", "b" * 64),
        ("transformer_sha256", "c" * 64),
        ("lora_rank", 16),
        ("lora_alpha", 32.0),
        ("hidden_layer", 20),
        ("prompt", "different prompt"),
        ("prompt_sha256", "d" * 64),
        ("fps", 24.0),
        ("base_fps", 10.0),
        ("context_tokens", 2400),
        ("context_dim", 4096),
        ("backbone", "cosmos2b"),
        ("builder", "unknown.py"),
        ("augmented", True),
        ("preprocessing_version", "unknown"),
    ],
)
def test_cosmos3_cache_rejects_mismatched_identity(cosmos3_assets, field, value):
    from scripts.video_vam import train_smolexpert as trainer

    path = cosmos3_assets.paths["eval2"]
    payload = json.loads(path.read_text())
    payload["provenance"][field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Cosmos3"):
        trainer.preflight_cosmos3_eval_caches(cosmos3_assets.args)


@pytest.mark.parametrize(
    "field",
    [
        "lora_sha256",
        "lora_checkpoint",
        "lora_rank",
        "lora_alpha",
        "transformer_sha256",
        "hidden_layer",
        "prompt",
        "prompt_sha256",
        "fps",
        "base_fps",
        "context_tokens",
        "context_dim",
        "builder",
    ],
)
def test_cosmos3_cache_rejects_missing_identity(cosmos3_assets, field):
    from scripts.video_vam import train_smolexpert as trainer

    path = cosmos3_assets.paths["val"]
    payload = json.loads(path.read_text())
    del payload["provenance"][field]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Cosmos3"):
        trainer.preflight_cosmos3_eval_caches(cosmos3_assets.args)


def test_cosmos3_base_cache_requires_explicit_no_adapter(cosmos3_assets):
    from scripts.video_vam import train_smolexpert as trainer

    args = cosmos3_assets.args
    args.backbone_lora_weights = None
    with pytest.raises(ValueError, match="lora_sha256"):
        trainer.preflight_cosmos3_eval_caches(args)
    for path in cosmos3_assets.paths.values():
        payload = json.loads(path.read_text())
        for field in ("lora_checkpoint", "lora_sha256", "lora_rank", "lora_alpha"):
            payload["provenance"][field] = None
        path.write_text(json.dumps(payload))
    identity, datasets = trainer.preflight_cosmos3_eval_caches(args)
    assert identity["lora_sha256"] is None
    assert len(datasets) == 2
    args.backbone_lora_weights = cosmos3_assets.lora
    with pytest.raises(ValueError, match="lora_sha256"):
        trainer.preflight_cosmos3_eval_caches(args)


@pytest.mark.parametrize("changed", ["checkpoint", "lora"])
def test_cosmos3_cache_checks_current_files_not_paths(cosmos3_assets, changed):
    from scripts.video_vam import train_smolexpert as trainer

    if changed == "checkpoint":
        (cosmos3_assets.checkpoint / "config.json").write_text('{"changed": true}')
    else:
        save_file(
            {"lora_A": torch.ones(2, 2)}, str(cosmos3_assets.lora), metadata={"rank": "2", "alpha": "4.0"}
        )
    with pytest.raises(ValueError, match="sha256"):
        trainer.preflight_cosmos3_eval_caches(cosmos3_assets.args)


def _unexpected(*args, **kwargs):
    raise AssertionError("Unexpected training, mutation, download, or backbone initialization")


@pytest.mark.parametrize("failure", ["mismatch", "missing_manifest"])
def test_cosmos3_preflight_before_writes_or_models(cosmos3_assets, monkeypatch, failure):
    from lerobot.policies.vam import cosmos3_features
    from scripts.video_vam import train_smolexpert as trainer

    out = cosmos3_assets.root / "existing_run"
    out.mkdir()
    sentinel = out / "normalizer.safetensors"
    sentinel.write_bytes(b"do not replace")
    original_stat = sentinel.stat().st_mtime_ns
    if failure == "missing_manifest":
        cosmos3_assets.paths["eval2"].unlink()
        error = FileNotFoundError
    else:
        path = cosmos3_assets.paths["eval2"]
        payload = json.loads(path.read_text())
        payload["provenance"]["lora_sha256"] = "f" * 64
        path.write_text(json.dumps(payload))
        error = ValueError
    monkeypatch.setattr(trainer, "clean_own_artifacts", _unexpected)
    monkeypatch.setattr(trainer, "compute_online_training_normalizer", _unexpected)
    monkeypatch.setattr(trainer.OnlineVideoDataset, "__init__", _unexpected)
    monkeypatch.setattr(cosmos3_features, "Cosmos3FeatureExtractor", _unexpected)
    monkeypatch.setattr(trainer.SmolExpertActionDecoder, "from_pretrained", _unexpected)
    with pytest.raises(error):
        trainer.main(cosmos3_assets.argv + ["--output-dir", str(out), "--overwrite"])
    assert list(out.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"do not replace"
    assert sentinel.stat().st_mtime_ns == original_stat


def test_cosmos3_no_caches_keeps_raw_eval_and_no_auto_discovery(tmp_path, monkeypatch):
    from scripts.video_vam import train_smolexpert as trainer

    class RawDatasetReachedError(Exception):
        pass

    def raw_dataset(self, **kwargs):
        assert kwargs["episodes"] == tuple(range(32, 40))
        raise RawDatasetReachedError

    monkeypatch.setattr(trainer, "_cosmos3_identity_from_args", _unexpected)
    monkeypatch.setattr(trainer, "UnifiedFeatureCacheDataset", _unexpected)
    monkeypatch.setattr(trainer.OnlineVideoDataset, "__init__", raw_dataset)
    with pytest.raises(RawDatasetReachedError):
        trainer.main(
            ["--online-backbone", "cosmos3_edge", "--output-dir", str(tmp_path / "out"), "--device", "cpu"]
        )


def test_cosmos3_direct_cached_extractor_cannot_bypass_identity(cosmos3_assets):
    from scripts.video_vam import train_smolexpert as trainer

    args = cosmos3_assets.args
    config = SimpleNamespace(
        backbone_name="cosmos3-edge",
        checkpoint_path=args.backbone_checkpoint,
        lora_checkpoint=args.backbone_lora_weights,
        lora_rank=args.backbone_lora_rank,
        lora_alpha=args.backbone_lora_alpha,
        hidden_layers=(args.backbone_layer,),
        prompt=args.backbone_prompt,
        fps=10.0,
        base_fps=24.0,
        dtype="bfloat16",
        pool_spatial=None,
        vae_path=None,
    )
    extractor = SimpleNamespace(config=config, extract=_unexpected)
    dataset = trainer.UnifiedFeatureCacheDataset(cosmos3_assets.paths["val"])
    decoder = trainer.build_synthetic_tiny_decoder(input_channels=2048)
    result = trainer.evaluate_validation(
        decoder, dataset, extractor=extractor, device=torch.device("cpu"), num_steps=1
    )
    assert result["val_samples"] == 1
    dataset.payload["provenance"]["lora_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="lora_sha256"):
        trainer.evaluate_validation(decoder, dataset, extractor=extractor, device=torch.device("cpu"))
    assert decoder.training


def test_raw_validation_is_unaugmented_and_has_no_random_fallback(monkeypatch):
    from scripts.video_vam import train_smolexpert as trainer

    rgb = torch.arange(5 * 3 * 4 * 4, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(5, 3, 4, 4)
    item = trainer.OnlineVideoItem(
        rgb=rgb,
        state=torch.zeros(ACTION_DIM),
        action=torch.zeros(ACTION_HORIZON, ACTION_DIM),
        action_is_pad=torch.zeros(ACTION_HORIZON, dtype=torch.bool),
        sample_id="original-id",
        episode_index=32,
        frame_index=4820,
    )
    frames = []

    def extract(*, rgb_frames):
        frames.append(rgb_frames.clone())
        return SimpleNamespace(features=torch.zeros(1, 4, 16))

    monkeypatch.setattr(trainer, "augment_temporal_window_gpu", _unexpected)
    decoder = trainer.build_synthetic_tiny_decoder(input_channels=16)
    rng = torch.get_rng_state()
    trainer.evaluate_validation(
        decoder, [item], extractor=SimpleNamespace(extract=extract), device=torch.device("cpu"), num_steps=1
    )
    torch.testing.assert_close(frames[0], rgb.float().permute(1, 0, 2, 3).unsqueeze(0) / 255)
    assert torch.equal(torch.get_rng_state(), rng)
    assert decoder.training
    with pytest.raises(ValueError, match="random features are forbidden"):
        trainer.evaluate_validation(decoder, [item], device=torch.device("cpu"))
    assert decoder.training


@pytest.fixture
def saved_cosmos3_run(cosmos3_assets, monkeypatch):
    from lerobot.policies.vam import cosmos3_features
    from scripts.video_vam import train_smolexpert as trainer

    run = cosmos3_assets.root / "original_run"
    run.mkdir()
    normalizer = SmolVLANormalizer(
        state_mean=torch.full((ACTION_DIM,), 3.0),
        state_std=torch.ones(ACTION_DIM),
        action_mean=torch.full((ACTION_DIM,), 4.0),
        action_std=torch.ones(ACTION_DIM),
    )
    norm = run / "normalizer.safetensors"
    save_file(normalizer.state_dict(), str(norm))
    decoder = trainer.build_synthetic_tiny_decoder(input_channels=2048, normalizer=normalizer)
    checkpoints = {}
    for role in ("best", "last"):
        path = run / f"{role}.safetensors"
        save_file(decoder.state_dict(), str(path))
        checkpoints[role] = {
            "path": path.name,
            "sha256": trainer.sha256_file(path),
            "optimizer_step": 12500 if role == "best" else 22500,
        }
        with torch.no_grad():
            decoder.context_adapter.projection.weight.add_(0.1)
    saved = {k: str(v) if isinstance(v, Path) else v for k, v in vars(cosmos3_assets.args).items()}
    saved.update(
        {
            "seed": 42,
            "num_steps": 2,
            "batch_size": 1,
            "overwrite": True,
            "normalizer_path": "/must-not-use-external-normalizer",
            "output_dir": str(run),
            "augment": True,
        }
    )
    original_records = {}
    for split, replacement_path in cosmos3_assets.paths.items():
        original_payload = json.loads(replacement_path.read_text())
        original_payload["provenance"]["lora_sha256"] = "b" * 64
        original_dir = cosmos3_assets.root / f"original-{split}-cache"
        original_dir.mkdir()
        for entry in original_payload["entries"]:
            copyfile(replacement_path.parent / entry["safetensors"], original_dir / entry["safetensors"])
        original_path = original_dir / "manifest.json"
        original_path.write_text(json.dumps(original_payload))
        original_records[split] = {
            "source_path_hint": str(original_path),
            "manifest_sha256": trainer.sha256_file(original_path),
            "metadata": {key: value for key, value in original_payload.items() if key != "entries"},
        }
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "trainer": "scripts/video_vam/train_smolexpert.py",
        "code": {"commit": "original-commit", "dirty": True},
        "arguments": saved,
        "policy": {
            "type": "SmolExpertActionDecoder",
            "pretrained_source": saved["checkpoint_path"],
            "action_dim": ACTION_DIM,
            "action_horizon": ACTION_HORIZON,
            "num_steps": saved["num_steps"],
        },
        "normalizer": {
            "path": norm.name,
            "sha256": trainer.sha256_file(norm),
            "metadata": {"source_split": "train"},
        },
        "checkpoints": checkpoints,
        "datasets": original_records,
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    (run / "metrics.jsonl").write_text("original metrics\n")
    loaded = []

    def tiny_pretrained(cls, checkpoint, *, normalizer, device, **kwargs):
        loaded.append({"checkpoint": checkpoint, "normalizer": normalizer, "kwargs": kwargs})
        assert str(device) == "cpu"
        model = trainer.build_synthetic_tiny_decoder(
            input_channels=kwargs["input_channels"], normalizer=normalizer
        )
        model.num_steps = kwargs["num_steps"]
        return model

    # Exercise the real from_training_checkpoint/load_training_checkpoint strict loader,
    # mocking only the initial pretrained architecture construction/download.
    monkeypatch.setattr(trainer.SmolExpertActionDecoder, "from_pretrained", classmethod(tiny_pretrained))
    monkeypatch.setattr(cosmos3_features, "Cosmos3FeatureExtractor", _unexpected)
    monkeypatch.setattr(trainer.OnlineVideoDataset, "__init__", _unexpected)
    for name in (
        "compute_online_training_normalizer",
        "compute_training_normalizer",
        "clean_own_artifacts",
        "save_weights_artifact",
        "build_run_manifest",
        "build_scheduler",
    ):
        monkeypatch.setattr(trainer, name, _unexpected)
    monkeypatch.setattr(trainer.torch.optim, "AdamW", _unexpected)
    monkeypatch.setattr(trainer, "code_identity", lambda root: {"commit": "reeval-commit", "dirty": True})
    return SimpleNamespace(
        run=run,
        manifest=manifest,
        assets=cosmos3_assets,
        loaded=loaded,
        report=cosmos3_assets.root / "new-report.json",
    )


def _source_snapshot(run):
    from scripts.video_vam import train_smolexpert as trainer

    return {
        str(path.relative_to(run)): (trainer.sha256_file(path), path.stat().st_mtime_ns)
        for path in run.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("role", ["best", "last"])
def test_isolated_reeval_preserves_sources_hashes_args_and_seeds(saved_cosmos3_run, monkeypatch, role):
    from scripts.video_vam import train_smolexpert as trainer

    setup = saved_cosmos3_run
    before = _source_snapshot(setup.run)
    evaluate = trainer.evaluate_validation
    calls = []

    def spy(decoder, dataset, **kwargs):
        calls.append((kwargs["seed"], kwargs["num_steps"], [e["sample_id"] for e in dataset.entries]))
        assert kwargs.get("extractor") is None
        return evaluate(decoder, dataset, **kwargs)

    monkeypatch.setattr(trainer, "evaluate_validation", spy)
    assert (
        trainer.main(
            [
                "--eval-run-dir",
                str(setup.run),
                "--eval-report",
                str(setup.report),
                "--eval-checkpoint",
                role,
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert _source_snapshot(setup.run) == before
    report = json.loads(setup.report.read_text())
    assert report["checkpoint"]["role"] == role
    assert report["checkpoint"]["sha256"] == setup.manifest["checkpoints"][role]["sha256"]
    assert report["normalizer"]["sha256"] == setup.manifest["normalizer"]["sha256"]
    assert report["source_run"]["manifest_sha256"] == before["run_manifest.json"][0]
    assert report["current_backbone_identity"] == setup.assets.identity
    assert report["code"] == {"commit": "reeval-commit", "dirty": True}
    assert report["seeds"] == {"val": 42, "eval2": 1042}
    assert [c[:2] for c in calls] == [(42, 2), (1042, 2)]
    assert calls[0][2] == ["episode-0032-frame-004820"]
    assert calls[1][2] == ["episode-0090-frame-011174"]
    assert report["arguments"]["backbone_layer"] == 7
    assert report["arguments"]["backbone_prompt"] == "saved non-default prompt"
    assert "overwrite" not in report["arguments"]
    assert report["evaluation_augmented"] is False
    assert report["source_manifests"]["eval2"]["dataset_revision_verified"] is False
    assert report["source_manifests"]["val"]["sha256"] == trainer.sha256_file(setup.assets.paths["val"])
    verified = report["sample_verification"]
    for split in ("val", "eval2"):
        assert verified[split]["samples_verified"] == 1
        assert (
            verified[split]["original_manifest"]["sha256"]
            == setup.manifest["datasets"][split]["manifest_sha256"]
        )
        assert verified[split]["ordered_anchors_equal"] is True
        assert all(verified[split]["labels_equal"].values())
        assert verified[split]["original_metadata_and_artifact_hashes_verified"] is True
        assert verified[split]["revision_correction"]["changed"] is False
    assert report["original_selection_cache_matches_current_identity"] is False
    assert any("Historical online adapter digest" in limit for limit in report["limitations"])
    assert any("Original best" in limit for limit in report["limitations"])
    assert setup.loaded[0]["kwargs"] == {
        "num_steps": 2,
        "input_channels": 2048,
        "vlm_config_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    }
    assert torch.equal(setup.loaded[0]["normalizer"].state_mean, torch.full((ACTION_DIM,), 3.0))
    with pytest.raises(FileExistsError):
        trainer.main(
            ["--eval-run-dir", str(setup.run), "--eval-report", str(setup.report), "--device", "cpu"]
        )
    assert _source_snapshot(setup.run) == before


@pytest.mark.parametrize(
    ("artifact", "failure"),
    [
        ("best", "missing"),
        ("last", "missing"),
        ("best", "hash"),
        ("normalizer", "missing"),
        ("normalizer", "hash"),
    ],
)
def test_isolated_reeval_requires_recorded_own_artifacts(saved_cosmos3_run, artifact, failure):
    from scripts.video_vam import train_smolexpert as trainer

    setup = saved_cosmos3_run
    path = setup.run / f"{artifact}.safetensors"
    if failure == "missing":
        path.unlink()
    else:
        path.write_bytes(b"corrupted")
    before = _source_snapshot(setup.run)
    with pytest.raises((FileNotFoundError, ValueError)):
        trainer.main(
            [
                "--eval-run-dir",
                str(setup.run),
                "--eval-report",
                str(setup.report),
                "--eval-checkpoint",
                "last" if artifact == "last" else "best",
                "--device",
                "cpu",
            ]
        )
    assert not setup.loaded
    assert not setup.report.exists()
    assert _source_snapshot(setup.run) == before


def test_isolated_reeval_validates_explicit_cache_overrides(saved_cosmos3_run):
    from scripts.video_vam import train_smolexpert as trainer

    setup = saved_cosmos3_run
    manifest_path = setup.run / "run_manifest.json"
    setup.manifest["arguments"]["val_manifest"] = "/missing-original-cache"
    manifest_path.write_text(json.dumps(setup.manifest))
    before = _source_snapshot(setup.run)
    assert (
        trainer.main(
            [
                "--eval-run-dir",
                str(setup.run),
                "--eval-report",
                str(setup.report),
                "--val-manifest",
                str(setup.assets.paths["val"]),
                "--eval2-manifest",
                str(setup.assets.paths["eval2"]),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert _source_snapshot(setup.run) == before
    payload = json.loads(setup.assets.paths["eval2"].read_text())
    payload["provenance"]["lora_sha256"] = "0" * 64
    setup.assets.paths["eval2"].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="lora_sha256"):
        trainer.main(
            [
                "--eval-run-dir",
                str(setup.run),
                "--eval-report",
                str(setup.report.with_name("bad.json")),
                "--val-manifest",
                str(setup.assets.paths["val"]),
                "--device",
                "cpu",
            ]
        )
    assert not setup.report.with_name("bad.json").exists()
    assert len(setup.loaded) == 1


@pytest.mark.parametrize("unsafe", ["inside_run", "dangling_symlink"])
def test_isolated_reeval_rejects_unsafe_report_paths(saved_cosmos3_run, unsafe):
    from scripts.video_vam import train_smolexpert as trainer

    setup = saved_cosmos3_run
    report = setup.run / "new.json" if unsafe == "inside_run" else setup.report
    if unsafe == "dangling_symlink":
        report.symlink_to(setup.assets.root / "does-not-exist")
    before = _source_snapshot(setup.run)
    with pytest.raises((ValueError, FileExistsError)):
        trainer.main(["--eval-run-dir", str(setup.run), "--eval-report", str(report), "--device", "cpu"])
    assert not setup.loaded
    assert _source_snapshot(setup.run) == before


def test_eval_report_publication_is_atomic_no_clobber(tmp_path, monkeypatch):
    from scripts.video_vam import train_smolexpert as trainer

    report = tmp_path / "report.json"
    link = trainer.os.link

    def racing_link(source, destination):
        destination.write_text("other writer")
        link(source, destination)

    monkeypatch.setattr(trainer.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        trainer.write_new_eval_report(report, {"complete": True})
    assert report.read_text() == "other writer"
    assert list(tmp_path.iterdir()) == [report]


def test_eval_run_cli_requires_report_and_disallows_training_overrides(tmp_path):
    from scripts.video_vam import train_smolexpert as trainer

    base = [
        "--eval-run-dir",
        str(tmp_path),
        "--eval-report",
        str(tmp_path / "report.json"),
        "--device",
        "cpu",
    ]
    args = trainer.parse_args(base)
    assert args.eval_only and args.output_dir is None and args.eval_checkpoint == "best"
    for extra in (
        ["--overwrite"],
        ["--output-dir", "other"],
        ["--model-checkpoint", "other"],
        ["--seed", "0"],
        ["--dry-run"],
    ):
        with pytest.raises(SystemExit):
            trainer.parse_args(base + extra)
    with pytest.raises(SystemExit):
        trainer.parse_args(["--eval-run-dir", str(tmp_path)])


@pytest.mark.parametrize(("field", "value"), [("rank", "invalid"), ("alpha", "nan"), ("alpha", "0")])
def test_cosmos3_identity_rejects_malformed_adapter_metadata(cosmos3_assets, field, value):
    from scripts.video_vam import train_smolexpert as trainer

    metadata = {"rank": "2", "alpha": "4.0", field: value}
    save_file({"lora_A": torch.zeros(2, 2)}, str(cosmos3_assets.lora), metadata=metadata)
    with pytest.raises(ValueError, match="rank/alpha"):
        trainer.preflight_cosmos3_eval_caches(cosmos3_assets.args)


@pytest.mark.parametrize(
    "failure", ["seed_type", "missing_checkpoint_record", "recorded_feature_identity", "head_keys"]
)
def test_isolated_reeval_fails_closed_on_invalid_run(saved_cosmos3_run, failure):
    from scripts.video_vam import train_smolexpert as trainer

    setup = saved_cosmos3_run
    if failure == "seed_type":
        setup.manifest["arguments"]["seed"] = "42"
    elif failure == "missing_checkpoint_record":
        del setup.manifest["checkpoints"]["best"]
    elif failure == "recorded_feature_identity":
        setup.manifest["online_feature_identity"] = {**setup.assets.identity, "lora_sha256": "f" * 64}
    else:
        path = setup.run / "best.safetensors"
        save_file({"unexpected_key": torch.ones(1)}, str(path))
        setup.manifest["checkpoints"]["best"]["sha256"] = trainer.sha256_file(path)
    (setup.run / "run_manifest.json").write_text(json.dumps(setup.manifest))
    before = _source_snapshot(setup.run)
    with pytest.raises(ValueError):
        trainer.main(
            ["--eval-run-dir", str(setup.run), "--eval-report", str(setup.report), "--device", "cpu"]
        )
    assert not setup.report.exists()
    assert _source_snapshot(setup.run) == before
    assert len(setup.loaded) == (1 if failure == "head_keys" else 0)


def test_isolated_reeval_rejects_external_normalizer_symlink(saved_cosmos3_run):
    from scripts.video_vam import train_smolexpert as trainer

    setup = saved_cosmos3_run
    path = setup.run / "normalizer.safetensors"
    outside = setup.assets.root / "external-normalizer.safetensors"
    path.rename(outside)
    path.symlink_to(outside)
    before = _source_snapshot(setup.run)
    with pytest.raises(ValueError, match="inside the source run"):
        trainer.main(
            ["--eval-run-dir", str(setup.run), "--eval-report", str(setup.report), "--device", "cpu"]
        )
    assert _source_snapshot(setup.run) == before
    assert not setup.loaded
    assert not setup.report.exists()


@pytest.mark.parametrize(
    "change",
    [
        "sample_id",
        "episode_index",
        "frame_index",
        "window_indices",
        "repo_id",
        "state",
        "action",
        "action_is_pad",
    ],
)
def test_reeval_rejects_changed_benchmark_before_model(saved_cosmos3_run, change):
    from safetensors.torch import load_file

    from scripts.video_vam import train_smolexpert as trainer

    setup = saved_cosmos3_run
    path = setup.assets.paths["val"]
    payload = json.loads(path.read_text())
    entry = payload["entries"][0]
    if change == "sample_id":
        entry[change] += "-changed"
    elif change in ("episode_index", "frame_index"):
        entry[change] += 1
    elif change == "window_indices":
        entry[change][0] += 1
    elif change == "repo_id":
        payload["dataset"][change] = "different/noise-seed-identity"
    else:
        artifact = path.parent / entry["safetensors"]
        tensors = load_file(str(artifact))
        key = "target_action" if change == "action" and "target_action" in tensors else change
        if change == "action_is_pad":
            tensors[key][0] = True
        else:
            tensors[key].view(-1)[0] += 0.25
        save_file(tensors, str(artifact))
        entry["safetensors_sha256"] = trainer.sha256_file(artifact)
    path.write_text(json.dumps(payload))
    before = _source_snapshot(setup.run)
    with pytest.raises(
        ValueError,
        match="(ordered anchor mismatch|dataset identity changed|state mismatch|action mismatch|action_is_pad mismatch)",
    ):
        trainer.main(
            ["--eval-run-dir", str(setup.run), "--eval-report", str(setup.report), "--device", "cpu"]
        )
    assert not setup.loaded
    assert not setup.report.exists()
    assert _source_snapshot(setup.run) == before


def test_reeval_rejects_changed_anchor_order(saved_cosmos3_run):
    from scripts.video_vam import train_smolexpert as trainer

    setup = saved_cosmos3_run
    record = setup.manifest["datasets"]["val"]
    original_path = Path(record["source_path_hint"])
    for path in (original_path, setup.assets.paths["val"]):
        payload = json.loads(path.read_text())
        second = dict(payload["entries"][0])
        second["sample_id"] += "-second"
        second["frame_index"] += 20
        second["window_indices"] = [frame + 20 for frame in second["window_indices"]]
        payload["entries"].append(second)
        if path != original_path:
            payload["entries"].reverse()
        path.write_text(json.dumps(payload))
    record["manifest_sha256"] = trainer.sha256_file(original_path)
    (setup.run / "run_manifest.json").write_text(json.dumps(setup.manifest))
    with pytest.raises(ValueError, match="ordered anchor mismatch"):
        trainer.main(
            ["--eval-run-dir", str(setup.run), "--eval-report", str(setup.report), "--device", "cpu"]
        )
    assert not setup.loaded
    assert not setup.report.exists()


@pytest.mark.parametrize(
    "failure",
    [
        "missing_record",
        "missing_manifest",
        "manifest_hash",
        "metadata",
        "missing_artifact",
        "artifact_hash",
        "missing_artifact_hash",
    ],
)
def test_reeval_requires_verifiable_original_benchmark(saved_cosmos3_run, failure):
    from scripts.video_vam import train_smolexpert as trainer

    setup = saved_cosmos3_run
    record = setup.manifest["datasets"]["eval2"]
    path = Path(record["source_path_hint"])
    payload = json.loads(path.read_text())
    artifact = path.parent / payload["entries"][0]["safetensors"]
    if failure == "missing_record":
        del setup.manifest["datasets"]["eval2"]
    elif failure == "missing_manifest":
        path.unlink()
    elif failure == "manifest_hash":
        record["manifest_sha256"] = "0" * 64
    elif failure == "metadata":
        record["metadata"]["dataset"]["repo_id"] = "different/recorded-metadata"
    elif failure == "missing_artifact":
        artifact.unlink()
    elif failure == "artifact_hash":
        artifact.write_bytes(b"corrupt original cache")
    else:
        del payload["entries"][0]["safetensors_sha256"]
        path.write_text(json.dumps(payload))
        record["manifest_sha256"] = trainer.sha256_file(path)
    (setup.run / "run_manifest.json").write_text(json.dumps(setup.manifest))
    before = _source_snapshot(setup.run)
    with pytest.raises((ValueError, FileNotFoundError)):
        trainer.main(
            ["--eval-run-dir", str(setup.run), "--eval-report", str(setup.report), "--device", "cpu"]
        )
    assert not setup.loaded
    assert not setup.report.exists()
    assert _source_snapshot(setup.run) == before


def test_reeval_allows_only_revision_correction_and_reports_it(saved_cosmos3_run):
    from scripts.video_vam import train_smolexpert as trainer

    setup = saved_cosmos3_run
    path = setup.assets.paths["eval2"]
    payload = json.loads(path.read_text())
    old_revision = payload["dataset"]["revision"]
    payload["dataset"]["revision"] = "5d0325cc1412f4774223a0beb528958108814962"
    path.write_text(json.dumps(payload))
    assert (
        trainer.main(
            ["--eval-run-dir", str(setup.run), "--eval-report", str(setup.report), "--device", "cpu"]
        )
        == 0
    )
    report = json.loads(setup.report.read_text())
    verified = report["sample_verification"]["eval2"]
    assert verified["revision_correction"] == {
        "old": old_revision,
        "new": payload["dataset"]["revision"],
        "changed": True,
        "independently_verified": False,
    }
    assert verified["dataset_id"] == "Orellius/cube_out_of_box_v2"
    assert report["seeds"] == {"val": 42, "eval2": 1042}
    assert all(verified["labels_equal"].values())
    payload["dataset"]["extra_metadata"] = "not a revision correction"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="only revision metadata may differ"):
        trainer.main(
            [
                "--eval-run-dir",
                str(setup.run),
                "--eval-report",
                str(setup.report.with_name("bad.json")),
                "--device",
                "cpu",
            ]
        )
    assert len(setup.loaded) == 1


@pytest.mark.parametrize(
    "kind", ["empty", "metadata_only", "missing_transformer", "missing_vae", "empty_vae"]
)
def test_cosmos3_checkpoint_requires_relevant_nonempty_weights(cosmos3_assets, kind):
    from scripts.video_vam import train_smolexpert as trainer

    checkpoint = cosmos3_assets.checkpoint
    if kind == "empty":
        for path in checkpoint.rglob("*"):
            if path.is_file():
                path.unlink()
    elif kind == "metadata_only":
        for path in checkpoint.rglob("*.safetensors"):
            path.unlink()
    else:
        component = "transformer" if kind == "missing_transformer" else "vae"
        path = checkpoint / component / "diffusion_pytorch_model.safetensors"
        if kind == "empty_vae":
            path.write_bytes(b"")
        else:
            path.unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint lacks nonempty"):
        trainer.preflight_cosmos3_eval_caches(cosmos3_assets.args)


def test_online_eval_cache_identity_rejects_mismatched_lora_alpha_and_sigma(tmp_path: Path):
    from types import SimpleNamespace

    from scripts.video_vam.train_smolexpert import sha256_file, validate_online_eval_cache_identity

    adapter = tmp_path / "best_lora.safetensors"
    adapter.write_bytes(b"adapter")
    sha = sha256_file(adapter)
    manifest = tmp_path / "manifest.json"

    def write(provenance):
        manifest.write_text(json.dumps({"provenance": provenance, "entries": []}))

    args = SimpleNamespace(
        backbone_lora_weights=str(adapter),
        backbone_lora_rank=16,
        backbone_lora_alpha=16.0,
        online_backbone="cosmos2b",
        high_noise_sigma=80.0,
        backbone_layer=20,
    )
    good = {"lora_weights": {"sha256": sha, "rank": 16, "alpha": 16.0}, "high_noise_sigma": 80.0}
    write(good)
    validate_online_eval_cache_identity(manifest, args, source="val_manifest")

    write({**good, "lora_weights": {"sha256": sha, "rank": 16, "alpha": 32.0}})
    with pytest.raises(ValueError, match="alpha"):
        validate_online_eval_cache_identity(manifest, args, source="val_manifest")

    write({**good, "high_noise_sigma": 10.0})
    with pytest.raises(ValueError, match="high_noise_sigma"):
        validate_online_eval_cache_identity(manifest, args, source="val_manifest")

    write({**good, "lora_weights": {"sha256": "0" * 64, "rank": 16, "alpha": 16.0}})
    with pytest.raises(ValueError, match="sha256"):
        validate_online_eval_cache_identity(manifest, args, source="val_manifest")


def test_backbone_lora_alpha_is_resolved_from_adapter_and_conflicts_rejected(tmp_path: Path):
    from safetensors.torch import save_file

    from scripts.video_vam.train_smolexpert import parse_args

    adapter = tmp_path / "adapter.safetensors"
    save_file({"x": torch.zeros(1)}, str(adapter), metadata={"lora_rank": "16", "lora_alpha": "16.0"})
    base = [
        "--val-manifest",
        "v.json",
        "--backbone-lora-weights",
        str(adapter),
        "--output-dir",
        str(tmp_path / "o"),
    ]
    args = parse_args(base)
    assert (args.backbone_lora_rank, args.backbone_lora_alpha) == (16, 16.0)
    with pytest.raises(SystemExit):
        parse_args([*base, "--backbone-lora-alpha", "32.0"])


def test_select_metric_val_flow_loss_and_warmstart_seed(tmp_path: Path, monkeypatch):
    from scripts.video_vam import train_smolexpert as trainer

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

    init_ckpt = tmp_path / "init.safetensors"
    init_decoder = build_synthetic_tiny_decoder(input_channels=16)
    save_file(init_decoder.state_dict(), str(init_ckpt))

    original_eval = trainer.evaluate_validation
    eval_calls = 0
    # step 0 (seed): loss=0.05; step 1: loss=0.04 (best); step 2: loss=0.06 (bad 1);
    # step 3: loss=0.07 (bad 2 -> early stop even though val_rmse keeps dropping).
    schedule = [(10.0, 0.05), (5.0, 0.04), (4.0, 0.06), (3.0, 0.07), (2.0, 0.03)]

    def fake_evaluate(*args, **kwargs):
        nonlocal eval_calls
        metrics = original_eval(*args, **kwargs)
        rmse_val, loss_val = schedule[min(eval_calls, len(schedule) - 1)]
        eval_calls += 1
        metrics["val_rmse"] = rmse_val
        metrics["val_flow_loss"] = loss_val
        return metrics

    monkeypatch.setattr(trainer, "evaluate_validation", fake_evaluate)
    out_dir = tmp_path / "out_flow"
    ret = trainer.main(
        [
            "--train-manifest",
            str(train_m),
            "--val-manifest",
            str(val_m),
            "--output-dir",
            str(out_dir),
            "--model-checkpoint",
            str(init_ckpt),
            "--select-metric",
            "val_flow_loss",
            "--val-every",
            "1",
            "--patience",
            "2",
            "--min-steps",
            "0",
            "--max-steps",
            "5",
            "--dry-run",
            "--device",
            "cpu",
            "--no-wandb",
        ]
    )
    assert ret == 0
    best_meta = json.loads((out_dir / "best_metrics.json").read_text())
    summary = json.loads((out_dir / "training_summary.json").read_text())
    assert best_meta["step"] == 1
    assert best_meta["select_metric"] == "val_flow_loss"
    assert best_meta["val_flow_loss"] == pytest.approx(0.04)
    assert summary["best_step"] == 1
    assert summary["select_metric"] == "val_flow_loss"
    assert summary["best_select_metric"] == pytest.approx(0.04)
    assert summary["total_steps"] == 3


def test_cosmos3_nano_eval_identity_and_args(tmp_path: Path):
    from scripts.video_vam.train_smolexpert import cosmos3_eval_identity, parse_args

    ckpt = tmp_path / "cosmos3-nano"
    for comp in ("transformer", "vae"):
        (ckpt / comp).mkdir(parents=True)
        save_file({"w": torch.ones(1)}, str(ckpt / comp / "diffusion_pytorch_model-00001-of-00002.safetensors"))

    ident = cosmos3_eval_identity(
        checkpoint_path=ckpt,
        lora_checkpoint=None,
        lora_rank=16,
        lora_alpha=32.0,
        hidden_layer=36,
        prompt="grab cube",
        backbone_name="cosmos3-nano",
    )
    assert ident["backbone"] == "cosmos3-nano"
    assert ident["context_dim"] == 4096
    assert ident["hidden_layer"] == 36

    with pytest.raises(ValueError, match="Invalid Cosmos3 layer"):
        cosmos3_eval_identity(
            checkpoint_path=ckpt,
            lora_checkpoint=None,
            lora_rank=16,
            lora_alpha=32.0,
            hidden_layer=37,
            prompt="grab cube",
            backbone_name="cosmos3-nano",
        )

    args = parse_args(
        [
            "--online-backbone",
            "cosmos3_nano",
            "--backbone-layer",
            "18",
            "--select-metric",
            "val_flow_loss",
            "--val-manifest",
            "v.json",
        ]
    )
    assert args.online_backbone == "cosmos3_nano"
    assert args.backbone_layer == 18
    assert args.select_metric == "val_flow_loss"
