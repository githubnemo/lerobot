import copy
import json
from pathlib import Path

import pytest

from lerobot.policies.vam.cosmos_cache_dataset import (
    CACHE_SCHEMA_VERSION,
    CosmosFeatureCacheManifestError,
    load_cache_manifest,
    manifest_sha256,
    write_cache_manifest,
)
from scripts.video_vam.subset_cosmos_cache_manifest import (
    evenly_spaced_indices,
    subset_manifest,
)


def make_source_manifest(tmp_path: Path, count: int = 64) -> tuple[Path, dict]:
    entries = []
    for index in range(count):
        frame = index + 4
        stem = f"episode-0000-frame-{frame:06d}"
        tensor = tmp_path / f"{stem}.safetensors"
        sidecar = tmp_path / f"{stem}.json"
        tensor.write_bytes(f"tensor-{index}".encode())
        sidecar.write_text(json.dumps({"index": index}))
        entries.append(
            {
                "sample_id": stem,
                "episode_index": 0,
                "frame_index": frame,
                "window_indices": list(range(frame - 4, frame + 1)),
                "noise_seed": index + 100,
                "safetensors": tensor.name,
                "sidecar": sidecar.name,
                "safetensors_sha256": "a" * 64,
                "sidecar_sha256": "b" * 64,
                "bytes": tensor.stat().st_size + sidecar.stat().st_size,
            }
        )
    payload = {
        "schema_version": 1,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset": {"repo_id": "dataset", "revision": "revision"},
        "subset": {"episodes": [0], "frame_start": None, "frame_end": None, "max_samples": count},
        "provenance": {"builder": "test", "status": "diagnostic_only_non_rollout"},
        "global_seed": 17,
        "entries": entries,
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "runtime": {},
    }
    manifest = tmp_path / "manifest.json"
    write_cache_manifest(payload, manifest)
    return manifest, payload


def test_evenly_spaced_64_to_8_spans_endpoints():
    assert evenly_spaced_indices(64, 8) == (0, 9, 18, 27, 36, 45, 54, 63)


def test_evenly_spaced_count_one_is_first_and_unique():
    assert evenly_spaced_indices(64, 1) == (0,)
    for entry_count in range(1, 65):
        for selected_count in range(1, entry_count + 1):
            indices = evenly_spaced_indices(entry_count, selected_count)
            assert len(indices) == len(set(indices))
            assert indices == tuple(sorted(indices))


def test_subset_preserves_entries_and_records_parent_provenance(tmp_path):
    source, payload = make_source_manifest(tmp_path)
    output = tmp_path / "manifest-even8.json"

    subset_manifest(source, output, 8)

    result = load_cache_manifest(output)
    expected_indices = evenly_spaced_indices(64, 8)
    assert result.payload["entries"] == [payload["entries"][index] for index in expected_indices]
    assert result.payload["dataset"] == payload["dataset"]
    assert result.global_seed == payload["global_seed"]
    assert result.payload["subset"]["max_samples"] == 8
    assert result.payload["total_bytes"] == sum(entry.bytes for entry in result.entries)
    provenance = result.payload["provenance"]
    assert provenance["parent_manifest_filename"] == source.name
    assert provenance["parent_manifest_sha256"] == manifest_sha256(source)
    assert provenance["selection_strategy"] == "evenly-spaced"
    assert provenance["source_entry_count"] == 64
    assert provenance["selected_entry_count"] == 8
    assert provenance["selected_sample_ids"] == [entry.sample_id for entry in result.entries]
    assert all((tmp_path / entry.safetensors).is_file() for entry in result.entries)


def test_subset_rejects_count_larger_than_source(tmp_path):
    source, _ = make_source_manifest(tmp_path, count=4)
    with pytest.raises(ValueError, match="cannot exceed"):
        subset_manifest(source, tmp_path / "too-many.json", 5)


def test_subset_rejects_output_outside_source_root(tmp_path):
    source, _ = make_source_manifest(tmp_path)
    with pytest.raises(ValueError, match="same directory"):
        subset_manifest(source, tmp_path.parent / "manifest-even8.json", 8)


def test_subset_requires_explicit_overwrite(tmp_path):
    source, _ = make_source_manifest(tmp_path)
    output = tmp_path / "manifest-even8.json"
    subset_manifest(source, output, 8)
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        subset_manifest(source, output, 1)
    assert output.read_bytes() == original
    subset_manifest(source, output, 1, overwrite=True)
    assert len(load_cache_manifest(output).entries) == 1


def test_subset_strictly_inherits_source_manifest_validation(tmp_path):
    source, payload = make_source_manifest(tmp_path)
    tampered = copy.deepcopy(payload)
    tampered["subset"]["episodes"] = [1]
    source.write_text(json.dumps(tampered))

    with pytest.raises(CosmosFeatureCacheManifestError, match="outside the selected episodes"):
        subset_manifest(source, tmp_path / "manifest-even8.json", 8)
