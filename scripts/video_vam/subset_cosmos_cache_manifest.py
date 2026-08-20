#!/usr/bin/env python3
"""Select a deterministic diagnostic subset of an existing Cosmos cache manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from lerobot.policies.vam.cosmos_cache_dataset import (
    load_cache_manifest,
    manifest_sha256,
    write_cache_manifest,
)

_STRATEGY = "evenly-spaced"


def _positive_count(value: str) -> int:
    count = int(value)
    if count <= 0:
        raise argparse.ArgumentTypeError("count must be positive")
    return count


def evenly_spaced_indices(entry_count: int, selected_count: int) -> tuple[int, ...]:
    """Return unique indices spanning both endpoints when selecting more than one."""
    if entry_count <= 0:
        raise ValueError("entry_count must be positive")
    if selected_count <= 0:
        raise ValueError("selected_count must be positive")
    if selected_count > entry_count:
        raise ValueError("selected_count cannot exceed entry_count")
    if selected_count == 1:
        return (0,)
    # Integer floor interpolation over [0, entry_count - 1]; entry_count >= selected_count
    # makes adjacent numerators sufficiently separated to avoid duplicate indices.
    return tuple(index * (entry_count - 1) // (selected_count - 1) for index in range(selected_count))


def subset_manifest(
    source_manifest: Path,
    output_manifest: Path,
    selected_count: int,
    *,
    strategy: str = _STRATEGY,
    overwrite: bool = False,
) -> Path:
    """Write a diagnostic subset manifest without touching referenced artifacts."""
    if strategy != _STRATEGY:
        raise ValueError(f"unsupported selection strategy: {strategy!r}")
    source_manifest = source_manifest.expanduser().resolve()
    output_manifest = output_manifest.expanduser().resolve()
    if source_manifest.parent != output_manifest.parent:
        raise ValueError("output manifest must be in the same directory as the source manifest")

    source = load_cache_manifest(source_manifest)
    indices = evenly_spaced_indices(len(source.entries), selected_count)
    payload = source.payload
    selected_entries = [dict(payload["entries"][index]) for index in indices]
    provenance: dict[str, Any] = dict(payload["provenance"])
    provenance.update(
        {
            "parent_manifest_filename": source_manifest.name,
            "parent_manifest_sha256": manifest_sha256(source_manifest),
            "selection_strategy": strategy,
            "source_entry_count": len(source.entries),
            "selected_entry_count": len(selected_entries),
            "selected_sample_ids": [entry["sample_id"] for entry in selected_entries],
            "status": "diagnostic_only_non_rollout",
        }
    )
    subset = dict(payload["subset"])
    subset["max_samples"] = selected_count
    subset_payload = dict(payload)
    subset_payload["subset"] = subset
    selection = dict(
        provenance.get("selection", {"stride": int(subset.get("stride", 1)), "ordered_pairs": []})
    )
    selection["ordered_pairs"] = [
        [entry["episode_index"], entry["frame_index"]] for entry in selected_entries
    ]
    selection["stride"] = int(subset.get("stride", 1))
    provenance["selection"] = selection
    subset_payload["provenance"] = provenance
    subset_payload["entries"] = selected_entries
    subset_payload["total_bytes"] = sum(int(entry["bytes"]) for entry in selected_entries)
    return write_cache_manifest(subset_payload, output_manifest, overwrite=overwrite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--count", required=True, type=_positive_count)
    parser.add_argument("--strategy", choices=(_STRATEGY,), default=_STRATEGY)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = subset_manifest(
        args.source_manifest,
        args.output_manifest,
        args.count,
        strategy=args.strategy,
        overwrite=args.overwrite,
    )
    print(f"manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
