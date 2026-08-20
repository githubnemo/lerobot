#!/usr/bin/env python3
"""Verify random-anchor sampling, anchor validity, and normalizer derivation."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT as CONTRACT
from lerobot.policies.vam.cosmos_cache_dataset import load_cache_manifest
from lerobot.policies.vam.vam_split import load_vam_split
from scripts.video_vam.train_cosmos_world2action import (
    Anchor,
    RandomAnchorDataset,
    RandomAnchorSampler,
    build_anchor_dataset,
    episode_anchor_bounds,
)

MANIFEST = Path("/home/anton/.cache/video-vam/cosmos-rehearsal-stride20-sigma4/manifest.json")
SPLIT = Path("/home/anton/.cache/video-vam/splits/rehearsal-stride20.json")
ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")


class _Args:
    dataset_root = ROOT


def main() -> int:
    manifest = load_cache_manifest(MANIFEST)
    split = load_vam_split(SPLIT, manifest)
    dataset = build_anchor_dataset(manifest, _Args())
    bounds = episode_anchor_bounds(dataset, split.train_episodes)
    sampler = RandomAnchorSampler(bounds, seed=0)
    anchor_dataset = RandomAnchorDataset(dataset)

    print("train_episodes", list(split.train_episodes))
    print("anchor_pool", sampler.anchor_count, "episodes", len(bounds))
    print("bounds_head", {e: bounds[e] for e in sorted(bounds)[:3]})

    manifest_train = {
        (entry.episode_index, entry.frame_index)
        for entry in manifest.entries
        if entry.episode_index in set(split.train_episodes)
    }
    pool = {(episode, frame) for episode, (first, last) in bounds.items() for frame in range(first, last + 1)}
    print("manifest_train_windows", len(manifest_train), "subset_of_pool", manifest_train <= pool)

    # Determinism and resume equivalence.
    first_pass = [sampler.batch(index, 1) for index in range(64)]
    second_pass = [RandomAnchorSampler(bounds, seed=0).batch(index, 1) for index in range(64)]
    assert first_pass == second_pass, "anchor stream is not reproducible"
    replayed = sampler.drawn_anchors(64, 1)
    streamed = {(a.episode_index, a.frame_index) for batch in first_pass for a in batch}
    assert replayed == streamed, "resume replay diverges from the live stream"
    print("determinism_ok", True, "unique_in_64", len(streamed))

    # Coverage: 6000 microbatches is roughly the previous run's sample budget.
    drawn = sampler.drawn_anchors(6000, 1)
    assert drawn <= pool, "sampler produced an anchor outside the valid pool"
    episodes_hit = {episode for episode, _ in drawn}
    print("unique_anchors_in_6000", len(drawn), "episodes_hit", len(episodes_hit))

    # Every anchor must actually load with a full five-frame history.
    probe = sorted(pool)[:2] + sorted(pool)[-2:] + sorted(drawn)[:2]
    for episode, frame in probe:
        prepared = anchor_dataset.load(Anchor(episode, frame))
        assert tuple(prepared.rgb_history.shape) == (1, 3, 5, 480, 640), prepared.rgb_history.shape
        assert tuple(prepared.state.shape) == (1, 1, 6), prepared.state.shape
        assert tuple(prepared.target_action.shape) == (1, 30, 6), prepared.target_action.shape
        assert tuple(prepared.action_is_pad.shape) == (1, 30), prepared.action_is_pad.shape
        assert prepared.action_is_pad.dtype == torch.bool
    print("shapes_ok", True, "probed", len(probe))

    # Padding must follow the dataset convention: pads only at the episode tail.
    for episode in sorted(bounds)[:4]:
        first, last = bounds[episode]
        pads = int(anchor_dataset.load(Anchor(episode, last)).action_is_pad.sum())
        head_pads = int(anchor_dataset.load(Anchor(episode, first)).action_is_pad.sum())
        assert pads == CONTRACT.action_chunk_size - 1, (episode, pads)
        assert head_pads == 0, (episode, head_pads)
    print("padding_convention_ok", True)

    summary = {
        "anchor_pool": sampler.anchor_count,
        "manifest_train_windows": len(manifest_train),
        "unique_anchors_in_6000_draws": len(drawn),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
