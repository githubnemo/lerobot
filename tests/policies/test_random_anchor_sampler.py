import pytest

from scripts.video_vam.train_cosmos_world2action import (
    Anchor,
    RandomAnchorSampler,
    validate_checkpoint_metadata,
)

BOUNDS = {0: (4, 106), 1: (111, 190), 2: (195, 299)}


def test_anchor_pool_counts_every_valid_frame_in_every_episode():
    sampler = RandomAnchorSampler(BOUNDS, seed=0)
    assert sampler.anchor_count == 103 + 80 + 105
    assert sampler.episodes == (0, 1, 2)


def test_anchors_stay_inside_their_episode_bounds():
    sampler = RandomAnchorSampler(BOUNDS, seed=7)
    for microbatch in range(200):
        for anchor in sampler.batch(microbatch, 4):
            first, last = BOUNDS[anchor.episode_index]
            assert first <= anchor.frame_index <= last


def test_anchor_stream_is_reproducible_for_a_given_seed():
    first = [RandomAnchorSampler(BOUNDS, seed=3).batch(index, 2) for index in range(32)]
    second = [RandomAnchorSampler(BOUNDS, seed=3).batch(index, 2) for index in range(32)]
    assert first == second
    assert first[0] != RandomAnchorSampler(BOUNDS, seed=4).batch(0, 2)


def test_replayed_anchors_match_the_live_stream_so_resume_reports_the_truth():
    sampler = RandomAnchorSampler(BOUNDS, seed=0)
    streamed = {
        (anchor.episode_index, anchor.frame_index)
        for microbatch in range(64)
        for anchor in sampler.batch(microbatch, 1)
    }
    assert sampler.drawn_anchors(64, 1) == streamed


def test_sampling_covers_far_more_scenes_than_a_stride_twenty_manifest_would():
    sampler = RandomAnchorSampler(BOUNDS, seed=0)
    drawn = sampler.drawn_anchors(2_000, 1)
    stride_twenty_windows = sum(len(range(first, last + 1, 20)) for first, last in BOUNDS.values())
    assert len(drawn) > 5 * stride_twenty_windows
    assert {episode for episode, _ in drawn} == set(BOUNDS)


def test_sampler_rejects_an_empty_pool_and_invalid_requests():
    with pytest.raises(ValueError):
        RandomAnchorSampler({}, seed=0)
    sampler = RandomAnchorSampler(BOUNDS, seed=0)
    with pytest.raises(ValueError):
        sampler.batch(-1, 1)
    with pytest.raises(ValueError):
        sampler.batch(0, 0)


def test_anchor_is_hashable_so_unique_scene_counting_works():
    assert len({Anchor(0, 4), Anchor(0, 4), Anchor(0, 5)}) == 2


def test_checkpoint_metadata_accepts_the_optional_normalizer_source_key_only():
    payload = {
        "schema_version": 1,
        "artifact": "world2action_training_checkpoint",
        "checkpoint_kind": "last",
        "status": "validation_trained_non_rollout",
        "rollout_readiness": False,
        "robot_ready": False,
        "step": 1,
        "best_step": 1,
        "best_val_fixed_flow_loss": 0.5,
        "manifest": "m",
        "manifest_sha256": "a",
        "split": "s",
        "split_sha256": "b",
        "normalizer": "n",
        "normalizer_metadata": "nm",
        "backbone": {},
        "decoder_config": {},
        "parameter_count": {},
        "hyperparameters": {},
        "optimizer": {},
        "scheduler": {},
        "autocast": "cuda_bfloat16",
        "frozen_context_dtype": "bfloat16",
        "action_padding_semantics": "x",
        "provenance": {},
        "resume_state": "r",
    }
    validate_checkpoint_metadata(payload)
    validate_checkpoint_metadata({**payload, "normalizer_source": {"derivation": "x"}})
    with pytest.raises(ValueError):
        validate_checkpoint_metadata({**payload, "unexpected_key": 1})
    with pytest.raises(ValueError):
        validate_checkpoint_metadata({key: value for key, value in payload.items() if key != "step"})
