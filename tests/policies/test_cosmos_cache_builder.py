import pytest

from scripts.video_vam.build_cosmos_feature_cache import _validate_args, parse_args


def test_builder_accepts_episode_zero_and_bounded_subset():
    args = parse_args(["--episodes", "0", "--max-samples", "4", "--frame-start", "4", "--frame-end", "8"])
    _validate_args(args)
    assert args.episodes == [0]
    assert args.max_samples == 4


def test_builder_rejects_ambiguous_resume_overwrite_and_ranges():
    args = parse_args(["--episodes", "0", "--resume", "--overwrite"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_args(args)
    args = parse_args(["--episodes", "0", "--frame-start", "8", "--frame-end", "8"])
    with pytest.raises(ValueError, match="greater"):
        _validate_args(args)
