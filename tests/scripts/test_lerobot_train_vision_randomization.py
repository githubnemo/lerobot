from types import SimpleNamespace

import pytest

from lerobot.scripts.lerobot_train import _apply_policy_randomization_hook


class _FakePolicy:
    def __init__(self):
        self.seeds = []

    def apply_vision_randomization(self, seed: int):
        self.seeds.append(seed)


def test_training_hook_uses_cfg_seed_only_when_enabled():
    cfg = SimpleNamespace(
        policy=SimpleNamespace(type="smolvla", randomize_vision=True),
        resume=False,
        seed=321,
    )
    policy = _FakePolicy()

    _apply_policy_randomization_hook(cfg, policy)

    assert policy.seeds == [321]


def test_training_hook_rejects_randomization_on_resume():
    cfg = SimpleNamespace(
        policy=SimpleNamespace(type="smolvla", randomize_vision=True),
        resume=True,
        seed=321,
    )

    with pytest.raises(ValueError, match="cannot be used with --resume"):
        _apply_policy_randomization_hook(cfg, _FakePolicy())


def test_training_hook_skips_disabled_flag():
    cfg = SimpleNamespace(
        policy=SimpleNamespace(type="smolvla", randomize_vision=False),
        resume=False,
        seed=321,
    )
    policy = _FakePolicy()

    _apply_policy_randomization_hook(cfg, policy)

    assert policy.seeds == []
