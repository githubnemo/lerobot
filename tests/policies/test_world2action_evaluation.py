import json
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from lerobot.policies.vam.world2action import World2ActionConfig, World2ActionDecoder
from scripts.video_vam.evaluate_cosmos_world2action_cache import (
    _atomic_json,
    _masked_physical_mse,
    aggregate_results,
    evaluate_entries,
)


class FakeDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, **kwargs):
        return torch.zeros_like(kwargs["xt_B_HA_A"]) + self.scale


@dataclass
class Item:
    sample_id: str
    state: torch.Tensor
    target_action: torch.Tensor
    context: torch.Tensor
    action_is_pad: torch.Tensor


def make_decoder():
    return World2ActionDecoder(World2ActionConfig(device="cpu"), denoiser=FakeDenoiser())


def make_items():
    context = torch.zeros(1, 2, 2048, dtype=torch.bfloat16)
    return [
        Item(
            "episode-0-frame-0",
            torch.zeros(1, 1, 6),
            torch.ones(1, 30, 6),
            context,
            torch.cat((torch.zeros(1, 15, dtype=torch.bool), torch.ones(1, 15, dtype=torch.bool)), dim=1),
        ),
        Item(
            "episode-0-frame-1",
            torch.zeros(1, 1, 6),
            torch.full((1, 30, 6), 2.0),
            context,
            torch.zeros(1, 30, dtype=torch.bool),
        ),
    ]


def test_evaluation_aggregates_all_entries_and_masks_padding():
    results = evaluate_entries(make_decoder(), make_items(), device=torch.device("cpu"), seed=17)

    assert [result["sample_id"] for result in results] == ["episode-0-frame-0", "episode-0-frame-1"]
    aggregate = aggregate_results(results)
    assert aggregate["count"] == 2
    assert aggregate["sampled_action_physical_mse"]["mean"] == pytest.approx(
        sum(result["sampled_action_physical_mse"] for result in results) / 2
    )
    items = make_items()
    assert _masked_physical_mse(
        torch.zeros(1, 30, 6), items[0].target_action, items[0].action_is_pad
    ).item() == pytest.approx(1.0)
    assert aggregate["sampled_action_physical_rmse"] == pytest.approx(
        aggregate["sampled_action_physical_mse"]["mean"] ** 0.5
    )


def test_evaluation_seed_is_deterministic():
    first = evaluate_entries(make_decoder(), make_items(), device=torch.device("cpu"), seed=123)
    second = evaluate_entries(make_decoder(), make_items(), device=torch.device("cpu"), seed=123)

    assert first == second
    assert [result["evaluation_seed"] for result in first] == [123, 124]


def test_evaluation_json_requires_explicit_overwrite(tmp_path):
    output = tmp_path / "evaluation.json"
    payload = {"status": "diagnostic_only_non_rollout"}
    _atomic_json(output, payload, overwrite=False)
    with pytest.raises(FileExistsError):
        _atomic_json(output, payload, overwrite=False)
    _atomic_json(output, payload, overwrite=True)
    assert json.loads(output.read_text()) == payload
