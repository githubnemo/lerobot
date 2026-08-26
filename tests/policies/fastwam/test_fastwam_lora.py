from __future__ import annotations

import torch
from torch import nn

from lerobot.policies.fastwam.fastwam_lora import (
    FASTWAM_VIDEO_LAYER_COUNT,
    FASTWAM_VIDEO_LORA_TARGETS,
    LoRALinear,
    inject_lora,
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
    save_lora_state_dict,
)


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(4, 4)
        self.k = nn.Linear(4, 4)
        self.v = nn.Linear(4, 4)
        self.o = nn.Linear(4, 4)


class TinyVideoBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = TinyAttention()
        self.cross_attn = TinyAttention()
        self.ffn = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


class TinyMoTLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleDict({"video": TinyVideoBlock(), "action": nn.Linear(4, 4)})


class TinyMoT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # These stand in for the registered, non-block video/action expert weights.
        self.mixtures = nn.ModuleDict({"video": nn.Linear(4, 4), "action": nn.Linear(4, 4)})
        self.layers = nn.ModuleList([TinyMoTLayer() for _ in range(FASTWAM_VIDEO_LAYER_COUNT)])


class TinyFastWAM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mot = TinyMoT()


def test_fastwam_lora_covers_exact_video_targets() -> None:
    model = TinyFastWAM()
    wrapped = inject_lora(model, rank=2, alpha=4)

    assert len(FASTWAM_VIDEO_LORA_TARGETS) == FASTWAM_VIDEO_LAYER_COUNT * 10 == 300
    assert wrapped == FASTWAM_VIDEO_LORA_TARGETS
    assert all(isinstance(model.get_submodule(name), LoRALinear) for name in FASTWAM_VIDEO_LORA_TARGETS)


def test_fastwam_lora_delegates_linear_attributes() -> None:
    base = nn.Linear(4, 7, bias=True)
    wrapped = LoRALinear(base, rank=2)

    assert wrapped.weight is base.weight
    assert wrapped.bias is base.bias
    assert wrapped.in_features == base.in_features
    assert wrapped.out_features == base.out_features


def test_fastwam_lora_freezes_video_and_keeps_action_trainable() -> None:
    model = TinyFastWAM()
    inject_lora(model, rank=2)

    video_base_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if (name.startswith("mot.mixtures.video.") or ".blocks.video." in name)
        and not name.endswith(("lora_A", "lora_B"))
    ]
    action_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("mot.mixtures.action.") or ".blocks.action." in name
    ]
    adapters = lora_parameters(model)

    assert video_base_parameters
    assert action_parameters
    assert all(not parameter.requires_grad for parameter in video_base_parameters)
    assert all(parameter.requires_grad and parameter.dtype == torch.float32 for parameter in adapters)
    assert all(parameter.requires_grad for parameter in action_parameters)


def test_fastwam_lora_save_load_roundtrip(tmp_path) -> None:
    source = TinyFastWAM()
    inject_lora(source, rank=2, alpha=3)
    with torch.no_grad():
        for index, parameter in enumerate(lora_parameters(source)):
            parameter.fill_(index + 1)

    path = tmp_path / "fastwam-lora.safetensors"
    save_lora_state_dict(source, path)

    target = TinyFastWAM()
    inject_lora(target, rank=2, alpha=3)
    load_lora_state_dict(target, path)

    source_state = lora_state_dict(source)
    target_state = lora_state_dict(target)
    assert source_state.keys() == target_state.keys()
    assert all(torch.equal(source_state[name], target_state[name]) for name in source_state)
