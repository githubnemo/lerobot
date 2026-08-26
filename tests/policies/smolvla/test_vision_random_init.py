from types import SimpleNamespace

import torch
from torch import nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.smolvlm_with_expert import (
    SmolVLMWithExpertModel,
    _reinitialize_module,
)


class _TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.connector = nn.Linear(4, 4)


class _TinyVLMWithExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = _TinyVLM()
        self.lm_expert = nn.Linear(4, 4)
        self.set_requires_grad_calls = 0

    def get_vlm_model(self):
        return self.vlm

    def set_requires_grad(self):
        self.set_requires_grad_calls += 1
        self.vlm.vision_model.eval()
        self.vlm.vision_model.requires_grad_(False)


class _TinyPolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm_with_expert = _TinyVLMWithExpert()
        self.state_proj = nn.Linear(4, 4)
        self.action_in_proj = nn.Linear(4, 4)
        self.action_out_proj = nn.Linear(4, 4)
        self.action_time_mlp_in = nn.Linear(4, 4)
        self.action_time_mlp_out = nn.Linear(4, 4)


def _tiny_policy(seed: int = 0):
    torch.manual_seed(seed)
    policy = object.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SmolVLAConfig(device="cpu", randomize_vision=True)
    policy.model = _TinyPolicyModel()
    return policy


def _checkpointed_tiny_vlm():
    module = _TinyVLM()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(0.25)
    return module


class _TinyPretrainedVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self._is_hf_initialized = True
        self.projection._is_hf_initialized = True

    def init_weights(self):
        for submodule in self.modules():
            if not getattr(submodule, "_is_hf_initialized", False):
                reset_parameters = getattr(submodule, "reset_parameters", None)
                if callable(reset_parameters):
                    reset_parameters()


def test_load_time_checkpoint_loading_is_inert(monkeypatch):
    policy_model = _TinyPolicyModel()
    checkpoint_state = {name: value.detach().clone() for name, value in policy_model.state_dict().items()}
    loaded_policy = SimpleNamespace(
        config=SmolVLAConfig(device="cpu", vision_random_init_seed=123),
        model=policy_model,
    )

    def fake_from_pretrained(cls, *args, **kwargs):
        return loaded_policy

    monkeypatch.setattr(PreTrainedPolicy, "from_pretrained", classmethod(fake_from_pretrained))
    result = SmolVLAPolicy.from_pretrained("synthetic-checkpoint")

    assert result is loaded_policy
    for name, value in policy_model.state_dict().items():
        assert torch.equal(value, checkpoint_state[name]), name
    assert policy_model.vlm_with_expert.set_requires_grad_calls == 0


def test_apply_vision_randomization_is_deterministic_and_vision_only():
    first = _tiny_policy()
    second = _tiny_policy()
    before_nonvision = {
        name: value.detach().clone()
        for name, value in first.model.state_dict().items()
        if "vision_model" not in name
    }

    first_metadata = first.apply_vision_randomization(seed=17)
    second_metadata = second.apply_vision_randomization(seed=17)

    assert first_metadata["seed"] == 17
    assert first_metadata["before_sha256"] != first_metadata["after_sha256"]
    assert first_metadata["changed_tensor_count"] > 0
    assert first_metadata["after_sha256"] == second_metadata["after_sha256"]
    assert first.config.vision_randomization_seed == 17
    assert first.config.vision_randomization_applied is True
    assert first.model.vlm_with_expert.set_requires_grad_calls == 1
    assert first.model.vlm_with_expert.vlm.vision_model.training is False
    assert all(
        not parameter.requires_grad for parameter in first.model.vlm_with_expert.vlm.vision_model.parameters()
    )
    for name, value in first.model.state_dict().items():
        if "vision_model" not in name:
            assert torch.equal(value, before_nonvision[name]), name


def test_reinitialize_module_clears_transformers_initialized_markers():
    module = _TinyPretrainedVision()
    before = {name: value.detach().clone() for name, value in module.state_dict().items()}

    _reinitialize_module(module, seed=17)

    assert any(not torch.equal(value, before[name]) for name, value in module.state_dict().items())


def test_reinitialize_module_is_deterministic_and_preserves_cpu_rng_state():
    first = _checkpointed_tiny_vlm().vision_model
    second = _checkpointed_tiny_vlm().vision_model
    before = torch.random.get_rng_state()

    _reinitialize_module(first, seed=17)
    after = torch.random.get_rng_state()
    _reinitialize_module(second, seed=17)

    assert torch.equal(before, after)
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name]) for name in first.state_dict()
    )


def test_reinitialize_module_changes_with_seed_and_only_touches_vision():
    first = _checkpointed_tiny_vlm()
    second = _checkpointed_tiny_vlm()
    connector_before = {name: value.detach().clone() for name, value in first.connector.state_dict().items()}

    _reinitialize_module(first.vision_model, seed=17)
    _reinitialize_module(second.vision_model, seed=18)

    assert any(
        not torch.equal(first.vision_model.state_dict()[name], second.vision_model.state_dict()[name])
        for name in first.vision_model.state_dict()
    )
    assert all(
        torch.equal(value, connector_before[name]) for name, value in first.connector.state_dict().items()
    )


def test_normal_freeze_recipe_keeps_randomized_vision_frozen_and_eval():
    model = object.__new__(SmolVLMWithExpertModel)
    nn.Module.__init__(model)
    model.vlm = nn.Module()
    model.vlm.model = nn.Module()
    model.vlm.model.vision_model = nn.Linear(4, 4)
    model.vlm.model.text_model = nn.Linear(4, 4)
    model.lm_expert = nn.Linear(4, 4)
    model.freeze_vision_encoder = True
    model.train_expert_only = True

    model.set_requires_grad()

    assert model.vlm.model.vision_model.training is False
    assert all(not parameter.requires_grad for parameter in model.vlm.model.vision_model.parameters())
    assert all(not parameter.requires_grad for parameter in model.vlm.parameters())
