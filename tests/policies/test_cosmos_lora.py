import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.cosmos_lora import (
    FULL_CLIP_LENGTH,
    LoRALinear,
    action_context_for_arm,
    capture_prediction_and_layer20,
    full_clip_delta_timestamps,
    full_clip_window_indices,
    inject_lora,
    latent_valid_mask,
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
    merge_lora_into_base,
    prepare_full_clip_sample,
    rectified_flow_target,
    rectified_flow_video_loss,
    sample_upstream_sigma,
    save_lora_state_dict,
)
from lerobot.policies.vam.world2action import World2ActionConfig, World2ActionDecoder


class _TinyAttention(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.output_proj = nn.Linear(width, width, bias=False)
        self.not_a_projection = nn.Linear(width, width, bias=False)


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(8, 16, bias=False)
        self.layer2 = nn.Linear(16, 8, bias=False)


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _TinyAttention()
        self.cross_attn = _TinyAttention()
        self.mlp = _TinyMLP()


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock() for _ in range(28)])
        self.head = nn.Linear(8, 8, bias=False)


class _TinyCaptureBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(28)])
        self.final = nn.Linear(2, 2)

    def forward(self, x_B_C_T_H_W, **kwargs):  # noqa: N803
        del kwargs
        value = x_B_C_T_H_W
        for block in self.blocks:
            value = block(value)
        return self.final(value)


class _TinyActionDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))

    def forward(  # noqa: N803
        self,
        state_B_HO_O,  # noqa: N803
        xt_B_HA_A,  # noqa: N803
        timesteps_B_T,  # noqa: N803
        context_timesteps_B_1,  # noqa: N803
        crossattn_emb,
        obs_dropout,  # noqa: N803
    ):
        del state_B_HO_O, timesteps_B_T, context_timesteps_B_1, obs_dropout
        value = crossattn_emb.mean(dim=(1, 2), keepdim=True) * self.anchor
        return value.expand(xt_B_HA_A.shape[0], 30, 6)


class _TinyScheduler:
    alpha = 1.0
    beta = 1.0
    num_denoising_steps = 10


def test_capture_runs_one_forward_and_returns_pre_detach_layer_20():
    model = _TinyCaptureBackbone()
    prediction, hidden = capture_prediction_and_layer20(model, x_B_C_T_H_W=torch.ones(1, 2, 2))
    assert prediction.shape == (1, 2, 2)
    assert hidden.shape == (1, 2, 2)
    assert hidden.requires_grad


def test_decoder_context_detach_switch_controls_action_gradients():
    decoder = World2ActionDecoder(
        World2ActionConfig(device="cpu", dtype=torch.float32),
        denoiser=_TinyActionDenoiser(),
        scheduler=_TinyScheduler(),
    )
    state = torch.zeros((1, 1, 6), dtype=torch.float32)
    action = torch.zeros((1, 30, 6), dtype=torch.float32)
    context = torch.randn((1, 4, 2048), dtype=torch.bfloat16, requires_grad=True)
    loss = decoder.flow_matching_loss(
        state, action, context, t=torch.tensor([0.5]), epsilon=torch.zeros_like(action), detach_context=True
    )
    loss.backward()
    assert context.grad is None

    context = torch.randn((1, 4, 2048), dtype=torch.bfloat16, requires_grad=True)
    loss = decoder.flow_matching_loss(
        state, action, context, t=torch.tensor([0.5]), epsilon=torch.zeros_like(action), detach_context=False
    )
    loss.backward()
    assert context.grad is not None


def test_lora_injection_targets_only_all_28_attention_and_mlp_projections(tmp_path):
    model = _TinyBackbone()
    wrapped = inject_lora(model, rank=2)
    assert len(wrapped) == 28 * 10
    assert all(
        isinstance(module, LoRALinear)
        for name, module in model.named_modules()
        if name.endswith(("q_proj", "k_proj", "v_proj", "output_proj", "layer1", "layer2"))
    )
    assert isinstance(model.head, nn.Linear)
    assert all(parameter.dtype == torch.float32 for parameter in lora_parameters(model))
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.endswith(("lora_A", "lora_B"))
    )

    path = tmp_path / "adapter.safetensors"
    before = lora_state_dict(model)
    save_lora_state_dict(model, path)
    with torch.no_grad():
        for parameter in lora_parameters(model):
            parameter.add_(1.0)
    load_lora_state_dict(model, path)
    assert all(
        torch.equal(parameter, before[name]) for name, parameter in model.named_parameters() if name in before
    )


def test_frozen_base_weights_do_not_receive_gradients():
    base = nn.Linear(3, 4, bias=False)
    module = LoRALinear(base, rank=2)
    output = module(torch.ones(2, 3)).sum()
    output.backward()
    assert base.weight.grad is None
    assert module.lora_A.grad is not None
    assert module.lora_B.grad is not None


def test_merge_lora_into_base_matches_wrapped_linear():
    model = _TinyBackbone()
    inject_lora(model, rank=2, alpha=4)
    wrapped = model.blocks[0].self_attn.q_proj
    inputs = torch.randn(3, 8)
    with torch.no_grad():
        wrapped.lora_A.copy_(torch.arange(16, dtype=torch.float32).reshape(2, 8) / 10)
        wrapped.lora_B.copy_(torch.arange(16, dtype=torch.float32).reshape(8, 2) / 20)
    expected = wrapped.base_layer.weight.detach() + 2 * (wrapped.lora_B @ wrapped.lora_A)
    before = wrapped(inputs)
    merged = merge_lora_into_base(model)
    fused = model.blocks[0].self_attn.q_proj
    assert len(merged) == 28 * 10
    assert isinstance(fused, nn.Linear) and not isinstance(fused, LoRALinear)
    assert torch.allclose(fused.weight, expected)
    assert torch.allclose(fused(inputs), before)
    assert not any(isinstance(module, LoRALinear) for module in model.modules())


def test_arm_b_detaches_action_context_but_arm_a_does_not():
    features = torch.randn(1, 4, 8, requires_grad=True)
    assert action_context_for_arm(features, action_grad_to_backbone=True) is features
    detached = action_context_for_arm(features, action_grad_to_backbone=False)
    assert detached is not features
    assert not detached.requires_grad


def test_sigma_sampler_matches_upstream_mixture_and_is_reproducible():
    first_generator = torch.Generator().manual_seed(17)
    second_generator = torch.Generator().manual_seed(17)
    first = sample_upstream_sigma(20_000, generator=first_generator)
    second = sample_upstream_sigma(20_000, generator=second_generator)
    assert torch.equal(first, second)
    tail = first >= 200.0
    assert 0.02 < tail.float().mean().item() < 0.08
    assert torch.all(first > 0)
    assert torch.all(first[tail] >= 200.0)


def test_rectified_flow_target_and_video_loss_use_velocity_parametrization():
    clean = torch.zeros((1, 2, 4, 1, 1))
    noise = torch.tensor([[[[[2.0]], [[3.0]], [[4.0]], [[5.0]]], [[[7.0]], [[8.0]], [[9.0]], [[10.0]]]]])
    target = rectified_flow_target(clean, noise)
    assert torch.equal(target, noise - clean)
    prediction = target.clone()
    valid = torch.ones((1, 4), dtype=torch.bool)
    assert rectified_flow_video_loss(prediction, clean, noise, torch.tensor([4.0]), valid) == 0
    prediction[:, :, 2:] += 1.0
    assert rectified_flow_video_loss(prediction, clean, noise, torch.tensor([4.0]), valid) > 0


def test_full_clip_contract_extends_only_camera_and_masks_repeat_last_padding():
    config = CUBE_OUT_OF_BOX_CONTRACT
    camera = torch.zeros((FULL_CLIP_LENGTH, 3, 480, 640), dtype=torch.uint8)
    camera_pad = torch.zeros(FULL_CLIP_LENGTH, dtype=torch.bool)
    camera_pad[-7:] = True
    sample = {
        config.camera_key: camera,
        f"{config.camera_key}_is_pad": camera_pad,
        config.state_key: torch.zeros(config.sample_state_shape),
        config.action_key: torch.zeros(config.sample_action_shape),
        f"{config.action_key}_is_pad": torch.zeros(config.action_chunk_size, dtype=torch.bool),
        "episode_index": torch.tensor(3),
    }
    prepared = prepare_full_clip_sample(sample, frame_index=20)
    assert prepared.rgb_clip.shape == (1, 3, FULL_CLIP_LENGTH, 480, 640)
    assert prepared.window_indices == tuple(range(16, 77))
    assert full_clip_window_indices(20) == tuple(range(16, 77))
    valid = latent_valid_mask(prepared.camera_is_pad)
    assert valid.shape == (1, 16)
    assert valid[0, :14].all()
    assert not valid[0, -1]
    timestamps = full_clip_delta_timestamps(config)
    assert len(timestamps[config.camera_key]) == FULL_CLIP_LENGTH
    assert timestamps[config.camera_key][0] == -0.4
    assert timestamps[config.camera_key][-1] == 5.6
    assert config.delta_timestamps()[config.camera_key] == [-0.4, -0.3, -0.2, -0.1, 0.0]


def test_lora_roundtrip_across_activation_checkpoint_wrapper(tmp_path):
    source = _TinyBackbone()
    inject_lora(source, rank=2)
    for index, block in enumerate(source.blocks):
        source.blocks[index] = checkpoint_wrapper(block, preserve_rng_state=False)
    expected = lora_state_dict(source)
    path = tmp_path / "wrapped-adapter.safetensors"
    save_lora_state_dict(source, path)

    target = _TinyBackbone()
    inject_lora(target, rank=2)
    load_lora_state_dict(target, path)

    actual = lora_state_dict(target)
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[name], actual[name]) for name in expected)
