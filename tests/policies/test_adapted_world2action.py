import torch
from torch import nn

from lerobot.policies.vam.adapted_world2action import AdaptedWorld2ActionDecoder
from lerobot.policies.vam.world2action import World2ActionConfig


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.action = nn.Linear(6, 6)
        self.context = nn.Linear(2048, 6)

    def forward(self, **kwargs):
        action = self.action(kwargs["xt_B_HA_A"])
        context = self.context(kwargs["crossattn_emb"].float().mean(dim=1, keepdim=True))
        return action + context


def test_adapter_is_inside_decoder_and_loss_decreases():
    torch.manual_seed(0)
    config = World2ActionConfig(device="cpu", dtype=torch.float32)
    decoder = AdaptedWorld2ActionDecoder(
        4096,
        config,
        denoiser=TinyDenoiser(),
    )
    context = torch.randn(2, 8, 4096, dtype=torch.bfloat16)
    state = torch.zeros(2, 1, 6)
    action = torch.ones(2, 30, 6)
    epsilon = torch.zeros_like(action)
    t = torch.full((2,), 0.25)
    optimizer = torch.optim.SGD(decoder.parameters(), lr=0.05)
    losses = []
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        loss = decoder.flow_matching_loss(state, action, context, t=t, epsilon=epsilon, obs_dropout=0.0)
        loss.backward()
        assert decoder.context_adapter.projection.weight.grad is not None
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert not context.requires_grad
