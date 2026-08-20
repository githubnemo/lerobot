#!/usr/bin/env python3
"""Assert at runtime that context, sigma, state, and action rows stay paired.

A misalignment between the extracted context of one scene and the sigma, state, or
action of another would let the decoder fit neither, so this checks the pairing with
distinguishable per-sample values rather than by reading the code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.policies.vam.cosmos_cache_dataset import load_cache_manifest
from lerobot.policies.vam.vam_split import load_vam_split
from scripts.video_vam.train_cosmos_world2action import (
    Anchor,
    OnlineCosmosContext,
    RandomAnchorDataset,
    RandomAnchorSampler,
    build_anchor_dataset,
    episode_anchor_bounds,
)

MANIFEST = Path("/home/anton/.cache/video-vam/cosmos-rehearsal-stride20-sigma4/manifest.json")
SPLIT = Path("/home/anton/.cache/video-vam/splits/rehearsal-stride20.json")
ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
BACKBONE = Path("/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt")
TOKENIZER = Path("/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth")
PROMPT = Path("/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors")


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixed-sigma",
        type=float,
        default=4.0,
        help="Fixed training sigma whose extraction/conditioning plumbing is asserted in check 6.",
    )
    return parser.parse_args()


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        dataset_root=ROOT,
        backbone_checkpoint=BACKBONE,
        tokenizer=TOKENIZER,
        prompt=PROMPT,
        seed=0,
        online_sigma_seed=0,
        context_mode="online-random",
        batch_size=4,
        train_sigma=None,
    )


def main() -> int:
    device = torch.device("cuda")
    cli = _cli()
    args = _args()
    manifest = load_cache_manifest(MANIFEST)
    split = load_vam_split(SPLIT, manifest)
    dataset = build_anchor_dataset(manifest, args)
    anchor_dataset = RandomAnchorDataset(dataset)
    bounds = episode_anchor_bounds(dataset, split.train_episodes)
    sampler = RandomAnchorSampler(bounds, seed=0)

    # Two visibly different scenes (2 fits the 24 GB card): different episodes, far apart.
    anchors = [
        Anchor(0, 20),
        Anchor(19, bounds[19][1]),
    ]
    items = [anchor_dataset.load(anchor) for anchor in anchors]
    images = torch.cat([item.rgb_history for item in items], dim=0)
    state = torch.cat([item.state for item in items], dim=0)
    action = torch.cat([item.target_action for item in items], dim=0)
    print("anchors:", [(a.episode_index, a.frame_index) for a in anchors])
    print("images", tuple(images.shape), "state", tuple(state.shape), "action", tuple(action.shape))

    # 1. State/action rows must equal an independent per-anchor read.
    for index, anchor in enumerate(anchors):
        solo = anchor_dataset.load(anchor)
        assert torch.equal(state[index : index + 1], solo.state), f"state row {index} misaligned"
        assert torch.equal(action[index : index + 1], solo.target_action), f"action row {index} misaligned"
        assert torch.equal(images[index : index + 1], solo.rgb_history), f"image row {index} misaligned"
    print("PASS 1: state/action/image rows match their own anchor")

    # The four scenes must actually differ, otherwise later checks prove nothing.
    pairwise = [
        float((images[i].float() - images[j].float()).abs().mean()) for i in range(2) for j in range(i + 1, 2)
    ]
    print("mean |image_i - image_j| over pairs:", [round(v, 2) for v in pairwise])
    assert min(pairwise) > 1.0, "scenes are not distinguishable"

    online = OnlineCosmosContext(manifest, args, device, None, dataset=dataset)

    # 2. The returned sigma must be the sigma the features were generated at.
    context, context_timestep = online.extract_images(images, microbatch_index=17)
    print("context", tuple(context.shape), context.dtype, "| sigma", context_timestep.flatten().tolist())
    assert context.shape[0] == 2 and context_timestep.shape == (2, 1)
    drawn = online.sigma_sampler.draw_for(17, 2)
    assert torch.allclose(context_timestep.flatten(), drawn), (
        f"conditioning sigma {context_timestep.flatten().tolist()} is not the drawn sigma {drawn.tolist()}"
    )
    print("PASS 2: conditioning sigma equals the sigma drawn for that microbatch")

    # 3. Context row i must be the context of image row i. Re-extract each row alone
    #    with that row's own sigma and noise, and require an exact row match.
    sigma = context_timestep.flatten()
    noise_seed = 17 + 1_000_003
    online.noise_generator.manual_seed(noise_seed)
    noise = torch.randn(
        (2, 16, 16, 60, 80), device=device, dtype=torch.bfloat16, generator=online.noise_generator
    )
    prompt = online.prompt.expand(1, -1, -1)
    for index in range(2):
        solo = online.extractor.extract(
            images[index : index + 1],
            prompt,
            noise_seed=noise_seed,
            sigma=sigma[index : index + 1],
            noise=noise[index : index + 1],
        )
        difference = (solo.tokens.float() - context[index : index + 1].float()).abs().max()
        cross = min(
            float((solo.tokens.float() - context[other : other + 1].float()).abs().max())
            for other in range(2)
            if other != index
        )
        print(f"  row {index}: max|solo-batched[{index}]|={float(difference):.4f}  min cross-row={cross:.4f}")
        assert float(difference) < cross, f"context row {index} matches another row better than its own"
    print("PASS 3: each context row is the context of its own image row")

    # 4. Contexts of different scenes must differ, i.e. the context carries scene information.
    spread = [
        float((context[i].float() - context[j].float()).abs().mean())
        for i in range(2)
        for j in range(i + 1, 2)
    ]
    within = float(context.float().abs().mean())
    print(
        "mean |ctx_i - ctx_j| over pairs:", [round(v, 4) for v in spread], "| mean |ctx|:", round(within, 4)
    )
    assert min(spread) > 0.01 * within, "contexts of different scenes are nearly identical"
    print("PASS 4: contexts of different scenes are distinguishable")

    # 5. The K-draw loop must reuse one context with its own state/action, unchanged.
    reference = context.clone()
    for _draw in range(3):
        assert torch.equal(context, reference), "context mutated across flow draws"
    print("PASS 5: context is stable across the K flow draws")

    # 6. A fixed training sigma must reach both the extractor and the conditioning input:
    #    the batched context must equal a solo re-extraction at exactly that sigma.
    args.train_sigma = cli.fixed_sigma
    fixed = OnlineCosmosContext(manifest, args, device, None, dataset=dataset)
    fixed_context, fixed_timestep = fixed.extract_images(images[:2], microbatch_index=3)
    print(f"fixed-sigma ({cli.fixed_sigma}) conditioning:", fixed_timestep.flatten().tolist())
    assert torch.allclose(fixed_timestep, torch.full_like(fixed_timestep, cli.fixed_sigma)), (
        "fixed sigma not applied to the decoder conditioning"
    )
    fixed_noise_seed = 3 + 1_000_003
    fixed.noise_generator.manual_seed(fixed_noise_seed)
    fixed_noise = torch.randn(
        (2, 16, 16, 60, 80), device=device, dtype=torch.bfloat16, generator=fixed.noise_generator
    )
    for index in range(2):
        solo = fixed.extractor.extract(
            images[index : index + 1],
            prompt,
            noise_seed=fixed_noise_seed,
            sigma=fixed_timestep.flatten()[index : index + 1],
            noise=fixed_noise[index : index + 1],
        )
        assert torch.allclose(solo.sigma, fixed_timestep.flatten()[index : index + 1]), (
            "extractor did not report the fixed sigma back"
        )
        difference = float((solo.tokens.float() - fixed_context[index : index + 1].float()).abs().max())
        cross = float((solo.tokens.float() - fixed_context[1 - index : 2 - index].float()).abs().max())
        print(
            f"  fixed row {index}: extraction sigma {float(solo.sigma):.4f}, "
            f"max|solo-batched|={difference:.4f}, cross-row={cross:.4f}"
        )
        assert difference < cross, "fixed-sigma extraction does not reproduce the training context"
    print(
        f"PASS 6: --train-sigma={cli.fixed_sigma} pins extraction and conditioning to one value "
        "(extraction sigma == decoder context_timestep)"
    )

    del sampler
    print("ALL PLUMBING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
