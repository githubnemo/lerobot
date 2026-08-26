# Video-features-to-action connectors: literature survey

_Compiled 2026-08-20. Question: when a policy conditions on features from a
pretrained video model, what does the interface ("connector") between the video
model and the action head look like — and is mimic-video's full unpooled
19,200-token cross-attention typical?_

## Bottom line

A full ~19,200 × 2,048 video-token grid cross-attended by every block of a
separate action decoder is **not the dominant modular design**. It is an
outlier. Modular systems (frozen video model + separate action head) compress
aggressively — usually to 32–512 tokens — before the action head sees anything.
Full visual grids appear only in _unified_ models where video and action share
one transformer.

The main connector patterns:

1. **Learned-token compression** (query tokens / perceiver / "video former"):
   32–256 tokens.
2. **Temporal/global compression**: one vector, or roughly one token per future
   frame.
3. **Latent-frame injection / shared attention**: no separate connector at all
   (unified backbones).
4. **Dense multi-level transfer**: emerging (DeVA, τ0-WM), token counts mostly
   unreported.

## Closest direct comparisons

### mimic-video ([arXiv 2512.15692](https://arxiv.org/abs/2512.15692))

Our reference. 2B Cosmos-Predict2, hidden state after video block 20 (of 28) at
sampled video noise level. The `(B, 16, 30, 40, 2048)` grid is flattened by a
single `reshape` to `(B, 19200, 2048)`; **every one of the 24 action-DiT blocks
cross-attends to it, K/V recomputed from all 19,200 tokens per block**. No
pooling, no resampler, no connector (verified at commit `e3355db`, see
`mimic_video_reference.md`). The backbone is frozen; features are computed
online per training step, never cached. The strongest published example of the
"full unpooled grid" design — affordable for them via multi-GPU training, not
via any efficiency machinery.

### Video Prediction Policy / VPP ([arXiv 2412.14803](https://arxiv.org/abs/2412.14803))

1.5B Stable Video Diffusion fine-tuned as a manipulation video predictor; a
single encoder-style forward pass on highly noised input. Aggregates multiple
up-sampling layers, then a learned **Video Former** (spatial attention per
frame, then temporal attention) reduces everything to **224 tokens** (e.g.
16 × 14 × 384 on Calvin). The diffusion action policy cross-attends only to
those. One video forward < 160 ms → 7–10 Hz control. **Ablation: removing the
Video Former made performance worse and latency ~450 ms** — compression helped
quality, not just speed. The single most relevant precedent for our efficiency
arm.

### Video Generators are Robot Policies ([arXiv 2508.00795](https://arxiv.org/abs/2508.00795))

SVD U-Net, five decoder layers (9/14/17/20/23) at every denoising step; a CNN
adapter collapses everything into **a single global vector** conditioning a 1D
action U-Net.

### VidMan ([arXiv 2411.09153](https://arxiv.org/abs/2411.09153))

Open-Sora VDT evaluated at max diffusion step (pure noise input, no iterative
denoising). Fuses **all 12 layers** via a small number of learnable action
tokens inserted at every layer (layer-wise self-attention adapter). Action head
sees only those learned tokens.

### MinD ([arXiv 2506.18897](https://arxiv.org/abs/2506.18897))

DiffMatcher maps `(B, C, T, H, W)` video latents to `(B, T, D)` — **about one
compact token per temporal latent frame** — via per-frame MLPs plus a temporal
transformer. Explicitly designed to avoid passing the full spatial map. ~11.3
FPS real-world.

### FLARE ([arXiv 2505.15659](https://arxiv.org/abs/2505.15659)) / GR00T N1.5

Not a video-DiT connector, but the design GR00T N1.5 adopted: a Q-Former
compresses vision features to **32 learned query tokens**; the action DiT
aligns 32 learnable future tokens against compact future embeddings instead of
generating frames.

### DeVA ([arXiv 2607.24159](https://arxiv.org/abs/2607.24159))

Cosmos-Predict2 features from multiple backbone levels, layer-wise
cross-attention into matched action blocks plus **learnable bridge tokens**;
affordance/depth decoders add physically grounded keys/values. Modern dense
transfer, but still aggregates via bridge tokens; token counts not reported.

## Related but architecturally different

- **GR00T N1** ([arXiv 2503.14734](https://arxiv.org/abs/2503.14734)): Eagle-2
  VLM (not a video generator), 64 image tokens per frame, action DiT
  cross-attends layer-12 VLM states. 63.9 ms per 16-action chunk on an L40.
- **DreamGen** ([arXiv 2505.12705](https://arxiv.org/abs/2505.12705)): video
  world model generates synthetic trajectories; interface is generated pixels +
  pseudo-actions (IDM/LAPA), not hidden states.
- **UniPi / UniSim / SuSIE**: generate video or subgoal images; a separate
  inverse-dynamics or goal-conditioned policy consumes _pixels_, never internal
  activations.
- **Unified shared-backbone models** (GR-2, WorldVLA, PAD, UVA, VideoVLA,
  Cosmos Policy [arXiv 2601.16163](https://arxiv.org/abs/2601.16163),
  DreamZero, Motus, LingBot-VA, Fast-WAM): video and action tokens live in one
  transformer; hundreds-to-thousands of visual tokens are normal there, but
  there is no external decoder repeatedly cross-attending a frozen feature
  grid. Motus notably _downsamples video 6× temporally_ to balance token
  counts — token imbalance is a recognized practical problem even in unified
  models.
- **τ0-WM** ([arXiv 2606.01027](https://arxiv.org/abs/2606.01027)):
  architecturally closest to mimic (separate ~0.5B action branch cross-attends
  intermediate video features at matched stages) but token counts unreported.

## Scale of our problem

One mimic-style context is `19,200 × 2,048 ≈ 39.3M` scalars, ~75 MiB in bf16
before K/V projections — recomputed in each of 24 decoder blocks, per flow
draw. This is why our cached-feature training step (4.76 s) is still ~40×
slower than SmolVLA's full online step (0.12 s).

## Literature-aligned ablation menu for our cached features

- Full 19,200 tokens (faithful mimic reference — already run: runD, 16.92°).
- 2×/4× spatial pooling (no new parameters).
- Temporal slicing: conditioning-frame tokens only (2 × 1,200 = 2,400) vs
  generated-frame tokens only (14 × 1,200 = 16,800) vs both.
- One token per latent frame (16 tokens, MinD-style).
- 224–512 learned query tokens (VPP-style Video Former).
- Later, once efficient: multi-layer features (VidMan/DeVA-style), layer sweep.

Note on attention structure: the Cosmos video DiT uses **full bidirectional 3D
attention** (`attn_mask_type="no_mask"`, `causal=False` in the vendored
`text2image_dit.py` / `module/attention.py`). By layer 20 every token mixes
information from the whole grid, including the two clean conditioning latent
frames. Token _position_ therefore does not cleanly separate "observed" from
"imagined" content — which is exactly why the temporal-slicing ablations above
are informative rather than redundant.
