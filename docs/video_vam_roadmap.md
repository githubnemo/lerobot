# Video-VAM roadmap and open ablations

Written 2026-08-26. Companion docs: research diary (journey), cube-out-of-box
leaderboard (numbers), world-model usage comparison (architecture survey),
mimic-video reference (upstream fidelity).

## The journey in five sentences

We built a causal training/eval pipeline on the SO-101 cube-out-of-box dataset
(40 episodes, gripper-mounted FPV camera) and compared action policies driven
by frozen video-foundation-model features against a conventional VLA. Frozen
Cosmos-2B layer-20 features with a SmolVLA-style action expert reach 13.81
deg RMSE; frozen LTX-2.5-22B features reach 13.65; a converged, leakage-free
SmolVLA baseline reaches 14.93. Both video backbones therefore beat the VLA
under the frozen protocol, at the cost of ~1 Hz inference vs the 10 Hz target.
Randomized-backbone controls degrade markedly, so pretrained features carry
signal beyond the architecture. Joint LoRA co-training (video+action) on
Cosmos was worse than frozen features in our one attempt.

## Feature interface (exact shapes)

- Cosmos-2B, layer 20 of the DiT: tokens (B, 19200, 2048); the 19200 is a
  flattened T,H,W grid of 16 latent frames x 30 x 40 spatial patches.
  Latent frames 0-1 are the VAE-encoded observed frames; frames 2-15 enter
  as rectified-flow noise (the future slots). One forward pass, no sampling.
- Pooling is spatial-only per latent frame, channels and temporal axis
  untouched: pool2 = adaptive 2x2 avg -> (16,15,20) = 4800 tokens;
  pool4 -> (16,8,10) = 1280; frame_mean -> 16 tokens. Ablation showed
  little RMSE cost down to pool2, large training-speed gain.
- LTX-2.5-22B: tokens (B, 2400, 4096) = 8 latent frames x 15 x 20 (stronger
  VAE compression); unpooled LTX is already smaller than pool2 Cosmos.
  Layer probe (running) taps six depths and learns a scalar mix to pick
  the readout layer.

## Confounders we can name, and which ablation isolates each

| Confounder                                                          | Status                                                                                                                                                   |
| ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Architecture + conditioning alone (no pretraining)                  | Random-init Cosmos control done: much worse. Caveat: single seed; random init may be ill-conditioned rather than knowledge-free. Multi-seed repeat open. |
| Just model scale, any pretraining                                   | Partially covered (SmolVLA 500M vs 2B vs 22B, but recipes differ). Clean scale ablation open: Cosmos 2B vs 14B, identical recipe.                        |
| Time-as-input vs video pretraining (would an image DiT do as well?) | OPEN. Strongest missing ablation: same conditioning through an image-only diffusion backbone of similar scale.                                           |
| Video pretraining data domain (third-person web video vs our FPV)   | Evidence of mismatch: generated rollouts hallucinate human hands. Bridge-adapted Cosmos test queued (partial run tracked generic almost exactly).        |
| Readout depth                                                       | Cosmos: layer 20 fixed (mimic-video choice). LTX: layer probe running now.                                                                               |
| Decoder capacity vs feature quality                                 | Done: SmolVLA expert on Cosmos features vs World2Action decoder; expert head better.                                                                     |

## Next steps (priority order)

1. LTX layer probe -> retrain best LTX configuration to plateau (no wall-clock
   cap; cap only as safety net).
2. Cosmos to plateau under the same stopping rule, so converged-vs-converged.
3. LoRA fine-tune of the backbone on our 32 training episodes, video loss
   only, then freeze and retrain the action expert. Distinct from the failed
   co-training arm (joint losses). Do for LTX-2.5 first (untested, our best
   backbone), Cosmos second.
4. Real-robot deployment of best SmolVLA / Cosmos / LTX policies on a freshly
   collected dataset; success-rate is the metric that matters, RMSE is proxy.
   Real-time gap: video backbones ~1 Hz vs 10 Hz target; RTC + chunking is
   the mitigation, quantization/compile gains largely exhausted on 4090.
5. Cheap high-value ablations if time permits: image-only backbone control,
   multi-seed random-init, Cosmos-14B scale point.

## What this argues for a BFL forward-deployed-engineer story

Taking a frontier video model (LTX-2.5), building a feature interface for it,
beating both the incumbent VAM recipe (Cosmos/mimic-video) and a standard VLA
on a real robot dataset with 32 demos, with documented ablations separating
pretraining, scale, and architecture effects - plus a clear-eyed account of
the real-time gap and the domain-mismatch (FPV) limitation.
