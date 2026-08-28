# Connector-efficiency ablation and step-time profile (2026-08-20)

## Question

mimic-video feeds the full, unpooled Cosmos layer-20 hidden grid — 19,200 tokens
x 2048 dims for our 480x640/61-frame inputs — into the action decoder via
cross-attention in every block. That made our cached-mode VAM ~40x slower per
optimizer step than SmolVLA. Can we compress the context without losing
validation RMSE, and where does the remaining step time actually go?

## Setup

Six 30-minute trainings differing **only** in `--context-transform`, all on the
stride-3 sigma-80 train cache (episodes 0-31) and the stride-20 sigma-80
validation cache (episodes 32-39), batch 2 x grad-accum 2, K=8 flow draws,
lr 1e-4. Metric: the trainer's fixed-probe validation RMSE in degrees.
Reference points: `state_repeat` baseline 18.86, SmolVLA 14.83, and runD
(transform `none`, same data) 45.81 at 30 minutes / 16.59 after 14 hours.

## Results (30-minute budget each)

| arm                  | tokens | what it keeps                               | steps | best val RMSE |
| -------------------- | ------ | ------------------------------------------- | ----- | ------------- |
| runD @30min (`none`) | 19,200 | full grid                                   | 355   | 45.81         |
| **`pool2`**          | 4,800  | all frames, 2x2 spatial pool                | 808   | **23.32**     |
| `cond_frames`        | 2,400  | only the 2 clean conditioning latent frames | 810   | 24.28         |
| `pool4`              | 1,280  | all frames, 4x4 pool                        | 804   | 24.04         |
| `frame_mean`         | 16     | one token per latent frame                  | 820   | 26.94         |
| `gen_frames_pool2`   | 4,200  | only the 14 noise-derived frames, pooled    | 796   | 26.24         |
| `global_mean`        | 1      | single mean token                           | 815   | 25.66         |

## Findings

1. **Every compressed arm beats the full grid at equal wall-clock by ~2x**
   (23-27 vs 45.8 RMSE at 30 min). The full grid only catches up after 14 hours
   (16.59). Compression is a pure win at short budgets.
2. **The conditioning-frame tokens carry most of the signal.** `cond_frames`
   (2,400 tokens, only the clean observed frames) nearly matches `pool2`, while
   `gen_frames_pool2` (only the noise-derived future frames) is among the worst.
   This is consistent with the video DiT's bidirectional 3D attention: observed
   content propagates into future-frame tokens, but the reverse direction adds
   noise-dominated tokens with little extra information.
3. **Step time is NOT context-size dependent.** All six arms ran ~2.2 s/step
   whether the context had 19,200 or 1 token(s).
4. **None of the compressed arms had plateaued at 30 minutes** — `pool2` and
   `cond_frames` were still improving steeply, so the overnight to-plateau run
   determines whether compression also wins asymptotically or the full grid's
   16.59 stands.

## Step-time profile (pool2 arm, after the sweep)

Instrumented decomposition of one optimizer step (batch 2 x accum 2, K=8):

| phase                                | seconds/step |
| ------------------------------------ | ------------ |
| dataloader wait (2 prefetch workers) | 0.002        |
| H2D copy + context transform         | 0.016        |
| forward (16 small passes)            | 0.810        |
| backward (16 small passes)           | 1.184        |
| optimizer                            | 0.017        |

Loader-only throughput is 0.40 s per microbatch (the 75 MB full-grid context
files dominate disk reads), but two prefetch workers hide it completely behind
the ~2 s of compute. **The bottleneck is the K=8 flow draws executed as 16
sequential small forward/backward passes — launch-latency bound, not
FLOP-bound.**

## Optimization applied: batched flow draws

`--batched-flow-draws --flow-draw-chunk 4` folds the K draws into the batch
dimension in chunks of 4 (all 8 at once OOMs a 24 GB card with 4,800-token
contexts). The gradient is mathematically identical: flow time and noise are
drawn per batch element, and the padding mask is replicated so per-replica
valid counts match. Smoke test at the exact overnight config:

- 2.23 s/step -> **1.28 s/step (1.75x)**, peak VRAM 22.1 GiB, loss curve healthy.

Not applied (documented for later): pre-deriving reduced caches (would cut the
75 MB-per-sample disk reads 4-32x; irrelevant while prefetch workers hide I/O,
becomes relevant at larger batch sizes), K/V-reuse across decoder blocks, and
multi-layer feature taps.

## Follow-up: `cond_frames` vs observed-only DiT (`state_t=2`)

`cond_frames` kept the full 16-frame Cosmos forward and trained the expert only
on latent frames 0-1 (2,400 tokens). That is why it could not reduce DiT
latency. The 30-minute result (24.28 vs pool2 23.32) is still the best evidence
that those two observed frames carry most of the action signal.

`state_t=2` (implemented 2026-08-28) stops the DiT after those two VAE latents:
2,400 tokens from the transformer itself, pool2 -> 600 for the expert. Cache
build and inference should be several times faster; RMSE versus converged
pool2 is unknown because the features are no longer mixed with future-frame
noise via self-attention. Next training step: rebuild the LoRA-adapted cache
with `--state-t 2` and train SmolExpert on it. Do not evaluate the 4,800-token
experts on 600-token context.

## Deviation from mimic-video

Upstream keeps the full 19,200-token context (documented in
`docs/mimic_video_reference.md`). The compression transforms are our
efficiency contribution; the `none` arm remains the faithful reference.

## Next step (running overnight 2026-08-20 -> 21)

`pool2` + batched draws trained until fixed-probe validation RMSE stops
improving, then evaluated with the frozen action-RMSE protocol against
SmolVLA (14.83), `state_repeat` (18.86), and runD (16.92).
