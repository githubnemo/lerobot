# Design: LoRA co-training of the Cosmos backbone (video + action joint loss)

Status: agreed design, 2026-08-21 evening (user + assistant). Implementation
in preparation; run queued after the baselines queue.

## Why (the insight chain)

1. **Our controls showed frozen pretrained Cosmos features ≈ random features**
   (24.04 vs 23.94 fixed-probe RMSE at 30 min). The frozen-backbone,
   pure-noise-future setup does not let the pretraining pay off.
2. **Partial noising is not a leak — it is the enabler of joint training.**
   Upstream feeds noised GROUND-TRUTH future latents at a randomly sampled
   sigma. With `GT + noise` as input, the same forward pass supports BOTH the
   video denoising/flow loss (impossible from pure noise — no target) AND the
   action loss from tapped features. Randomized sigma is just the standard
   diffusion training distribution. BFL's Self-Flow makes the pairing explicit
   (dual-timestep scheduling for generation + representation learning), and
   FLUX-mimic keeps the backbone trainable. Their stated position: co-training
   the backbone to predict video better during action conditioning is key.
3. **Deployment needs no generated futures.** The single high-sigma forward IS
   an implicit one-step future prediction ("internal representation of
   predicted future frames... without ever rendering pixels"). Both upstream
   and we deploy this way; the difference is their backbone makes a good
   one-step guess on their data. Ours guesses badly on OUR data because:
   - it never saw FPV wrist-camera video (we never see the arm — OOD viewpoint;
     the bridge LoRA is also third-person), and
   - it was never adapted to this scene/task.
4. Therefore: **LoRA-adapt the backbone on our data with a joint loss.** This
   simultaneously (a) makes it a better world model for OUR videos, (b) shapes
   features for action decoding, and (c) restores upstream training fidelity
   (noised-GT futures at randomized sigma) — closing documented deviation #1
   in docs/mimic_video_reference.md.

## Training recipe

Per sample (anchor t in a train episode, all frames available in-dataset):

1. Load 5 past frames (t-4..t) AND the 56 future frames (t+1..t+56) — the
   full 61-frame clip, as upstream.
2. VAE-encode -> 16 latent frames. Frames 0-1 clean conditioning; frames 2-15
   = `GT_latent * (1-mask) -> noised at sigma` with sigma sampled from the
   upstream distribution: `sigma = 4 * exp(N(0,1))`, 5% heavy tail
   `exp(U(log 200, log 1e5))`.
3. ONE backbone forward with LoRA adapters active, capturing (a) the final
   velocity/denoising prediction and (b) layer-20 hidden states.
4. Losses from the same pass:
   - `L_video`: rectified-flow loss on the backbone prediction vs GT latents
     (non-conditioning frames only) — the backbone's own pretraining loss.
   - `L_action`: existing World2Action flow-matching loss on layer-20 features
     (sigma fed to the decoder as today; K flow draws per context still apply).
   - `L = L_action + lambda_video * L_video`.
5. Trainable: LoRA adapters + action decoder. Base weights frozen bf16.

Deployment/validation: unchanged protocol — 5 real frames + pure-noise future
at sigma 80, one forward, decoder conditioned on sigma. The sigma-randomized
training covers the high-sigma deployment regime; this is the same
train/deploy asymmetry upstream tolerates.

## Open questions and planned ablations (user-flagged)

1. **Does the action loss flow into the backbone (LoRA)?** Ablate:
   - arm A: both losses update LoRA (full co-training),
   - arm B: only L_video updates LoRA; action gradients stopped at the feature
     tap (backbone = better world model only),
   - (arm C, cheap reference: LoRA off = current frozen baseline).
2. **Self-Flow extension.** BFL claims large gains from dual-timestep
   scheduling (a generation timestep + a representation timestep in the same
   training step; Self-Flow, arXiv:2603.06507). Worth adding AFTER the basic
   joint loss works. Implementation must first re-read the paper for the exact
   scheduling/loss; do not improvise it.
3. **Loss weighting (lambda_video).** Annoying to tune, agreed. Plan: start
   lambda=0.5; log both raw losses; if one gradient norm dominates >10x, switch
   to gradient-norm balancing or uncertainty weighting rather than grid search.
   One-hour budget arms decide; no big sweeps.
4. Sigma distribution: upstream-randomized (with decoder conditioning) is the
   default here since it is required for a meaningful L_video across noise
   levels. Note our earlier 1-hour frozen-backbone result where randomized
   sigma hurt — that finding may not transfer to the co-training regime; watch
   the fixed-probe RMSE (evaluated at sigma 80) closely.

## Cost & feasibility (4090, 24 GB)

- No feature cache possible (backbone changes every step): online forward +
  backward through 2B DiT with LoRA. bf16 weights ~4 GB; gradient
  checkpointing on DiT blocks; batch 1 + grad accum; VAE-encode 61 frames per
  sample (consider precomputing GT latents per anchor once — latents are
  backbone-independent! 16x60x80x16ch bf16 ≈ 24 MB/sample — a latent cache
  IS valid and avoids repeated VAE cost).
- Expected ~2-4 s/step -> 1-2k steps/hour. This is a long overnight run, as
  the user anticipated. Early-stop on fixed-probe val RMSE.
- LoRA: rank 16-32 on attention q/k/v/o + MLP projections of all 28 blocks
  (~20-40M trainable params), lr ~1e-4 for LoRA, existing lr for decoder.

## What this tests (report framing)

The mimic thesis proper: does a co-trained video backbone beat (a) frozen
pretrained features, (b) random features, and (c) SmolVLA (14.83°) at equal
wall-clock on 32 episodes of FPV SO-101 data? Prior deviations are closed;
remaining known gap vs upstream: dataset scale and the FPV viewpoint itself.

## Full-clip boundary policy

The expanded camera contract requests offsets `-4..+56` without changing the
existing causal contract. At episode ends, LeRobot repeats the episode's final
frame and marks each out-of-range camera position with `camera_is_pad`. The
repeated pixels remain in the VAE input so every anchor has a fixed 61-frame
shape, but any latent temporal bin containing a padded pixel is excluded from
`L_video`. The five historical frames are required to be real; action padding
continues to use the existing `action_is_pad` mask.
