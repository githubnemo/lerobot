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
   Done for generic prefix-pool2 (13.81 deg at 36k) and video-LoRA expert
   (13.06 deg at 38k, W&B ixzworl4, stopped on patience 10).
3. LoRA fine-tune of the backbone on our 32 training episodes, video loss
   only, then freeze and retrain the action expert. Distinct from the failed
   co-training arm (joint losses). Cosmos video-only LoRA (step 6k) + SmolExpert
   is done and is the current offline best. LTX-2.5 video LoRA still open
   (22B will not LoRA-train on a 4090).
4. Fast video features for control (pretrained -> LoRA -> short forward ->
   action expert). Do not start another backbone LoRA until this is measured.
   Stack:
   1. Pretrained video model (Cosmos-2B; LTX later if needed).
   2. Video-only LoRA on our 32 episodes, then freeze. Cosmos step-6k is done
      (13.06 deg expert). Do not jointly train video+action (already worse).
   3. Cheap DiT forward, then train a **new** SmolExpert on that cache (train
      and eval must match). Two options, in this order:
      - **First: observed-only (`state_t=2`).** DiT on the two VAE latents
        only (2,400 tokens, pool2 -> 600). Cache build and inference get the
        2-4x; the expert does not. Closest prior is `cond_frames` (W&B
        rmcpaxtc): expert on those two frames _after_ a full 16-frame DiT,
        24.28 deg vs pool2 23.32 at 30 min. That slice does not speed up Cosmos.
        `state_t=2` never mixes prefix tokens with future noise, so RMSE vs
        13.81 / 13.06 is the open question. Extractor path is already in
        tree (`--state-t 2`).
      - **If T=2 drops quality: one-step future (T=3).** Keep the two cond
        frames and add one noisy/predicted latent (~4 RGB frames, ~0.4 s at
        10 Hz). Still ~5x fewer tokens than T=16. Optional extra LoRA for
        next-latent / short-horizon prediction only (Diffusion Forcing /
        Dreamer-style next-latent world models), then freeze again. This is
        not one-denoising-step distillation; we already do a single high-sigma
        forward. The win is fewer tokens. Do not rerun 16-frame video LoRA
        for this.
   4. Action expert on the resulting features. Even a tiny DiT does not
      reach 10 Hz alone (VAE ~120 ms, expert ~90 ms). T=2 DiT in the
      80-150 ms band is ~3 Hz per chunk, enough to hide behind a 30-step /
      3 s buffer. Overlap still needed for true 10 Hz.
      Details: `video_vam_latency_optimization_plan.md`,
      `video_vam_connector_ablation_report.md`.
5. After this `state_t=2` unpooled expert is tested (RMSE vs 13.06 / 13.81),
   train on the new cube-out-of-box **v2** dataset: video-LoRA cache + SmolExpert
   on the same short-forward contract. Do not start v2 until v1 T=2 is scored.
6. Real-robot deployment of best SmolVLA / Cosmos / LTX policies on a freshly
   collected dataset; success-rate is the metric that matters, RMSE is proxy.
   Real-time gap: video backbones ~1 Hz vs 10 Hz target; RTC + chunking plus
   the fast-feature stack in (4) if RMSE holds. `torch.compile` is the rollout
   default (~1.17x DiT); `--no-compile` disables it. Mac can only run SmolVLA
   locally; Cosmos/LTX go through abakus RPC.
7. Joint-state / calibration noise during SmolExpert training (not at deploy).
   Sample a per-window or per-batch bias/scale on the 6 joints in physical
   units, apply the same offset to action targets, then normalize. Val stays
   clean. Isolates SO-101 recalibration mismatch and the episode-24 sign flip
   without rebuilding video caches.
8. Pregenerate augmented feature caches: apply cheap image augs (photometric
   noise, small shifts/crops) _before_ Cosmos/LTX extract; write extra cache
   entries with aug seed in provenance; mix with the clean 1,572-window cache
   at train time. Val stays unaugmented. Do not replace the clean cache.
   Budget: clean train cache is already ~31 GB / ~30 min; K extra augs costs
   ~K times that. Start with K=2 photometric-only.
9. Cosmos+LoRA continuation-video comparison vs generic Cosmos on the same
   windows (episode 0 frame 4, episode 19 frame 2203) using
   preview_cosmos_video_prediction.py --checkpoint fused-step6000.pt.
10. Cheap high-value ablations if time permits: image-only backbone control,
    multi-seed random-init, Cosmos-14B scale point.

## What this argues for a BFL forward-deployed-engineer story

Taking a frontier video model (LTX-2.5), building a feature interface for it,
beating both the incumbent VAM recipe (Cosmos/mimic-video) and a standard VLA
on a real robot dataset with 32 demos, with documented ablations separating
pretraining, scale, and architecture effects - plus a clear-eyed account of
the real-time gap and the domain-mismatch (FPV) limitation.
