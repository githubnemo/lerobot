# Cosmos T=2 World Expert (`cosmos-t2-we`)

2026-08-29. Status: implemented and smoke-tested; full training run not yet
started. Motivation and design discussion happened after the Cosmos T=2
LoRA result (13.74° vs 13.06° for T=16 pool2) showed a 0.68° gap despite the
6x extraction speedup (204 ms vs full T=16).

## Idea

At T=16, the 14 future noisy latent tokens learn everything they know by
attending to observed-token K/V — the observed tokens' K/V across layers 0–19
are already a sufficient interface for predicting the future. The T=2 forward
severs (a) attention mass that went to future slots and (b) the iterative
cross-talk where observed tokens read back aggregates computed in future
slots ("scratchpad"). The world expert restores the training pressure on that
interface without paying for the 16,800 future-token trunk forward:

- Trunk: Cosmos 2B, **old video LoRA merged in at load time**, plus a
  **new LoRA on blocks 0–19 only**, run at `state_t=2` (2,400 tokens),
  tapped at layer 20 (= hidden state after block index 19).
- World expert: a SmolExpert-shaped diffusion transformer (SmolVLA expert
  layers: 720 wide, 16 layers, 15 heads / 5 KV heads) that denoises the
  **14 future VAE latent frames** conditioned on the layer-20 features via
  the exact prefix-K/V mechanism SmolExpert uses for actions. The expert
  cannot invent the future — all conditioning flows through the layer-20
  K/V of observed tokens, so the bottleneck is structural, not
  capacity-based. It optimizes the same interface the action expert reads.
- Objective: unchanged Cosmos rectified flow. Same upstream sigma
  distribution (`sample_upstream_sigma`), target `noise − clean`, loss
  weight `(1+σ)²/σ²`, `latent_valid_mask` padding rules. Only the
  architecture carrying the future tokens changed (2B trunk → 115M expert).

After training, the expert can be ditched for the action pipeline: rebuild
the T=2 feature cache with merged-old-LoRA + new-LoRA trunk and retrain
SmolExpert on it. The expert is kept for video-prediction evals.

## Design decisions

- **"Blocks 0–19", not 0–20.** Layer 20 is the hidden state after 20
  executed blocks, i.e. block indices 0..19. A LoRA on block 20 would train
  a block that never runs at extraction.
- **Merge-on-load, no stored merged checkpoint.** Load order is always
  base → strict-load old adapter (rank 16 / alpha 16) → `merge_lora_into_base`
  → inject new adapter (blocks 0–19). Verified bit-identical against
  `fused-step6000.pt` (max abs diff 0). The checkpoint sidecar records the
  full recipe (paths + SHA256 of both adapters).
- **Fresh LoRA instead of warm-start.** Merging bakes the T=16 adaptation
  into full-rank weights; the new adapter gets its whole rank budget for the
  T=2 objective plus fresh optimizer state.
- **Expert token granularity = trunk tokenization.** Future latents
  `[B,16,14,60,80]` are patchified 2×2 spatial / 1 temporal → 16,800 tokens
  of 64 channels (`latent_patch_size` config knob, default 2). Learned
  positional embedding added after the input projection.
- **Bidirectional attention among latent tokens** (no causal mask — video
  latents, not an action chunk). Attention uses SDPA instead of SmolExpert's
  explicit fp32 softmax, which would not scale to 16,800 tokens.
- **RoPE / odd-even layer pattern copied from SmolExpert verbatim** (even
  layers: self-attn with prefix K/V concatenated; odd layers: query-only
  cross-attn), including its position conventions — consistency with the
  production action expert over theoretical purity.
- **No state token, no prompt input, no CFG.** The prompt is consumed once
  by the trunk cross-attention; the expert sees only layer-20 features.
- **Trunk input = extraction path.** The differentiable
  `forward_features()` is the same code `extract()` wraps with
  `no_grad`+detach, so train/extract parity of the T=2 forward
  (observed-prefix VAE encode, sigma-80 conditioning) is structural.
  At T=2 all frames are conditioned, so the extraction noise seed does not
  enter the network input.
- **Targets from the full 61-frame clip encode** (frames 2:16 of
  `[B,16,16,60,80]`), asserted to be 16 frames — the observed-only
  zero-expansion fallback is forbidden. Encodes are cached under the run's
  `latent-cache/` since targets are constants.
- **Inference shape:** the trunk runs **once** per prediction; only the
  115M expert iterates over the 35 AB2 denoising steps
  (`OfficialRectifiedFlowAB2Scheduler`, sigma 80→0.002, order 7).

## Code

- `src/lerobot/policies/vam/world_expert.py` — `WorldExpert` module.
- `scripts/video_vam/train_cosmos_t2_world_expert.py` — joint trainer
  (new LoRA + expert; single shared backward, trunk blocks 0–18
  activation-checkpointed, tap block 19 not, per the cotrain precedent).
- `scripts/video_vam/preview_cosmos_t2_we_video_prediction.py` — video
  generation on the pinned anchors (episode 0 frame 4, episode 19 frame
  2203), same side-by-side/contact-sheet outputs as the existing previews.
- `scripts/video_vam/build_cosmos_feature_cache.py` — new optional
  `--merge-lora-weights` and `--lora-blocks 0-19`; both recorded in cache
  provenance and enforced on resume. Existing behavior unchanged when unset.
- `scripts/video_vam/run_cosmos_t2_we.sh` — train (12 h cap) + both anchor
  previews; commented cache-rebuild stage for the SmolExpert follow-up.
- `cosmos_lora.py` — `inject_lora(..., block_indices=...)`,
  `merge_lora_file_into_base()`.

## Smoke test (2026-08-29, RTX 4090)

30 steps + 8-anchor val + 2-step preview: **passed**.

- 2.2 s/step, peak VRAM 13.7 GiB.
- Trainables: 16,384,000 LoRA + 114,577,600 expert.
- merge_verification vs `fused-step6000.pt`: max_abs_diff = 0.
- train world loss ~1.7–2.8 (30 steps), val_world_loss 1.537 @ sigma 80.
- Preview produced `episode-0000-frame-000004-side-by-side.mp4` etc.

Reproduce:

```bash
source scripts/video_vam/cosmos_cuda_env.sh
source scripts/video_vam/gpu_lock.sh
acquire_gpu_lock cosmos-t2-we-smoke
.venv/bin/python -m scripts.video_vam.train_cosmos_t2_world_expert \
  --smoke --no-wandb --overwrite --output-dir <run-dir>
.venv/bin/python -m scripts.video_vam.preview_cosmos_t2_we_video_prediction \
  --run-dir <run-dir> --episode 0 --frame-index 4 --steps 2 --output-dir <run-dir>/preview --overwrite
```

## Evaluation plan

1. Full run: `scripts/video_vam/run_cosmos_t2_we.sh` (train ≤12 h, then both
   anchor previews). W&B project `video-vam-world2action`.
2. Three-arm video comparison on the pinned anchors: base Cosmos T=16,
   LoRA Cosmos T=16 (`fused-step6000.pt`), and `cosmos-t2-we`. If the T=2
   rollouts are comparable to LoRA T=16, the layer-20 interface carries the
   dynamics.
3. If videos hold up: rebuild T=2 caches with the new trunk (commented
   stage in the run script), retrain SmolExpert, compare RMSE against
   13.74° (T=2 LoRA) and 13.06° (T=16 pool2).

## Caveats

- Cheaper ablation not yet run: per-spatial-position linear readout
  (2048 → 14×16 per token, shared weights) as an auxiliary loss — nearly
  free; if it alone recovers the 0.68°, the expert is unnecessary for
  feature learning.
- The observed 2 latent frames used by the trunk come from the
  observed-prefix encode; the targets come from the full-clip encode. The
  VAE's temporal context differs between the two — deliberate (deployment
  uses observed-prefix), but the first 2 latent frames of the target encode
  are not bit-identical to the trunk's input latents.
- The T=2 SmolExpert hyperparameters were inherited from pool2-shaped runs;
  part of the 0.68° gap may be hyperparameters, which the video-prediction
  eval sidesteps.
