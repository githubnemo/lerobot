# mimic-video world2action — implementation reference and our compliance

Source of truth: `mimic-video/mimic-video` @ `e3355dbc93132b576c02f920a59b4fc18a4f5906`
(the commit our vendored Cosmos code came from). Extracted 2026-08-19 with file/line
citations; citations refer to that repo.

Rule: **stay as close as possible to this reference.** Every deviation must be listed
in the table at the bottom, with a reason. When an experiment underperforms, check this
document before inventing new mechanisms.

---

## 1. Architecture

- Frozen video backbone: Cosmos-Predict2 `Video2WorldPipeline` (2B, 480p, 10 fps),
  `requires_grad_(False)` (`world2action_model.py:118`).
- Trainable action decoder: `World2ActionDIT`, trained **from scratch**
  (`action_dit_path = ""`; random init when empty). Capacity: `model_channels=1024`,
  `num_blocks=24`, `num_heads=8`, `mlp_ratio=4.0`, `crossattn_emb_channels=2048`,
  `use_adaln_lora=True`, `adaln_lora_dim=128` (`defaults/world2action_pipe.py:9-39`).
- EMA disabled for the action decoder (`EMAConfig(enabled=False)`).

**Our status: matches** (decoder scratch-initialized, same capacity, no EMA).

## 2. Conditioning extraction

- Video input to the backbone: 61 pixel frames = 5 observation frames ending at the
  anchor `t` + 56 frames after `t` (ground-truth future during training, generated at
  deployment), VAE-encoded to 16 latent frames (`video2world.py:596-622`).
- First 2 latent frames are clean conditioning (`FRAME_REPLACE`, conditional sigma
  1e-4); the rest are noised (`video2world.py:710-726`).
- Features: hidden states of **layer 20 of 28**, kept **unpooled**, flattened to
  `(B, T·H·W, D) = (B, 19200, 2048)` (`world2action_model.py:335-352`).
- Camera processing: single third-person view, resize to 480x640 (no crop), RGB,
  range [-1, 1] (`data_transforms.py:201-223`).

**Our status: matches, with two intentional structural differences we cannot close on this
dataset:**

- We feed only the 5 past frames; the 56 future slots are pure noise at the model's own
  scale (verified: the zero pixel padding never reaches the network — the non-conditioning
  latents are overwritten with `noise·sigma·c_in`). Upstream's noised latents contain
  `true_future + noise`, so at moderate sigma some real future is visible to the backbone.
  Ours contain none. This is the deliberate "causal" choice; revisit only with a
  generate-the-future design.
- New online extraction and cache builds use `observed_prefix`: the 5 real pixels are VAE
  encoded to 2 latents, then zero-expanded to 16 before the same future-noise and mask
  assembly. `legacy_padded_vae` is an explicit compatibility/debug opt-out that restores
  the upstream-shaped 5 -> 61 pixel VAE input. Prefix encoding is intentionally approximate,
  not an exact upstream reproduction; its mode is recorded in new cache provenance.

## 3. Sigma (the backbone noise level)

- Randomized **per training sample**: `sigma = sqrt(state_t)·exp(N(0,1))`
  = `4·exp(N(0,1))` (median ≈ 4), and with probability 0.05 replaced by
  `exp(Uniform(log 200, log 100000))` (`world2action_model.py:360-379`).
- The sigma value is fed to the action decoder as conditioning through a
  `PairTimeEmbedder` (`world2action_dit.py:458-502`) — the decoder always knows how
  noisy its context is.
- At validation upstream sweeps sigma over the 35-step solver schedule; at deployment
  the sigma is whatever the video solver step provides.

**Our status: implemented faithfully (randomized per sample, fed to the decoder), but
empirically harmful at our budget** — fixed sigma 10 reached ~26° RMSE, randomized
reached 57° in a 1-hour run. Open question whether that is a bug or a budget effect.
We validate and deploy at a single fixed sigma (4.0), so a fixed-sigma training is a
defensible deviation _for now_; mark any such run clearly.

## 4. Targets and alignment

- Anchor `t` sampled randomly from episode timestamps (not from a precomputed
  stride grid). **Our stride-20 window manifest was a deviation; now fixed with
  uniform random anchors.**
- Chunk start is dataset-specific via `shift_right_by` (Bridge: `t+1` at 5 Hz;
  LIBERO: `t`). Ours starts at `t` (LeRobot convention) — consistent everywhere,
  acceptable.
- Actions upstream: 10-dim relative EEF pose + absolute gripper, rot as 6D. Ours:
  6 absolute joint positions (SO-101). **Deliberate deviation** (different robot,
  joint-space control); affects normalization choices below.
- Proprio state: 1 token, at frame `t`, with `obs_dropout=0.2` replacing it by a
  learned mask token during training (forces vision use). **Ours matches.**

## 5. Normalization

- Upstream: per-dimension **mean/std** (`VARIANCE` mode) with percentile-based
  clamping (p2/p98 widened 1.5x) for pos and gripper; rotations left untouched
  (`normalizer.py:165-236`). Stats include repeated-tail padding; loss has **no
  padding mask** (`world2action_model.py:307`).
- Ours: per-joint **min-max to [-1, 1]** computed on train episodes only, padded
  actions excluded from loss AND stats; the inverse clamps to [-1,1] (saturating
  outputs to the training joint range).

**Our status: deviates.** Defensible (min-max is common for joint control and the
padding mask is strictly cleaner), but it is a difference to remember when comparing
losses to upstream. Not currently suspected of causing failures.

## 6. Objective (flow matching)

- Interpolant `xt = (1−t)·x0 + t·ε`; target velocity `u = ε − x0`;
  `t ~ Uniform[0.001, 0.999]`; one `t` per batch upstream.
- **Asymmetry to preserve:** the network input is rescaled `xt / sqrt((1−t)²+t²)`
  but the target is NOT rescaled (`world2action.py:169` vs `world2action_model.py:296`).
- The denoiser output has a leading state token that must be sliced off
  (`[:, HO:, :]`, HO=1) before the loss (`world2action.py:190`).
- `loss_scale = 10.0`, `loss_reduce = "mean"`.

**Our status: matches**, except `loss_scale` lowered to 1.0 on 2026-08-19 after
measuring grad norms 225-335 against clip 10 (every update shrunk ~30x). Upstream
uses loss_scale 10 with clip 10 at batch 32-256, where norms are smaller. Equivalent
alternatives: loss_scale 1 + clip 10 (chosen) or loss_scale 10 + clip 100.

## 7. Inference

- Action decoder: 10 Euler steps from t=1.0 to 0, dt=−0.1, initial sample pure
  Gaussian noise, **no classifier-free guidance** (`world2action.py:213-238`).
- Video generation (when used): 35 steps, guidance 0.0.

**Our status: matches** (10 steps, no CFG).

## 8. Optimization recipe

- FusedAdamW, betas (0.9, 0.99), eps 1e-8, weight_decay 0.1, **lr 1e-4**,
  1000-step warmup then linear decay to 0.2x over 500k, grad clip 10.0,
  bfloat16, batch 32-256 (global), `max_iter=500_000` (a ceiling, not a measured
  requirement; released checkpoints' actual action-decoder step counts are not
  published).

**Our status: same optimizer family/lr/wd/betas; batch and step counts are far
smaller by necessity (single 24 GB GPU, ~4 s per backbone forward).** Warmup scaled
to the achievable step count. See deviations table.

## 9. Data regime

- Bridge: 15 actions @ 5 Hz (2.8 s horizon). LIBERO: 60 @ 20 Hz (2.95 s).
  Ours: 30 @ 10 Hz (3.0 s) — same wall-clock horizon, matched deliberately.
- Upstream trains on full Bridge V2 / LIBERO conversions (size not published).
  Ours: 40 episodes (~6,500 frames), train 0-31, val 32-39.

---

## Current deviations from upstream (keep this list honest)

| #   | Deviation                                                                                  | Why                                                                                    | Risk                                                                                             |
| --- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1   | Future frames are pure noise, not true/generated future                                    | Causal deployment without video generation                                             | The backbone never sees where the scene is going; may weaken the very signal the method exploits |
| 2   | 6-dim absolute joint actions instead of 10-dim relative EEF                                | Different robot (SO-101)                                                               | Low; but makes loss values incomparable to upstream                                              |
| 3   | Min-max normalization + padding mask instead of mean/std + no mask                         | Cleaner; predates reference doc                                                        | Low                                                                                              |
| 4   | loss_scale 1.0 instead of 10.0 (clip 10 kept)                                              | Measured 30x clipping shrinkage at our batch size                                      | Low — mathematically ~equivalent to raising clip                                                 |
| 5   | Effective batch 2-8 instead of 32-256                                                      | Single 24 GB GPU, ~4 s/backbone forward                                                | High gradient noise; interacts badly with randomized sigma                                       |
| 6   | K flow draws per context (K>1)                                                             | Amortize expensive backbone forward                                                    | NOT upstream; introduced 2026-08-19 together with other changes — must be validated in isolation |
| 7   | Fixed eval sigma 4.0 (upstream sweeps the solver schedule)                                 | Single comparable number across runs                                                   | Low for comparison purposes                                                                      |
| 8   | 1-hour training cap                                                                        | Method's own efficiency claim; practical iteration speed                               | Results are lower bounds on achievable quality                                                   |
| 9   | `observed_prefix` VAE input (5 pixels -> 2 latents -> 16) instead of 5 -> 61 pixel padding | Avoid encoding future-only pixels in new online/cache extraction; measured latency win | Representation drift; compare upstream-shaped runs with explicit `legacy_padded_vae`             |

## Experiment log discipline

Before launching a run, note which table rows apply. Change one deviation at a time.
Reference numbers on the held-out 88-window evaluation (degrees, lower better):
smolvla 14.996 · state_repeat 18.862 · mean_action 30.147 ·
best VAM so far 26.121 (fixed sigma 10, cached stride-20 windows, 2026-08-18) ·
proprio-only floor 26.65 (2026-08-19).

## Addendum 2026-08-20 — connector verified at code level (commit e3355db)

Read end-to-end by a subagent with file/line citations; settles the "is the
unpooled context really upstream behavior?" question.

- **No connector exists.** `world2action_model.py:349-350` flattens the
  layer-20 hidden grid `(B, 16, 30, 40, 2048)` with a single `reshape` to
  `(B, 19200, 2048)`. No pooling, temporal slicing, frame selection, or
  resampler anywhere between video DiT and action decoder.
- **All 24 action blocks cross-attend the full context**
  (`world2action_dit.py:917-925`), and K/V are recomputed from all 19,200
  tokens inside every block (`world2action_dit.py:299-302`).
- **Online, uncached, frozen.** One fresh frozen-backbone `denoise()` per
  training step with fresh noise (`world2action_model.py:113-124, 323-350`).
  Features are never cached upstream; our cache is our own efficiency device.
- **Video sigma recipe (for the record; we are deliberately not chasing it):**
  `exp(N(0,1)) * sqrt(state_t)=4`, with a 5% tail loguniform in
  [200, 100000] (`rectified_flow_scheduler.py:44-48`,
  `world2action_model.py:360-378`).
- **Conditioning:** first 2 of 16 latent frames are clean ground truth (the 5
  observed pixel frames); the other 14 are noise. All 16 stay in the context.
- **Attention is fully bidirectional** in the video DiT
  (`text2image_dit.py:353` uses `attn_mask_type="no_mask"`;
  `module/attention.py` defaults `causal=False`). By layer 20 every token
  mixes the whole grid, so token position does not cleanly separate observed
  from imagined content.
- Literature context for the interface design: see
  `video_vam_connector_survey.md` — the unpooled grid is an outlier; modular
  systems compress to 32–512 tokens (VPP's 224-token Video Former is the
  closest precedent, where compression improved both speed and success rate).

## Correction record (2026-08-22, do not re-litigate)

**mimic-video does NOT generate future latents.** It performs ONE forward pass:
clean conditioning latents + noise-filled future slots -> intermediate-layer
features -> action decoder. No solver steps, no denoising rollout, no predicted
future is ever materialized, at training or deployment. Any proposal in this
repo to "give the backbone generated future latents (a few solver steps)" is an
EXTENSION of ours, not a reproduction of mimic, and must be labeled as such.
This was repeatedly stated incorrectly in analysis; the user has corrected it. Treat
the single-forward-pass contract as ground truth about upstream.
