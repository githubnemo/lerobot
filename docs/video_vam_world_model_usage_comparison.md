# World-model-to-policy approaches: comparison and our measurements

Created 2026-08-24. Companion to `video_vam_cube_out_of_box_leaderboard.md` (numbers) and
`mimic_video_reference.md` (upstream fidelity). Dataset for all numbers: cube-out-of-box,
32 train episodes, frozen 88-anchor validation protocol, RMSE in degrees (lower better).
Baselines: state_repeat 18.86, mean_action 30.15.

## The design axes

Every "use a video/world model for control" system chooses along four axes:

1. **Backbone & its pretraining** — what world knowledge is imported.
2. **Action-conditioning interface** — where and how action computation reads the backbone
   (single-layer feature tap, prefix attention, per-layer mixed attention, ...).
3. **Training technique** — what is frozen, what adapts (full FT / LoRA / frozen), and
   whether a video objective is co-trained with the action objective.
4. **Action head** — size, architecture, and whether it starts from an action-pretrained prior.

## The systems

| System                                    | Backbone (pretrain)                                                | Interface                                                                                                         | Training technique                                                         | Action head                                    | Our RMSE                      | Converged?                                                |
| ----------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------- | ----------------------------- | --------------------------------------------------------- |
| Cosmos+World2Action (mimic-video replica) | Cosmos-Predict2 2B (general video)                                 | single forward, clean cond + noise futures, layer-20 tap, pool2 (4,800 tok)                                       | backbone frozen; features cached; decoder from scratch                     | 499M flow-matching DiT, cross-attn every block | **14.51**                     | yes (~24.6k steps, plateau)                               |
| Cosmos+SmolVLA expert (ours)              | Cosmos-Predict2 2B (general video)                                 | same layer-20 pool2 features via 2M adapter (2048->960) into expert prefix K/V                                    | backbone frozen; features cached; expert fine-tuned                        | 98M SmolVLA expert, **action-pretrained**      | **14.14**                     | convergence run in progress; band 14.1-14.7 since step 6k |
| SmolVLA end-to-end                        | SmolVLM VLM (image-text)                                           | VLM prefix K/V                                                                                                    | VLM partially frozen per recipe; expert trained                            | same 98M expert, action-pretrained             | **14.83**                     | yes (29.2k steps; batch 64 replicates 14.82)              |
| FastWAM LoRA                              | Wan-family 6B video model, **jointly action-pretrained on LIBERO** | no single tap: ActionDiT tower interleaved at **every layer** via MoT mixed attention; video K/V prefilled+cached | LoRA r16 on video expert + action tower; **joint video+action loss** (1:1) | ActionDiT tower (per-layer)                    | 20.36                         | no (still descending at 4.2k steps; 40k-step run queued)  |
| VLA-JEPA                                  | V-JEPA 2 encoder (video SSL)                                       | latent future-prediction embedding space                                                                          | full FT (2.8B), world-model + action loss                                  | small action head, chunk 30                    | 19.19                         | no (1k steps; wm_loss never moved)                        |
| Cosmos LoRA co-train Arm A (ours)         | Cosmos-Predict2 2B                                                 | layer-20 tap (pool2) while LoRA adapts backbone                                                                   | joint video+action loss bolted onto modular design                         | World2Action 499M                              | 24.37                         | best mid-run; degrading after                             |
| Random-init controls                      | Cosmos 2B random / SmolVLA vision random                           | as parent                                                                                                         | as parent                                                                  | as parent                                      | worse than pretrained parents | —                                                         |

## What our numbers say about the axes

- **Backbone pretraining pays** (random-init controls worse; general-video Cosmos features
  beat image-text VLM features when the head is held fixed: 14.14 vs 14.83).
- **Interface: a single frozen tap is enough at our scale.** The elaborate per-layer MoT
  (FastWAM) and latent-future objectives (VLA-JEPA) both scored below state_repeat under a
  1-hour-class budget. Caveat: neither was converged; FastWAM convergence run pending.
- **Training technique: freeze the backbone.** Both co-training attempts on the modular
  design (Arms A/B) underperformed frozen features by ~10 deg. Important nuance: FastWAM-style
  co-training is not thereby falsified — its trunk was _pretrained jointly_ with the action
  tower, so co-training does not destabilize a fixed feature interface the way our post-hoc
  LoRA arms did. Our evidence only says: do not bolt co-training onto a modular tap.
- **Action head: pretrained prior beats size.** 98M pretrained expert > 499M from-scratch
  DiT, reaching a better number ~5x faster (crosses SmolVLA plateau ~25 min into training).

## Deployment constraint (see research objective in the diary)

Cached features do not exist at deployment. The measured optimized Cosmos and LTX forwards
remain on the critical path for every chunk. SmolVLA end-to-end is the only currently
real-time recipe; neither frozen world-model path is yet viable at ~10 Hz.

## Bag of tricks: efficient world-model policies

This section is the implementation checklist for turning the best offline recipe into a
real-time policy while preserving its representation and task accuracy. Detailed benchmark
results and primary-source citations live in `video_vam_latency_optimization_plan.md`.

### Semantics-preserving gains to apply

1. **Encode only observed pixels.** Our causal contract has five real frames, yielding two
   conditioning latent frames. The remaining 14 latent slots are prescribed Gaussian
   diffusion noise at sigma; they do not need—and must not receive—pixel-space VAE encoding.
   New extraction defaults to `observed_prefix`: encode 5 -> 2 latents, zero-expand to 16,
   then apply the existing future-noise and conditioning-mask assembly. The 5 -> 61 path is
   retained only as the explicit `legacy_padded_vae` compatibility/debug mode. The prefix
   path has known representation drift from the padded path, so this is an intentional
   optimization/deviation rather than an exact upstream reproduction.
2. **Compile fixed-shape components separately.** Benchmark `torch.compile`/CUDA graphs for
   VAE encode, DiT-to-layer-20, and the expert rather than compiling one monolith. Preserve
   eager fallback and report compile warmup separately from steady-state p50/p90. The
   production `SmolExpertActionDecoder.sample_actions` now auto-selects its serialized
   fixed-shape CUDA graph only for CUDA batch-one `[1, 4800, 2048]` pool2, `[1, 30, 32]`
   action noise, and valid state shapes; all other inputs, explicit opt-outs, and capture
   failures use eager + KV cache. For CPU-streamed LTX, keep provider copies and parameter
   injection eager and compile only fixed compute regions; streaming itself does not imply a
   graph break, but a valid graph still needs an end-to-end speedup and correctness gate.
3. **Cache expert context K/V within each action chunk.** Cosmos context is fixed across all
   action-denoising steps. Project it once per expert layer and reuse it. The production graph
   runner copies fresh noise and every per-layer K/V tensor before each replay, serializes
   calls for a single robot, clones returned actions, and recaptures on shape/device/dtype or
   denoising-step changes. On the RTX 4090, production eager+KV was `141.250/141.520 ms`
   versus graph `86.533/86.544 ms` (median/p90, `1.63x` denoise speedup); including the
   one-time `2.557 ms` KV preparation, the combined medians were `143.807` versus `89.090 ms`
   (`1.61x`). Both observations in the stale-context smoke test had exactly zero graph drift.
4. **Validate fewer action-denoising steps.** The production graph winner's five-step probe
   measured `43.070/43.079 ms` versus the ten-step eager baseline's `129.468 ms` and had
   exact drift `0` against eager five-step inference. Its drift against ten-step eager output
   was RMSE `0.8303`; frozen-protocol action RMSE and rollout smoothness must decide whether
   five steps is acceptable.
5. **Select kernels explicitly.** Compare PyTorch SDPA, Transformer Engine attention, BF16,
   FP8, and compile as independent arms; validate hidden-state cosine error and action RMSE,
   not latency alone. FLUX-mimic discloses post-training backbone quantization but not its
   bit width, calibration, or kernels.

### Deployment scheduling gains

- **Incremental/streaming VAE:** consecutive five-frame windows overlap by four frames. Cache
  temporal encoder state only after verifying the tokenizer receptive field and shifted-window
  equivalence.
- **Asynchronous RTC:** execute the current action chunk while preparing the next, then blend
  or inpaint the committed prefix. Measure action age and chunk-boundary jerk, not just GPU Hz.
- **Pinned/zero-copy input:** overlap camera transfer and preprocessing with current chunk
  execution; keep persistence and video encoding off the control process.

### Deliberately deferred

- Lower spatial resolution changes the representation and requires cache rebuild/retraining.
- VAE/backbone distillation changes the model and is only justified if exact prefix encoding,
  compilation, kernels, quantization, caching, and RTC remain insufficient.

### Current RTX 4090 latency baseline

Five observed 480x640 frames, BF16, Cosmos layer 20, pool2, converged SmolVLA expert:
the historical `legacy_padded_vae` path measured VAE 1.334 s; DiT 0.814 s; 10-step expert
0.155 s; full path 2.33 s. The observed-prefix benchmark measured about 0.119 s for VAE
and 1.107 s for the full path, with known internal representation drift. Prefix mode is
now the default for new online/cache extraction; DiT compilation/quantization remain the
next optimization targets, and expert-only optimization cannot close the real-time gap.

### Measured LTX-2.5 facts

The official LTX-2.5 22B FP8-cast/CPU-streamed path uses five real frames padded to nine,
two clean conditioning latents, sigma-1 noise, and a block-34 tap. On the same RTX 4090,
full 48-block eager extraction measured `1.656819/1.659106 s` p50/p90; an explicit block-0..34 prefix measured `1.233393/1.234206 s` (`1.343x`) with cosine
`1.000000`, max abs `0`, and RMSE `0` against the validated hook/exception path. VAE encode
is ~71 ms; five input frames produce one clean latent while nine produce two, so nine-frame
padding is causal, not removable overhead.

Regional compile was technically valid: live tracing found 48 stable block identities, 84
parameter objects replaced per provider call, and two alternating 0.720-GiB storage slots;
Dynamo explain still produced one 106-op block graph with zero breaks. Whole-block,
stateless-weight, slot-keyed, attention/MLP, default, reduce-overhead, max-autotune, and
full-graph arms all measured about `1.232-1.244 s` end to end and did not beat eager beyond
run noise. The lesson is that CPU streaming does not prohibit regional compile; here BF16
matrix multiplication and SDPA already dominate, so compiling pointwise orchestration adds
no useful throughput.

Compiling only the VAE with `reduce-overhead` reduced VAE p50 from `70.815` to `53.177 ms`
and end-to-end p50/p90 to `1.216091/1.217239 s` (`1.014x` over eager prefix). It passed the
requested cosine gate (`0.999639663`) but had max abs `132`, RMSE `0.179009`, and relative
RMSE `2.6843%` at the hidden tap. It is therefore an explicit
`vae_compile_mode="reduce-overhead"` option, not the exact default, until action RMSE and
rollout checks accept the drift. Cold compile overhead amortizes after roughly 670 calls.

Persistence is mandatory: transformer open measured 42.428 s in the warm benchmark, while
a post-reboot cold construction/open probe measured 57.516 + 49.384 s. Each GPU streaming
slot is 0.720 GiB; retaining all 35 prefix blocks projects to 26.60 GiB allocated on the
23.52 GiB card, so full resident-prefix caching is rejected. One BF16 LTX feature is 18.75
MiB, roughly four times smaller than unpooled Cosmos context; at the existing ~1,560-anchor
stride-3 scale this projects to ~28.56 GiB and ~33.9 min compute-only including cold startup.
See `video_vam_ltx_latency_optimization.md` for commands, stages, and rejected arms.

A tiny 2-16-window real-feature overfit is now runtime-reasonable, but only after replacing
the zero-prompt benchmark tensor with a real Gemma-4 prompt embedding and passing one
real-feature consumer smoke. LTX still takes 1.216 s in its fastest correctness-gated arm before action decoding, so it
is not a 10 Hz deployment path.

## Data-scaling result (2026-08-25)

Same cube-out-of-box split and 88-anchor RMSE protocol. Cosmos uses frozen Predict2-2B
layer-20 pool2 prefix-VAE features + pretrained 98M SmolVLA expert; SmolVLA is end-to-end.

| Train episodes |       Cosmos + expert |           SmolVLA |
| -------------: | --------------------: | ----------------: |
|              4 |     32.81 (converged) |             28.66 |
|              8 |   **20.58** (1 h cap) |             25.24 |
|             16 |   **20.13** (1 h cap) |             25.85 |
|             32 | **13.81** (converged) | 14.82 (converged) |

Conclusion: Cosmos features improve low-data transfer at 8--32 episodes; four episodes are
insufficient for either policy. The 8/16 Cosmos points are budget-matched, not ceilings.
