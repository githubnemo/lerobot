# Video-action latency optimization plan

Status: measured 2026-08-24 on the RTX 4090 with 20 samples after 3 warmups.
This document separates public claims from implementation details that have not been
disclosed by Black Forest Labs or mimic robotics. It is a deployment plan, not a claim
that the FLUX-mimic numbers reproduce on the Cosmos path.

## Publicly verified mimic / FLUX-mimic claims

The primary sources are the [BFL FLUX 3 x mimic announcement](https://bfl.ai/blog/flux-3-mimic),
the [mimic FLUX-mimic announcement](https://www.mimicrobotics.com/blog/introducing-flux-mimic),
and the pinned public [mimic-video repository](https://github.com/mimic-video/mimic-video).

- **Partial video denoising:** FLUX-mimic does not generate a video rollout at
  inference time. Its action decoder reads an internal representation of the
  predicted future, so one action chunk costs one video-backbone forward rather
  than a pixel-video rollout. This already matches the intended Cosmos extractor
  contract: one frozen high-noise forward, hidden layer 20, then action decoding.
- **Post-training backbone quantization:** mimic says it aggressively quantizes
  the video backbone for deployment and validates the compressed checkpoint
  against the original. The quantizer, calibration data, precision, kernels, and
  acceptance thresholds are not disclosed.
- **Cached context and adaptive action denoising:** mimic says FLUX-mimic reuses
  video-backbone context during action decoding and can skip denoising steps when
  consecutive velocity predictions are similar. The cache key, skip threshold,
  exact schedule, and decoder implementation are not public.
- **RTC overlap:** Real-Time Chunking overlaps prediction with execution while
  preserving trajectory smoothness across chunks. This is a deployment/control
  scheduling optimization, not a change to the frozen representation.
- **Low-latency IPC:** mimic says its `mimic-ipc` middleware moves observations and
  actions between sensors, model, and actuators with minimal latency and jitter.
  The public post does not disclose a wire format or establish that the transport
  is literally zero-copy; that mechanism remains an implementation detail to
  verify before reproducing the claim.
- **Smaller/fewer-layer backbone:** BFL states that better representations permit
  a smaller backbone and that backbone depth is a dominant latency driver. It does
  not publish the FLUX-mimic layer count, width, pruning recipe, or the accuracy
  trade-off used for deployment.

### Exact disclosed latency and hardware

- BFL reports **less than 80 ms from input to world representation** on **one
  NVIDIA RTX 5090** for its optimized FLUX-mimic backbone. The <80 ms figure is
  not a full sensor-to-actuator measurement.
- BFL reports **101 ms end-to-end reaction time** for the complete self-contained
  deployment stack, including action decoding, sensor/model/actuator paths, and
  real-time chunking. These are vendor-reported deployment figures, not a public
  reproducible benchmark breakdown.
- For the public, non-FLUX mimic-video implementation, a maintainer reported **470
  ms on an H200** and **1.3 s on an RTX 4090** in [mimic-video issue #1](https://github.com/mimic-video/mimic-video/issues/1).
  The issue describes these as model inference latency figures and does not give
  a stage-by-stage breakdown. They are not interchangeable with the FLUX-mimic
  <80 ms representation figure.

### What is not disclosed

No public FLUX-mimic artifact specifies the exact quantization recipe, CUDA-graph
capture strategy, attention kernel/backend, context/KV layout, adaptive-step
thresholds, action-denoising count, decoder depth, IPC copy behavior, or chunk
horizon used for the latency claims. We should therefore record these as unknowns,
not infer them from the public numbers or from similarly named optimizations in
other diffusion systems.

## Mapping to this Cosmos path

The pinned extractor is the vendored `mimic-video` Cosmos-Predict2 closure at
commit `e3355dbc93132b576c02f920a59b4fc18a4f5906`. Its fixed contract is:

```text
5 RGB frames at 480x640
  -> 61-frame temporal padding
  -> Cosmos VAE, 16 latent frames
  -> one BF16 video DiT forward through hidden layer 20
  -> [B, 19200, 2048] hidden tokens
  -> production pool2, [B, 4800, 2048]
  -> Smol expert, [B, 30, 6], ten Euler flow steps
```

The mapping is intentionally conservative:

- No video rollout is added. The action decoder consumes the extracted hidden
  state, matching the partial-denoising claim at the architectural level.
- The current vendored Cosmos attention constructor accepts `minimal_a2a`, `torch`,
  and `transformer_engine`. The benchmark exposes these exact names through
  `--attention-backend` and records the selection. `flash_attn_no_cp` is not
  exposed because the current video DiT constructor rejects it; the
  `minimal_a2a` implementation uses the pinned attention helper's PyTorch SDPA
  dispatch.
- Transformer Engine remains required by the vendored Cosmos modules for their
  RMSNorm/fused rotary path even when the explicit attention dispatch is `torch`.
  Selecting another attention backend is a runtime experiment and needs GPU
  validation; the default remains `minimal_a2a`.
- The vendored graph seam is currently an explicit unsupported
  `_compat.create_cuda_graph` stub. The benchmark therefore exposes
  `--arm cuda_graph` only as an opt-in unsupported result; it is not silently
  measured as baseline and is not part of `--arm all`.
- The Smol expert already prepares the shared Cosmos/state prefix once per action
  sample. The new `context_cache` path additionally materializes each expert
  layer's exact prefix K/V once and reuses it across all Euler denoising calls.
  A CPU correctness test compares cached and uncached outputs with identical
  inputs and noise. The cache is valid only for the same state/context pair.
- The new `decoder_sweep` path measures the fixed **10/5/3/2/1** Euler-step
  alternatives with synchronized timings and output drift relative to ten steps.
  It does not claim adaptive skipping or RTC.

## Benchmark arms

`scripts/video_vam/benchmark_cosmos_extraction.py` supports the following explicit
arms:

- `baseline`: eager frozen Cosmos DiT in BF16.
- `compile`: fullgraph `torch.compile(mode="reduce-overhead")` over the compile-friendly
  native-PyTorch DiT variant.
- `compile_max`: fullgraph `torch.compile(mode="max-autotune")` over the same variant;
  this is the current correctness-gated winner.
- `compile_regional` and `compile_regional_max`: fullgraph compilation of each repeated
  transformer block, using `default` and `max-autotune-no-cudagraphs` respectively.
- `fp8`: the existing Transformer Engine delayed-scaling FP8 autocast arm.
- `vae_prefix`: historical paired 61-frame versus five-frame causal VAE benchmark with
  latent, hidden-grid, pool2, and optional action equivalence checks. The production-facing
  default is now the explicit `observed_prefix` mode; use `legacy_padded_vae` when reproducing
  the old 61-frame baseline or upstream-shaped inputs.
- `int8`: explicit rejected result because no validated CUDA int8 path exists for
  this Transformer Engine DiT.
- `cuda_graph`: manual fixed-shape DiT capture/replay, bypassing Dynamo; validated
  for bitwise output equivalence but retained as an explicit arm because it did not
  reduce latency on this GPU.
- `decoder_sweep`: extraction plus the Smol action-decoder 10/5/3/2/1 timing and
  drift report.
- `context_cache`: extraction plus cached/uncached Smol per-layer prefix-K/V
  timings and exact output-drift metrics.

### `expert_compile` decoder benchmark

The explicit `expert_compile` arm isolates the VLM-free SmolVLA expert from Cosmos
extraction using one fixed `[1, 4800, 2048]` pool2 context and the corresponding
state from the benchmark sample. It loads the converged
`/home/anton/.cache/video-vam/runs/smolexpert-cosmos/train-converge/best.safetensors`
through `SmolExpertActionDecoder.from_training_checkpoint` with the canonical
train-split normalizer. Per-layer prefix K/V preparation is timed once per observation
and kept outside the measured denoising region.

The arm reports 20 synchronized iterations after three warmups for the current eager
KV-cache path, `torch.compile(mode="reduce-overhead")`,
`torch.compile(mode="max-autotune")`, and a manual CUDA graph containing the full
10-step Euler loop. Each candidate uses the same fixed noise and is correctness-gated
against eager with `max_abs < 1e-2`; exact max/mean/RMSE drift is recorded. The fastest
passing candidate is also measured at five denoising steps. CUDA graph capture is
attempted because the context, state, cache, noise, and loop lengths are fixed; if
capture is not feasible on the runtime, the report records the exception and keeps the
other variants. Results are intentionally left to the benchmark run below.

The unattended `all` expansion is:

```text
baseline, compile, fp8, vae_prefix
```

`vae_prefix` remains a diagnostic comparison arm and requires the explicit
`legacy_padded_vae` baseline. Its numerical validation failed, so `observed_prefix` is not
claimed to be representation-equivalent to the 61-frame path; it is the intentional default
for new online extraction and cache builds because the task accepts this internal drift.

This keeps the queued benchmark from launching rejected arms while leaving every
arm directly selectable. The existing queued invocation remains valid without edits:

```bash
.venv/bin/python scripts/video_vam/benchmark_cosmos_extraction.py \
  --arm all --iterations 20 --warmups 3 --cudnn-benchmark \
  --output-dir /home/anton/.cache/video-vam/cosmos-extraction-benchmark
```

## Prioritized implementation sequence

1. **Compile-friendly DiT, then context/KV cache.** The compile-friendly
   `compile_max` arm is implemented, CPU-checked, and correctness-gated on the
   fixed GPU sample. Manual CUDA graph capture is also validated but does not win
   on latency; next validate the Smol per-layer cache's GPU speedup and numerical
   drift, then compare the supported attention dispatches on the same sample.
2. **Quantization and token sweeps.** Keep FP8 behind the existing arm until its
   output drift and latency are measured. A post-training quantization arm needs a
   real quantizer and calibration/equivalence protocol. Token pooling/slicing is a
   representation change, so report downstream drift and task metrics rather than
   calling it a free latency optimization.
3. **Denoising and RTC.** Use the decoder step sweep to choose a quality/latency
   point first. Adaptive velocity-based skipping and asynchronous chunk overlap
   require a control-loop contract, fresh-observation policy, and trajectory
   continuity tests; they are not safe to infer from a synchronous benchmark.
4. **Distillation.** Only after the representation, decoder, quantization, and
   deployment scheduling are stable. Distillation changes the learned function and
   must be evaluated as a new model, not folded into a kernel-latency result.

## RTX 4090 measurements (2026-08-24)

The benchmark ran on one NVIDIA GeForce RTX 4090 with PyTorch 2.11.0+cu128,
CUDA 12.8, `minimal_a2a` attention, cuDNN benchmark enabled, and 20 measured
iterations after 3 warmups. The decoder used the converged
`train-converge/best.safetensors` checkpoint and the corrected custom Smol expert
loader. Reports are preserved under:

- `/home/anton/.cache/video-vam/cosmos-extraction-benchmark-20260824-prefix`
- `/home/anton/.cache/video-vam/cosmos-extraction-benchmark-20260824-prefix-nocudnn`
- `/home/anton/.cache/video-vam/cosmos-extraction-benchmark-20260824-prefix-channels-last`
- `/home/anton/.cache/video-vam/cosmos-attention-torch-20260824`

### Baseline and dense arms

The current 61-frame path measured these median/p90 times in milliseconds:

- Baseline: VAE `1334.43/1334.89`, DiT through layer 20 `813.52/814.70`,
  extraction `2176.04/2177.24`, full extraction plus 10-step decoder
  `2343.28/2346.50`.
- `torch.compile` around the DiT: VAE `1335.59/1336.00`, DiT
  `814.74/816.93`, full `2346.12/2349.63`; no useful speedup.
- Transformer Engine FP8 autocast: VAE `1335.95/1336.85`, DiT
  `816.77/817.35`, full `2338.72/2341.37`; it does not accelerate the 3D VAE,
  and the total is within run-to-run noise of baseline. FP8 is not assumed for
  the VAE.

The explicit `torch` attention backend also loaded and measured `813.24/814.72`
ms for DiT layer 20. The `transformer_engine` attention backend was attempted but
could not load this checkpoint because the constructed TE operator requires the
missing `blocks.*.self_attn.attn_op._extra_state` metadata. It remains unvalidated,
not a claimed result.

### Causal VAE prefix experiment

The vendored tokenizer is causal: `CausalConv3d` left-pads temporal kernels and
never right-pads; the streaming encoder processes one initial frame followed by
chunks and emits `1 + (T - 1) // 4` latents. Therefore five frames are a valid
input and emit two latent frames, while 61 frames emit the production 16-frame
latent tensor.

In the paired `vae_prefix` run, current padding versus the five observed frames
measured:

- VAE encode: `1336.79/1337.22` versus `118.92/119.03` ms, an `11.24x` median
  VAE speedup.
- Extraction through pool2: `2159.91/2161.46` versus `942.09/942.81` ms,
  `2.29x` median speedup.
- Full extraction plus the same seeded 10-step decoder: `2320.38/2326.10`
  versus `1106.52/1109.14` ms, `2.10x` median speedup.

The first two latent shapes matched (`[1, 16, 2, 60, 80]`), but values did not
meet the strict acceptance limits: max/mean/RMSE drift was
`0.03125/0.000402/0.001532` and cosine was `0.9999968`. More importantly, using
the same full 16-frame conditioning shape and identical seeded future noise gave:

- Layer-20 hidden-grid max/mean/RMSE drift `129.5/0.03406/0.07302`, cosine
  `0.9998351`.
- Production pool2 feature max/mean/RMSE drift `23.5/0.01928/0.03659`, cosine
  `0.9999413`.
- Final `[1, 30, 6]` action max/mean/RMSE drift
  `0.21076/0.06085/0.07967`, cosine `0.9999989`.

The strict limits were latent max/RMSE `1e-4/2e-5` with cosine at least
`0.9999999`, hidden/pool2 max/RMSE `1e-2/2e-3` with cosine at least `0.999999`,
and actions max/RMSE `1e-4/2e-5` with cosine at least `0.9999999`. All four
comparisons failed, so prefix encoding is not claimed to be representation-equivalent to
the padded path. Matching only the first two latent frames would have been insufficient;
the hidden, pool2, and action comparisons quantify the accepted internal drift. The
likely cause is shape-dependent numerical behavior in the VAE's cuDNN/streaming
convolution path, which is amplified by the DiT and decoder even though mean drift is
small.

With cuDNN benchmark disabled, prefix VAE time was `126.26/126.41` ms and the
same comparison still failed (latent max/RMSE `0.03125/0.001527`, layer-20 max
`46.0`, pool2 max `11.5`). `channels_last_3d` was compatible and reduced VAE time
to `110.06/110.13` ms, but also failed equivalence (latent max/RMSE
`0.03125/0.001135`, layer-20 max `46.0`, pool2 max `11.5`). These are useful
benchmark observations that quantify the drift; they are not equivalence claims.

### Current implementation decision (2026-08-24)

The measured representation drift is accepted for the internal causal path. New online
extraction, sample extraction, and cache builds therefore default to `observed_prefix`:
encode only the five real pixel frames to two latents, zero-expand to the full 16-latent
state, and then perform the existing future-noise and conditioning-mask assembly. The
explicit `legacy_padded_vae` mode restores 5 -> 61 pixel padding for compatibility and
upstream-shaped debugging. New manifests and artifact sidecars record the selected mode;
pre-mode manifests remain readable as legacy provenance and are never relabeled as prefix.
No existing cache tensors are modified by this change.

### VAE compile probe

`torch.compile(extractor.tokenizer.encode, mode="reduce-overhead")` is a real API,
but it is not a safe arm for this stateful, shape-varying tokenizer. The padded
shape spent `56.7 s` compiling and then ran at about `1.04 s`; the subsequent
five-frame shape exhausted the 24 GB GPU (`900 MiB` allocation failure, about
`22.65 GiB` allocated in private CUDA-graph pools). No compile arm or production
setting was added.

The remaining validated bottleneck is the DiT layer-20 forward at about `814 ms`
on the current contract. Prefix encoding would make the DiT dominant, but its
semantic drift prevents using that apparent `2.10x` full-path result in production.
Resolution reduction and VAE distillation remain later, semantic-changing model
options; they were not treated as direct kernel comparisons here.

## `torch.compile` iteration (2026-08-25)

The compile experiments used the production `observed_prefix` contract, BF16,
`minimal_a2a` eager attention for the reference, one fixed seed/noise tensor, and
20 measured iterations after 3 warmups on the RTX 4090. Correctness is reported
against the separately loaded eager-TE normalization/rotary reference using an
FP64 cosine reduction; the gate is strictly `cosine > 0.9999` on layer-20 `pool2`
features. Decoder timing was disabled so these numbers isolate VAE + DiT
extraction; the extraction and full columns are therefore equal.

- **Eager baseline:** DiT `810.48/811.81 ms` median/p90; extraction `956.05/957.69 ms`;
  full `956.06/957.70 ms`; `31.38/31.32` 30-step chunk Hz; correctness `1.0`,
  max abs `0.0`.
- **`compile_max`:** first-forward compile `38.47 s`; DiT `695.26/695.37 ms`;
  extraction `840.65/840.92 ms`; full `840.66/840.93 ms`; `35.69/35.67` Hz;
  cosine `0.9999038`, max abs `16`, gate **passed**. This is a `1.17x` DiT
  speedup and `1.14x` end-to-end extraction speedup.
- **`compile` (`reduce-overhead`):** first-forward compile `31.37 s`; DiT
  `709.63/710.45 ms`; full `857.28/858.19 ms`; `34.99/34.96` Hz; cosine
  `0.9998994`, max abs `13`, gate **failed** and DiT speedup was only `1.14x`.
  This run included `TORCH_LOGS=graph_breaks`; no graph-break diagnostics were
  emitted and fullgraph compilation completed.
- **`compile_regional`:** first-forward compile `4.97 s`; DiT `711.91/764.11 ms`;
  full `859.82/938.90 ms`; `34.89/31.95` Hz; cosine `0.9998907`, max abs `36`,
  gate **failed**.
- **`compile_regional_max`:** first-forward compile `18.23 s`; DiT `691.71/692.77 ms`;
  full `837.69/838.79 ms`; `35.81/35.77` Hz; cosine `0.9998971`, max abs `18`,
  gate **failed** despite the raw latency win.
- **`cuda_graph`:** manual capture `2.58 s`; DiT `808.83/809.82 ms`; full
  `954.79/955.92 ms`; `31.42/31.38` Hz; cosine `1.0`, max abs `0.0`, gate
  **passed**, but there was no meaningful speedup versus eager.

The implementation that made `compile_max` pass the gate keeps native RMSNorm
semantics behind a `torch.library` custom op so Inductor cannot change its
rounding, while the native RoPE replacement keeps its FP32 multiply-accumulate
before the final BF16 cast. The extractor exposes this as the opt-in
`CosmosPredict2ExtractorConfig.compile_friendly=True` variant; the benchmark
selects it automatically for the four compile arms. Plain linear layers already
remain ordinary `torch.nn.Linear` modules with checkpoint weights preserved. The historical `fp8` arm
therefore wrapped ordinary Linear execution in FP8 autocast without converting its GEMMs to TE FP8
Linear modules; it did not provide real Linear-FP8 acceleration.

The 2–4x path is the observed-only `state_t=2` Cosmos DiT contract: 2,400 tokens instead of 19,200,
followed by a pool2 context of 600 tokens. This changes the representation, so the Smol expert must be
retrained on the new 600-token pool2 context before it can be used for rollout.

The winning arm is therefore `compile_max` for the DiT stage: it clears the
correctness gate and exceeds the `1.15x` target. It is not a `1.15x` whole-pipeline
win because the uncompiled prefix VAE remains about `119 ms` of the measured
`841 ms` extraction.

## Smol expert compile measurements (2026-08-25)

The fixed-context `expert_compile` run used one `[1, 4800, 2048]` BF16 pool2
context, one state, the converged `train-converge/best.safetensors` artifact, and
20 synchronized samples after three warmups. Prefix K/V preparation was performed
once outside the denoise runners and cost `24.7077 ms`. Timings below are median/p90
in milliseconds; speedups are relative to the eager ten-step KV-cache baseline
(`129.4679/134.7666 ms`).

- **Eager + KV cache:** `129.4679/134.7666 ms`; exact reference drift `0`.
- **`torch.compile(mode="reduce-overhead")`:** `71.6703/71.6874 ms`, `1.81x`;
  compile setup `132.3353 s`; drift max/mean/RMSE
  `0.0627441/0.0139809/0.0193220`, gate **failed**.
- **`torch.compile(mode="max-autotune")`:** `26.0173/28.2921 ms`, `4.98x`;
  compile setup `180.3464 s`; drift max/mean/RMSE
  `0.0671005/0.0143816/0.0191992`, gate **failed**.
- **Manual CUDA graph, full ten-step loop:** `86.1317/86.1464 ms`, `1.50x`;
  capture `0.5599 s`; exact drift `0`, gate **passed**. Precomputing the fixed
  float32 timestep tensor on-device avoided the earlier CPU-to-CUDA capture failure.

The correctness-gated winner is therefore the manual CUDA graph. Its five-step
variant measured `43.0695/43.0786 ms` (`3.01x` versus the ten-step eager baseline),
with exact drift `0` against eager five-step inference. As expected for a different
Euler discretization, its drift against the ten-step eager output was
max/mean/RMSE `3.52386/0.606619/0.830310`; this is reported as a step-count quality
trade-off, not as a compile-equivalence failure. The JSON and Markdown reports are
preserved under `/home/anton/.cache/video-vam/runs/expert-compile-bench/`.

### Production `sample_actions` validation

The public inference method was then benchmarked with the same fixed context/state/noise,
20 synchronized calls after three warmups, and fresh graph input copies on every call.
Eager + KV measured `141.2501/141.5199 ms`; production CUDA graph measured
`86.5330/86.5441 ms`, a `1.6323x` median speedup. Adding the one-time `2.5573 ms`
KV preparation gave combined medians `143.8074 ms` versus `89.0903 ms` (`1.6142x`).
The graph capture cost was `0.5500 s` on the first call and is exposed through
`last_cuda_graph_capture_seconds`; subsequent calls reuse the graph.

A focused CUDA smoke test used two different contexts, states, and noise tensors. Each
graph output matched its corresponding eager output exactly, the two eager outputs differed,
and changing the denoising step count triggered recapture. Returned graph actions are
cloned so a later replay cannot mutate a caller-held result. Capture/replay is serialized
for the single-robot inference case; runtime failures emit one warning and permanently
fall back to eager KV-cache inference for that decoder instance.

## Deferred blockers

- **CUDA graphs:** the manual fixed-shape wrapper is validated for this checkpoint
  and hidden-state early-return path, but measured `~1.00x` versus eager. It remains
  available as an explicit arm rather than a production default.
- **Attention equivalence:** the explicit `torch` backend measured at
  `813.24/814.72` ms for DiT layer 20, but the Transformer Engine attention
  constructor could not load the checkpoint because its self-attention metadata is
  absent. No TE attention result or cross-backend numerical equivalence claim is
  made.
- **FLUX-specific quantization and smaller backbone:** proprietary details and
  checkpoints are unavailable, so they cannot be reproduced faithfully here.
- **Adaptive denoising/RTC/IPC:** these affect scheduling, observation freshness,
  and actuator timing beyond this extractor process. They need a rollout-side
  protocol and must not be represented as an extractor-only speedup.
- **Cosmos context cache across video denoising:** this benchmark has one Cosmos
  forward and does not implement a multi-step video denoising loop. There is no
  reusable video-backbone trajectory context to cache in the current path; adding
  one would be a broader inference redesign, not a safe benchmark toggle.
