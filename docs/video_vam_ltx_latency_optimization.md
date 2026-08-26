# LTX-2.5 extraction latency optimization

Status: measured 2026-08-25 on `abakus` (RTX 4090, PyTorch 2.11.0+cu128).
The exact default winner is persistent CPU-offload FP8-cast execution through an explicit block-0..34 prefix. It returns the bit-identical BF16 `[1, 2400, 4096]` feature and reduces steady p50 from 1.656 s to 1.233 s (`1.34x`). An opt-in `reduce-overhead` VAE compile lowers the best correctness-gated end-to-end p50 further to 1.216091 s (`1.014x` over eager prefix), but introduces measured hidden drift and is not the default.

## Fixed contract

- Checkpoint: `LTX-2.5-22B-distilled`, source commit `400fd31054597515f47125691032c04b1c3ee24e`, model revision `6c7e5e573ac1667efc83407806fe9b0b93730e60`.
- Input: one causal `[1, 3, 5, 480, 640]` window at 10 Hz.
- VAE: repeat the last observed frame to nine frames, producing two clean latent frames.
- State: eight `15x20` latent frames; two clean conditioning frames and six deterministic sigma-1 noise frames.
- Tap: block 34 (zero-based), output `[1, 2400, 4096]` BF16 tokens.
- Runtime: official `ltx-core` FP8-cast policy, CPU block streaming, BF16 VAE/hidden states.
- Attention: `SDPA[CUDNN_ATTENTION>FLASH_ATTENTION>EFFICIENT_ATTENTION>MATH]`.
- Dataset/input: `hubnemo/cube_out_of_box_dataset@243370c3c08bcbd860133c4a0d658ea7c1d2e77e`, episode 0, frame 4, noise seed `4169759595`.
- Prompt: explicit zero-prompt benchmark arm only. Production must use a reusable real Gemma-4 embedding artifact.

## Exact commands

Real smoke:

```bash
source scripts/video_vam/cosmos_cuda_env.sh
scripts/video_vam/run_smoke_test_ltx_extractor.sh \
  --checkpoint /home/anton/.cache/video-vam/ltx-2.5-models/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --video-vae /home/anton/.cache/video-vam/ltx-2.5-models/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  --allow-zero-prompt
```

Repeated benchmark:

```bash
source scripts/video_vam/cosmos_cuda_env.sh
.venv/bin/python scripts/video_vam/benchmark_ltx_extraction.py \
  --allow-zero-prompt --iterations 10 --warmups 1 --vae-iterations 10 --overwrite

# Isolated VAE compile modes and Dynamo explain:
.venv/bin/python scripts/video_vam/benchmark_ltx_extraction.py \
  --allow-zero-prompt --iterations 10 --warmups 1 --vae-iterations 10 \
  --try-compile --overwrite

# Opt-in end-to-end winner (compare with the first command):
.venv/bin/python scripts/video_vam/benchmark_ltx_extraction.py \
  --allow-zero-prompt --iterations 10 --warmups 2 --vae-iterations 10 \
  --vae-compile-mode reduce-overhead --overwrite
```

The benchmark synchronizes CUDA at stage boundaries. JSON and Markdown artifacts are written under `~/.cache/video-vam/ltx-extraction-benchmark/`.

## Load, warmup, and memory

- Real-smoke first `extract` call: `50.488 s`, peak `5.17 GiB` allocated / `6.85 GiB` reserved.
- Persistent transformer open in the successful benchmark: `42.428 s`, peak `5.15 GiB` allocated / `6.67 GiB` reserved.
- First full-depth warmup: `2.925 s`; first early-exit warmup: `1.233 s`.
- After a reboot with cold file caches, extractor construction (VAE/configuration plus full checkpoint/VAE hashing) took `57.516 s`; transformer open took `49.384 s`.
- Steady eager and early-exit runs both peaked at `3.40 GiB` allocated / `6.83 GiB` reserved.

Construction/open are one-time per-process costs only when the transformer remains open. Rebuilding or rehashing the 42 GB checkpoint per window is not viable.

## Baseline and winner

| Arm                                                | Blocks |  p50 (s) |  p90 (s) |                                 Speedup |      Cosine | Max abs |     RMSE |
| -------------------------------------------------- | -----: | -------: | -------: | --------------------------------------: | ----------: | ------: | -------: |
| Full eager FP8/CPU offload                         |   0-47 | 1.656819 | 1.659106 |                                  1.000x |           - |       - |        - |
| Explicit eager prefix through block 34             |   0-34 | 1.233393 | 1.234206 |                                  1.343x |    1.000000 |       0 |        0 |
| Explicit prefix + compiled VAE (`reduce-overhead`) |   0-34 | 1.216091 | 1.217239 | 1.362x vs full / 1.014x vs eager prefix | 0.999639663 |     132 | 0.179009 |

The eager-prefix gate uses identical RGB, prompt tensor, noise seed, weights, offload policy, and BF16 output and is bit-identical. The compiled-VAE arm also clears the requested cosine `>=0.999` gate, but its hidden reference RMS is `6.668734`, relative RMSE is `0.026843`, and max absolute error is `132`; it therefore remains explicit opt-in pending action-level validation.

## Stage breakdown

Provider `get` is provider-call wall time, including CPU source lookup, H2D enqueue/ordering, and provider waits. It is not an isolated PCIe bandwidth measurement.

| Stage                     | Eager p50 (ms) | Eager p90 (ms) | Early-exit p50 (ms) | Early-exit p90 (ms) |
| ------------------------- | -------------: | -------------: | ------------------: | ------------------: |
| Input preprocessing + H2D |          1.951 |          2.160 |               1.900 |               1.952 |
| VAE encode                |         71.159 |         71.192 |              71.213 |              71.226 |
| Noise + latent assembly   |          6.418 |          6.483 |               6.501 |               6.621 |
| Prompt context prep       |          0.036 |          0.038 |               0.038 |               0.039 |
| Transformer to tap        |       1576.502 |       1579.121 |            1153.287 |            1154.223 |
| Provider block `get`      |         44.135 |         44.424 |              32.102 |              32.584 |
| Flatten + reduction       |          0.155 |          0.284 |               0.141 |               0.153 |
| Total                     |       1656.819 |       1659.106 |            1233.393 |            1234.206 |

Early exit accounts for almost all of the gain. The compiled VAE saves about 17.5 ms; transformer regional compilation did not reduce the remaining 1.15 s.

## Explicit prefix semantics

The previous validated path registered a forward hook on block 34 and raised an internal exception after the hook captured `TransformerArgs.x`. The production path now performs the same official eager preparation, materializes the same empty perturbation config, applies the official per-block input processor, and invokes blocks 0 through 34 inclusive. It returns block 34's pre-output-projection `x` directly. On the fixed real input it ran the same 35 provider loads and was bit-identical (`cosine 1`, max abs `0`, RMSE `0`) to the hook/exception path. If the pinned runtime no longer exposes the expected model, block list, preprocessor, and processor, the runner falls back to the validated hook/exception path.

## Streamed identities and compile boundary

Pinned official source `400fd31054597515f47125691032c04b1c3ee24e` has 48 distinct stable `BasicAVTransformerBlock` module identities. The provider owns two 773,349,760-byte GPU slots. Before each block, its hook calls `WeightsProvider.get(index)`, which evicts the oldest slot, copies one pinned contiguous CPU buffer into a newly carved GPU view, and replaces all 84 block `Parameter` objects with new parameters over those views. Live tracing showed block 2 reused block 0's storage address while the Python block identity stayed different. The post-hook records a compute-done event before slot reuse.

CPU streaming therefore does **not** intrinsically invalidate regional compilation. Keeping provider lookup, copies, parameter injection, and per-block orchestration eager leaves fixed-shape block compute compilable. Dynamo explain on a prepared block reported one 106-op graph, zero graph breaks, and no break reasons; `fullgraph=True` also executed. A functional arm made all 84 streamed tensors explicit graph inputs and likewise produced one graph with zero breaks. The limiting result is performance and numerical drift, not a superficial full-model graph break.

## Transformer compile arms

All steady values below are ten measured end-to-end extractions after two warmups unless marked diagnostic. First-call overhead subtracts the same-process eager-prefix call and includes lazy Inductor work. The arms used fixed RGB, zero prompt, noise seed `4169759595`, FP8-cast weights, two streamed GPU slots, and blocks 0..34.

| Arm                                                  |     Total p50/p90 (s) | Transformer p50 (s) | First call / estimated overhead (s) | Peak alloc/reserved (GiB) | Result                                              |
| ---------------------------------------------------- | --------------------: | ------------------: | ----------------------------------: | ------------------------: | --------------------------------------------------- |
| Whole block, default                                 |   1.233029 / 1.233686 |            1.152887 |                       7.536 / 6.302 |               3.43 / 6.83 | no speedup; max abs 100, RMSE 0.183064              |
| Functional weights, default                          |   1.232452 / 1.232908 |            1.152593 |                       6.998 / 5.764 |               3.43 / 6.83 | 0.08% apparent gain, below run noise; same drift    |
| Functional weights, reduce-overhead                  |   1.235248 / 1.236620 |            1.155187 |                       7.991 / 6.756 |               3.43 / 4.48 | slower; 1,156 non-static CUDA-graph inputs recorded |
| Functional weights, max-autotune                     |   1.235125 / 1.236542 |            1.155351 |                     31.372 / 30.137 |               3.43 / 4.42 | slower; max abs 112, RMSE 0.180931                  |
| Two slot-keyed functional callables, reduce-overhead |   1.234543 / 1.234743 |            1.154716 |                       6.430 / 5.194 |               3.65 / 4.69 | slower; slot specialization did not help            |
| Attention + MLP only, default                        |   1.233320 / 1.234799 |            1.153353 |                       7.612 / 6.377 |               3.43 / 6.83 | no speedup; 3 graphs, max abs 88, RMSE 0.125912     |
| Whole block, `fullgraph=True` diagnostic             | 1.243617 (one sample) |                 n/a |                    5.048 first call |               3.39 / 6.83 | graph succeeds, but is slower                       |

The default whole-block and functional arms each captured one graph; the regional arm captured three. Compiling a bound streamed block, passing weights statelessly, and keying two callables to the alternating slots all converge on essentially eager runtime. The heavy work remains external BF16 matrix multiplication and SDPA; Inductor's pointwise fusion and launch reduction do not move the end-to-end median. `reduce-overhead`/`max-autotune` add CUDA-graph handling for a very large set of changing weight inputs and are slightly slower. No transformer compile arm is retained in production.

The transformer experiments' original float32 cosine reduction rounded to `1.0`, but max-absolute and RMSE exposed non-bit-exact output. The benchmark now computes cosine and RMS metrics in float64 to avoid that saturation. Since none of these arms produced a meaningful latency win, their drift is an additional rejection reason rather than a trade accepted by production.

## VAE compile winner

Isolated ten-sample VAE results were eager `70.815 ms` p50 versus default compile `53.499 ms`, reduce-overhead `53.177 ms`, and max-autotune `53.255 ms`. All compiled modes had VAE cosine about `0.999888`, max abs `0.086-0.090`, and RMSE `0.01109`. Observed first-call times were `18.293`, `11.833`, and `96.096 s`, respectively; max-autotune has no steady advantage.

The end-to-end reduce-overhead arm measured `1.216091/1.217239 s` p50/p90 over ten samples, with VAE p50 `53.589 ms` and transformer p50 `1.153725 s`. Peak allocation/reservation was `3.42/3.89 GiB`. Against the validated eager-prefix hidden output, cosine was `0.999639663`, max abs `132`, RMSE `0.179009`, and relative RMSE `2.6843%`. The isolated cold first-call overhead implies roughly 670 calls to amortize compilation at a 17.6 ms saving; persistent long runs clear that threshold, short probes do not.

Production exposes this only as `LTXExtractorConfig(vae_compile_mode="reduce-overhead")`; `None` remains the exact eager default. Provenance records the selected mode. Action RMSE and rollout checks are still required before enabling it on a robot.

## Persistence and resident-prefix result

Persistence is now the default real-backend lifecycle. The first extraction lazily opens the transformer, subsequent calls reuse it, and `close()` or context-manager exit releases it.

The official streaming pool uses two GPU slots. Each slot is `773,349,760` bytes (`0.720 GiB`). Retaining blocks 0-34 needs 35 slots and projects to `26.60 GiB` allocated on the `23.52 GiB` 4090, before workspace headroom: projected headroom is `-3.09 GiB`. Full prefix residency is rejected.

The first resident-prefix implementation also built a second streamed model while the baseline remained open. It emitted only startup warnings in `logs/ltx25-benchmark-4.log`; the host became unreachable and the session disappeared after reboot. It produced no latency/correctness result and is preserved as a rejected operational arm. Bounded extra slots do not help sequential repeated passes under the provider's LRU policy: all 35 prefix blocks must fit to avoid reloading block 0 on the next pass.

## VAE causality and rejected arms

- Five real frames encode to `[1, 128, 1, 15, 20]`; nine frames encode to `[1, 128, 2, 15, 20]`. The two clean conditioning latents are part of the causal contract, so five-frame VAE input is rejected.
- Contiguous, `channels_last_3d`, `cudnn.benchmark`, and combined VAE arms were bit-identical. Their p50 values were `71.033`, `71.013`, `71.020`, and `71.029 ms`; none wins.
- Transformer `torch.compile` was exhaustively measured at whole-block, stateless-weight, two-slot, attention/MLP, and full-graph boundaries. Streaming can stay eager around a valid fixed compute graph, but no arm improved end-to-end latency beyond run noise and all introduced hidden drift.
- Expert/adapter micro-tuning is deferred because the 1.153 s transformer dominates the measured path.
- Full 48-block eager execution is retained only as the correctness baseline.

## Disk and cache-duration implications

Model files are `42,018,190,584 + 1,452,269,922` bytes (about `40.48 GiB`) plus the pinned source closure. One raw BF16 feature is `2,400 x 4,096 x 2 = 19,660,800` bytes (`18.75 MiB`), about four times smaller than unpooled Cosmos context.

Using the existing stride-3 scale (`~1,560` anchors) as a capacity proxy:

- raw LTX features: about `28.56 GiB`;
- steady extraction: `1,560 x 1.233393 s = 32.05 min`;
- including one cold init/open allowance: about `33.9 min` compute-only, before serialization, hashing, and filesystem overhead.

At stride 1 (`~4,680` anchors), raw features are `~85.69 GiB`; steady compute is `~96.2 min`, or `~98.0 min` including cold init/open. These are projections, not a cache-build run.

## Tiny-overfit gate result

The real-prompt action gate passed on 2026-08-25. `precompute_ltx_prompt.py` ran
the official Gemma-4 12B encoder and LTX connector in a standalone process and
wrote a strict `[1,1024,4096]` BF16 artifact for `take cube out of box` (embedding
SHA-256 `5beb92e61ff925f2c0a20b6240bb0e490aaa8414d798e44d7a39470cd3c3e5b3`).
The extraction/training process only loaded this artifact; it never loaded the
text encoder. The official LTX source requires Transformers `>=5.8,<5.15`; this
run used 5.14.1.

```bash
source scripts/video_vam/cosmos_cuda_env.sh
.venv/bin/python scripts/video_vam/precompute_ltx_prompt.py --overwrite
.venv/bin/python scripts/video_vam/train_ltx_world2action_tiny.py \
  --windows 0:4,0:24,0:44,0:64 --max-steps 1000 --max-minutes 25 \
  --eval-every 10 --fit-ratio 0.1 --overwrite
```

Four online causal windows were extracted with persistent FP8-cast CPU streaming
and true block-34 early exit. The first window took 2.726 s; the next three had
1.250 s p50 (`0.800 Hz`). Extraction peaked at 5.15 GiB allocated. The process
then closed/deleted LTX and retained only eight spatially averaged 4096-d tokens
per window in memory. A non-affine LayerNorm and bias-free 4096-to-2048 linear
adapter fed the unchanged native World2Action decoder.

The real consumer smoke produced loss 1.866681 and non-zero finite gradients in
both adapter and decoder. The deterministic three-window fit stopped at step 230
in 34.625 s (`6.643` steps/s) when fixed flow loss reached 0.222507 from 2.517702
(ratio 0.0884). Train action RMSE improved from 1.0083 to 0.3820 normalized and
73.08 to 26.58 degrees. Final controls were materially worse: zero context 66.48
degrees and shuffled context 48.37 degrees. Parameter probes moved (L2 0.1213;
max absolute 0.00568). Training peak allocation was 8.62 GiB. Total process wall
time, including checkpoint hashing/open, extraction, training, and evaluation,
was 153.29 s.

The fourth fixed window was excluded from optimization and scored 27.88 degrees
/ 0.5015 normalized. It is useful only as a smoke: one nearby held-out window is
not evidence of generalization. No feature cache or decoder checkpoint was
written. The metrics are in `~/.cache/video-vam/runs/ltx25-tiny-overfit/result.json`; the tmux harness log, exit code, and exact run script are beside it.

## Limitations

- The overfit establishes trainability and strong context sensitivity, not data
  efficiency or held-out task performance; the decoder and adapter were jointly
  optimized on only three windows.
- The real-prompt run measured three steady samples after one warmup on one RTX 4090. The earlier ten-sample zero-prompt benchmark remains the better latency
  estimate, and both agree on approximately 0.8 Hz.
- Provider `get` time is not isolated PCIe transfer time.
- No rollout or full-dataset training was run, and no LTX feature cache exists.
- At 1.216 s for the fastest gated arm, LTX is still roughly 12.2x slower than the 100 ms
  budget and cannot meet a 10 Hz online control loop.
- Compiled-VAE correctness was measured on one fixed real window and prompt tensor. Its
  action-level effect has not been measured, so eager VAE remains the default.
