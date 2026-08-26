# Video-VAM temporal consistency and RTC report

## Protocol

All primary numbers below use held-out episode 32, exact consecutive anchors 4820 and 4821 (0.1 s apart), and eight action-flow seeds (0–7). Cosmos and LTX contexts were extracted exactly once per anchor in memory; the action-only draws then reused those frozen tensors. No feature cache or other large artifact was written. The same analysis was also run on four stride-3 cached training anchors as a replication; JSON results are under `/home/anton/.cache/video-vam/dry-runs/*-consistency-episode-{0,32}.json`.

For fixed conditioning `c` and state `s`, samples are `a^(n) = FlowDecode(c,s,epsilon_n)`. The metric reports population standard deviation over draws at every `[timestep,joint]`, its horizon and joint reductions, and mean pairwise chunk RMSE. Adjacent-conditioning tests hold flow seed fixed. Feature-only and state-only counterfactuals change one input at a time. All aggregates mix five degree joints and one `range_0_100` gripper only when explicitly labeled aggregate; per-joint order is shoulder pan, shoulder lift, elbow, wrist flex, wrist roll, gripper.

## Action-flow sampling uncertainty

SmolVLA:

- Per-anchor mean std / pairwise RMSE / std at h1 / std at h30: frame 4820 `0.919 / 1.586 / 0.846 / 0.558`; frame 4821 `0.833 / 1.409 / 0.841 / 0.594`.
- Aggregate mean std `0.876`; mean pairwise RMSE `1.498`.
- Pairwise RMSE per joint: `[1.222, 2.072, 1.604, 1.600, 1.321, 0.273]`.
- Mean std per joint: `[0.810, 1.382, 1.050, 1.005, 0.837, 0.175]`; h1/h30 `0.844/0.576`.

Cosmos + SmolExpert:

- Per-anchor mean std / pairwise RMSE / std at h1 / std at h30: frame 4820 `1.336 / 2.221 / 1.245 / 1.432`; frame 4821 `1.373 / 2.277 / 1.145 / 1.696`.
- Aggregate mean std `1.355`; mean pairwise RMSE `2.249`.
- Pairwise RMSE per joint: `[2.624, 2.811, 2.416, 2.433, 1.174, 0.877]`.
- Mean std per joint: `[1.746, 1.894, 1.583, 1.627, 0.749, 0.529]`; h1/h30 `1.195/1.564`.

LTX-2.5 + SmolExpert:

- Per-anchor mean std / pairwise RMSE / std at h1 / std at h30: frame 4820 `1.233 / 1.975 / 1.089 / 1.394`; frame 4821 `1.187 / 1.941 / 1.003 / 1.417`.
- Aggregate mean std `1.210`; mean pairwise RMSE `1.958`.
- Pairwise RMSE per joint: `[2.118, 2.520, 1.935, 2.005, 1.255, 1.312]`.
- Mean std per joint: `[1.430, 1.690, 1.265, 1.320, 0.813, 0.740]`; h1/h30 `1.046/1.405`.

The full per-anchor `[30,6]` standard-deviation matrices and 30-step horizon curves are in the JSON artifacts.

## Adjacent conditioning and seam verdict

With fixed flow noise and time alignment (`old[1:]` versus `new[:-1]`), chunk RMSE is `3.258` for SmolVLA, `3.900` for Cosmos, and `4.612` for LTX. Averaging all eight draws barely changes the video-feature values (`3.791` Cosmos, `4.537` LTX), so draw averaging does not remove the adjacent-conditioning change.

For Cosmos, adjacent context relative RMS delta is `0.665` and cosine similarity `0.780`. Changing only context changes same-index actions by `3.355` aggregate with per-joint `[4.833, 3.816, 2.680, 3.857, 2.599, 0.889]`; changing only state costs `0.328` aggregate.

For LTX, context relative RMS delta is `0.664` and cosine similarity `0.780`. Changing only context changes same-index actions by `3.854` aggregate with per-joint `[1.853, 4.554, 3.944, 5.346, 3.137, 3.314]`; changing only state costs `0.413` aggregate.

The original dry-run seam compared new action 0 to the action executing when inference completed. That is not LeRobot RTC queue semantics: `ActionQueue` drops the new actions whose timestamps elapsed during inference. Re-indexing the existing episode-32 artifacts by `installed_frame - anchor_frame` changes the mean seams from:

- SmolVLA: `3.257` naive to `2.806` delay-aligned (six seams).
- Cosmos: `18.170` naive to `2.986` delay-aligned (one seam).
- LTX: `23.488` naive to `4.325` delay-aligned (one seam).

Verdict: the reported 18–23 unit video-policy seams are **not mostly flow sampling noise**, but they are also **not evidence of a 6× frozen-feature instability**. Most of that contrast is a timestamp/indexing error in the naive replacement scheduler. After correct delay alignment, residual seams are 3–4 units, comparable to fixed-noise adjacent-conditioning changes and only modestly above SmolVLA. Sampling spread (pairwise 2.25 Cosmos, 1.96 LTX) contributes but is smaller; feature-conditioned action change, not proprioceptive state change, explains most of the residual. This is a real sensitivity result, but much weaker than the original seam numbers implied.

## RTC

The decoder now uses LeRobot's existing `RTCProcessor` and shared Euler flow integrator. Previous physical actions are normalized with each checkpoint's train-only action statistics, padded from six to the expert's 32 latent action dimensions, guided during flow denoising, then denormalized exactly once. Guided calls bypass the unguided CUDA graph. RTC tests use independent old/new flow seeds, delays of 3/13/24 steps from the supplied 0.265/1.288/2.384 s measurements, and exact episode-32 adjacent contexts.

With the existing `s_min=10` convention (`execution_horizon=max(delay,10)`):

- SmolVLA: seam `3.714 -> 1.280` (65.5% reduction); first-five executed-prefix RMSE `8.015 -> 8.631` (cost `+0.616`). Guided seam per joint: `[0.843, 1.358, 0.141, 2.628, 0.581, 0.077]`.
- Cosmos: seam `3.013 -> 2.786` (7.5% reduction); prefix RMSE `16.429 -> 18.717` (cost `+2.288`). Guided seam per joint: `[2.522, 0.738, 0.157, 4.940, 3.592, 1.530]`.
- LTX: seam `4.325 -> 2.213` (48.8% reduction); prefix RMSE `13.182 -> 13.634` (cost `+0.452`). Guided seam per joint: `[0.419, 3.077, 1.567, 0.845, 4.010, 0.702]`.

Because Cosmos and LTX delays exceed ten steps, the default minimum leaves little or no soft transition after the guaranteed-to-execute prefix. A full-horizon diagnostic (`s_min=30`) gives:

- SmolVLA: `3.714 -> 1.423` (61.7%); accuracy cost `+0.742`.
- Cosmos: `3.013 -> 0.983` (67.4%); prefix RMSE `16.429 -> 17.794`, cost `+1.365`.
- LTX: `4.325 -> 1.038` (76.0%); prefix RMSE `13.182 -> 13.113`, cost `-0.068` on this one pair.

The `s_min=30` result is promising for slow video policies but is one held-out transition, not a tuned recommendation. Cosmos shows a material tracking tradeoff in both profiles.

## Production status and unverified items

A hardware-loadable `PreTrainedPolicy` adapter was intentionally not shipped because it cannot yet be faithful: `RTCInferenceEngine` retains only the latest observation snapshot, while both video experts require every consecutive 10 Hz frame in an exact five-frame causal history. A policy-local buffer would collect inference-spaced frames (roughly 1.3 s Cosmos / 2.4 s LTX) and silently diverge from training. The remaining production change is an observation-history boundary populated on every control tick, followed by a typed expert policy/processor wrapper and safety processor.

No robot was connected. Motor smoothness, task success, physical limits, camera cadence under extraction, behavior during queue starvation, and the best `s_min`/guidance weight remain unverified. Latencies were not remeasured because the GPU was contended; supplied pessimistic values were used. Exact-context LTX measurement took longer under contention but performed only two backbone forwards and wrote no feature artifacts.

## Follow-up: validation distributions and hardware history

### Distributional RTC measurement is blocked by cache cadence

The requested all-transition validation distribution cannot be computed honestly from the caches currently on `abakus`. Episodes 32-39 contain 1,688 valid five-frame anchors and 1,680 within-episode adjacent transitions. The available Cosmos and LTX validation caches contain only 88 anchors each, sampled every 20 frames (80 within-episode adjacent pairs).

That stride is not merely a lower-resolution substitute for RTC. With the fixed measured delays used by the analysis (3 steps SmolVLA, 13 Cosmos, 24 LTX), the previous-chunk comparison index is `stride + delay - 1`. At stride 20 this is 32 for Cosmos and 43 for LTX, both beyond the 30-action horizon, so there are **zero valid delay-aligned Cosmos/LTX RTC transitions** in the existing validation caches. Treating those anchors as consecutive would fabricate seams and repeat the scheduler-indexing mistake this analysis corrected.

A dense validation cache would occupy approximately 30.91 GiB for Cosmos pool2 plus 8.24 GiB for LTX pool2 at the observed per-anchor sizes. Streaming extraction could avoid that disk cost but would still run both video backbones over every validation frame. No extraction, policy benchmark, or competing GPU job was started while the queued training process was active. Consequently there are no defensible `n`, mean, median, p90, max, ordering-stability, `s_min=30`, LTX-prefix-cost, or worst-degradation results yet; the single-transition conclusions remain directional only.

The analysis output schema is ready for the eventual dense run: seam and per-joint RMSE summaries now include `n`, mean, median, p90, and max; RTC summaries include per-transition seam change, executed-prefix cost distributions, and the worst transition with sample IDs. The latency constants remain the previously measured contention-affected 0.265 s / 1.288 s / 2.384 s; they were not remeasured during this follow-up.

The episode-24 regime change does not invalidate policy-self-consistency seams, but it matters for absolute executed-prefix RMSE against recorded actions. All requested validation transitions are late-regime while most training data is early-regime. RTC baseline and guided costs remain paired against the same late-regime target, so their difference is interpretable even if both absolute errors are elevated. No distribution is available yet to quantify that effect.

### Faithful five-frame RTC history is implemented

`RTCInferenceEngine` now accepts the opt-in `--inference.observation_history_size=5`. At the ordinary 10 Hz policy cadence it snapshots each processed observation before the asynchronous inference thread can skip or overwrite it, waits for all five frames at episode startup, and hands the policy `observation.images.<camera>.history` with shape `[B,C,5,H,W]`. Reset clears the deque, preventing cross-episode leakage. The default is one and follows the old latest-observation path without copying or adding batch keys, so existing policies do not change.

The integration test labels five successive control-rate frames 0-4 and verifies that the policy receives those exact values in order, with no inference before the fifth frame. The complete rollout test modules pass (100 tests), and targeted pre-commit checks pass including ruff, mypy, bandit, and markdown formatting.

For one 480x640 RGB camera, the history costs about 4.39 MiB as raw uint8 snapshots and another 17.58 MiB while represented as float32 policy input, plus the normal current frame. This is negligible relative to the video backbones and scales linearly with camera count.

The remaining collection-day blocker is not frame cadence: it is the absent hardware-loadable `PreTrainedPolicy`/processor wrapper for Cosmos/LTX. That wrapper must consume the new history key, restore the extractor's uint8 `[B,3,5,480,640]` input exactly, load the selected backbone and SmolExpert normalizer/checkpoint, and return physical 30x6 actions through the existing RTC call. It also needs robot-side joint limits and a dry hardware safety check.

Recommended collection-day options, in fidelity order:

1. Build and rehearse that thin wrapper with history size 5. This preserves training fidelity and is the recommended path.
2. Run SmolVLA only if the wrapper is not ready; it already has the supported one-frame `PreTrainedPolicy` path.
3. Retrain or at least validate a one-frame/short-context expert before deploying it. This changes the model contract but can be made scientifically honest.
4. Emergency-only: repeat the latest image five times. This is shape-compatible but removes the trained 0.4 s temporal signal and is out of distribution; its accuracy cost is unmeasured.
5. Do not use an inference-spaced five-frame policy-local buffer. At the measured latencies it spans roughly 5.2 s for Cosmos or 9.6 s for LTX between oldest and newest frames rather than 0.4 s, so it is less faithful than the explicit repeated-frame fallback.
