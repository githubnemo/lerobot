# Cube-Out-of-Box Leaderboard

**Dataset:** `hubnemo/cube_out_of_box_dataset` @ `243370c3c08bcbd860133c4a0d658ea7c1d2e77e` (6,536 frames, 40 episodes, 10 fps).
**Protocol Split:** Train episodes 0–31 (32 demos), Val episodes 32–39 (8 held-out demos, 88 fixed validation anchors).
**W&B Project:** [video-vam-world2action](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action).
**Policy Horizon & Units:** Full 30 steps (offsets $0 \dots 29$ at 10 Hz). Mixed-unit aggregate: 5 arm joints in degrees ($^\circ$) + gripper in range $[0, 100]$.

> **Append-Only Policy:** This leaderboard preserves all historical runs for scientific accountability. Historical entries with methodological flaws or data leakage are explicitly marked **[RETRACTED / INVALID]** with accompanying audit notes, never silently deleted.

---

## 1. Official Validated Protocol 1.0 Benchmark Table

| Date       | Model / Backbone                     | Extraction / Conditioning                     | Full-30 Mixed RMSE | Immediate H1 ($t$) | First-5 ($t..t+4$) | Status / Integrity                             |
| :--------- | :----------------------------------- | :-------------------------------------------- | :----------------- | :----------------- | :----------------- | :--------------------------------------------- |
| 2026-08-28 | Cosmos-Predict2-2B Video-LoRA        | $T=16$, Layer 20, pool2 (4,800 tokens)        | **13.06**          | 4.76               | 6.19               | **Valid Protocol 1.0** (Offline Gold Standard) |
| 2026-08-28 | Cosmos-Predict2-2B Teacher Ref       | $T=16$ `cond_frames`, Layer 20 (2,400 tokens) | **13.08**          | 4.97               | 6.40               | **Empirical Teacher Reference** (Not ceiling)  |
| 2026-08-29 | Cosmos-Predict2-2B Direct Distilled  | $T=2$ Student LoRA, Layer 20 (2,400 tokens)   | **13.15**          | 4.58               | 6.04               | **Valid Protocol 1.0** (89.4% gap closure)     |
| 2026-08-24 | Cosmos-Predict2-2B Frozen Pretrained | $T=16$, Layer 20, pool2 (4,800 tokens)        | **13.65**          | —                  | —                  | **Valid Protocol 1.0** (Converged Frozen)      |
| 2026-08-28 | Cosmos-Predict2-2B Undistilled Base  | $T=2$, Layer 20 unpooled (2,400 tokens)       | **13.74**          | 4.78               | 6.55               | **Valid Protocol 1.0** (Fast Baseline)         |
| 2026-08-25 | Cosmos-Predict2-2B Prefix VAE        | $T=16$, Prefix VAE, pool2 (4,800 tokens)      | **13.81**          | —                  | —                  | **Valid Protocol 1.0**                         |
| 2026-08-26 | LTX-Video-2.5 22B                    | Layer 34 unpooled (2,400 tokens)              | **13.84**          | 4.55               | 6.24               | **Valid Protocol 1.0**                         |
| 2026-08-26 | LTX-Video-2.5 22B                    | Layer 34 pool2 (640 tokens)                   | **14.03**          | 4.50               | 6.28               | **Valid Protocol 1.0**                         |
| 2026-09-04 | Cosmos 3 Edge Pure 600 Vision        | Layer 20 `und_seq` (600 tokens)               | **14.26**          | 4.67               | 6.34               | **Valid Protocol 1.0** (Base Zero-Shot)        |
| 2026-08-20 | Cosmos-Predict2-2B World2Action      | Layer 20, pool2 (4,800 tokens)                | **14.51**          | —                  | —                  | **Valid Protocol 1.0**                         |
| 2026-08-24 | SmolVLA 450M Reference               | Pretrained 2D SigLIP vision (64 tokens)       | **14.83**          | 4.85               | 6.64               | **Valid Reference Baseline** (Converged)       |
| 2026-08-19 | State Repeat Baseline                | Repeat proprioceptive state across 30 steps   | **18.86**          | 0.89               | 4.45               | **Zero-Motion Baseline**                       |
| 2026-08-19 | Mean Action Baseline                 | Repeat train-split mean action vector         | **30.15**          | 12.40              | 18.20              | **Mean Action Baseline**                       |

---

## 2. Historical Runs & Retraction Ledger (Methodological Audit)

The following entries are recorded for historical provenance, but have been audited and retracted due to severe methodological defects:

| Date       | Run Description                      | Reported RMSE | Metric Type | Audit Finding & Retraction Notice                                                                                                                                             |
| :--------- | :----------------------------------- | :------------ | :---------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-09-05 | Cosmos 7B Base (Dual Layers 14 & 20) | _9.46_        | Trainer Val | **[RETRACTED / INVALID]** Frame-level random split (85/15) across stride-3 clips leaked adjacent trajectory frames (300 ms apart). Bilinear pseudo-latents bypassed true VAE. |
| 2026-09-05 | Cosmos 7B Video-LoRA Adapted         | _9.75_        | Trainer Val | **[RETRACTED / INVALID]** Frame-level data leakage. Bilinear pseudo-latents.                                                                                                  |
| 2026-09-05 | Cosmos 14B Base (Layers 18 & 30)     | _9.97_        | Trainer Val | **[RETRACTED / INVALID]** Frame-level data leakage. Ignored `action_is_pad`.                                                                                                  |
| 2026-09-05 | FLUX.2 [klein] Video-LoRA Multi-Ref  | _10.33_       | Trainer Val | **[RETRACTED / INVALID]** Frame-level data leakage. Bilinear pseudo-latents.                                                                                                  |
| 2026-09-05 | FLUX.2 [klein] Base Junction Tap     | _11.45_       | Trainer Val | **[RETRACTED / INVALID]** Frame-level data leakage. Bilinear pseudo-latents.                                                                                                  |
| 2026-09-06 | Cosmos 14B Base Converged            | _10.99_       | Trainer Val | **[RETRACTED / INVALID]** Frame-level data leakage across continuous trajectories.                                                                                            |
| 2026-09-06 | Cosmos 14B Video-LoRA (FP8 QLoRA)    | _11.08_       | Trainer Val | **[RETRACTED / INVALID]** Frame-level data leakage across continuous trajectories.                                                                                            |
| 2026-08-20 | Unadapted Early Cosmos 2B Baseline   | _26.12_       | Trainer Val | **[NON-COMPARABLE]** Early uncalibrated configuration; not comparable to standard Protocol 1.0.                                                                               |

---

## 3. Retraction & Correctness Audit Summary

Per `docs/video_vam_correctness_audit.md`:

1. **Frame-Level Data Leakage:** Any run utilizing in-memory `torch.utils.data.random_split` on robotic trajectory datasets allows the policy to perform simple trajectory interpolation rather than out-of-distribution generalization.
2. **Pseudo-Latent Bypassing:** Several large-model feature extractors downsampled RGB frames with `F.interpolate` and padded with zeros rather than passing through genuine causal 3D VAEs.
3. **Current Action:** Agents are implementing correct, leakage-free extraction pipelines based on `BaseVAMExtractor` and genuine VAEs. Reruns across Cosmos 3, Cosmos 7B, Cosmos 14B, and FLUX.2 will be published once code review and test verification are complete. No jobs are currently running.

## Phase 5: Expanded 100-Episode Demonstration Scaling & Dual-Evaluation Benchmark (2026-09-07)

Evaluates the exact scaling impact of expanding demonstration data from 40 to 100 episodes (40 original + 60 new samples from `Orellius/cube_out_of_box_v2` / 12,163 frames).
Compares Base feature representations against domain-adapted Video-LoRA representations under strict Protocol 1.0.

**Dual-Evaluation Benchmark Protocol:**

- **Train Set:** 82 episodes (Original 0–31 + 50 new demonstration episodes 40–89; stride 3, 3,065 samples).
- **Eval-Set 1 (Historical Benchmark):** 8 original held-out episodes 32–39 (strict Protocol 1.0, stride 20, exactly 88 anchors). Directly isolates and measures the generalization benefit of 100 vs 40 demonstration trajectories against historical benchmarks.
- **Eval-Set 2 (New Benchmark):** 10 held-out episodes from the new demonstrations (episodes 90–99; stride 20, 51 anchors). Measures in-distribution generalization on newly collected demonstration dynamics.
- **Action Masking:** `action_is_pad` strictly respected across normalizer statistics, flow loss, and trajectory evaluation.
- **Queue:** Sequentially running in detached tmux session `scale100_videolora_queue`.

| date       | arm                                                                  | Eval-Set 1 RMSE (Hist) | Eval-Set 2 RMSE (New) | best step   | budget                          | status                          | notes                                                                                   |
| :--------- | :------------------------------------------------------------------- | :--------------------- | :-------------------- | :---------- | :------------------------------ | :------------------------------ | :-------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| 2026-09-11 | **Cosmos 3 Edge Video-LoRA (600 tokens, Dual-Pathway) + SmolExpert** | **13.49°** (h1 3.62°)  | **17.20°** (h1 4.92°) | step 34,500 | max 50k (patience 20)           | **COMPLETED (Leader)**          | **State-of-the-Art on Scale-100**: lowest RMSE on both Eval-1 & Eval-2; ~80 ms latency. |
| 2026-09-11 | Cosmos 14B Base (Layers 18 & 30, FP8 Block-Streaming) + SmolExpert   | **14.02°** (h1 4.57°)  | **19.41°** (h1 6.08°) | step 18,000 | min 20k / max 60k (patience 25) | **COMPLETED**                   | 100 eps (82 train, 8 historical val, 10 new val). True VAE causal latents.              |
| 2026-09-11 | FLUX.2 [klein] Base (Multi-Reference Junction Tap) + SmolExpert      | **15.75°** (h1 5.18°)  | **20.45°** (h1 6.56°) | step 17,500 | min 15k / max 50k (patience 25) | **COMPLETED**                   | 100 eps (82 train, 8 historical val, 10 new val). Junction tap 256 tokens.              |
| 2026-09-11 | FLUX.2 [klein] Video-LoRA (Multi-Ref 5k steps) + SmolExpert          | **15.89°** (h1 5.07°)  | **21.02°** (h1 6.77°) | step 19,500 | min 15k / max 50k (patience 25) | **COMPLETED**                   | 100 eps (82 train, 8 historical val, 10 new val). Multi-Ref LoRA adaptation.            |
| 2026-09-11 | Cosmos 14B Video-LoRA (FP8 QLoRA 4k steps) + SmolExpert              | —                      | —                     | —           | —                               | min 20k / max 60k (patience 25) | **QUEUED**                                                                              | 100 eps (82 train, 8 historical val, 10 new val). Quantized LoRA on all 36 blocks. |
