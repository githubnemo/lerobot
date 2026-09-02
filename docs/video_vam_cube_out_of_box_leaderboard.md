# Cube-out-of-box leaderboard

Dataset: `hubnemo/cube_out_of_box_dataset` @ `243370c3c08bcbd860133c4a0d658ea7c1d2e77e`.
Split: train episodes 0–31, val episodes 32–39.
W&B project: [video-vam-world2action](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action).

This file is append-only. Do not delete or rewrite old rows. Add a new dated line when a run finishes.

Two RMSE columns exist and are **not interchangeable**:

- **Trainer val RMSE**: fixed-probe validation inside `train_cosmos_world2action.py` / LoRA trainer.
- **Frozen-protocol RMSE**: `evaluate_action_rmse.py` protocol 1.0 (88 val anchors, masked global RMSE, degrees).

SmolVLA was **not trained to convergence**. The 10-minute run (5,000 steps) and the ~1-hour run (29,200 steps) are both early-stopped / budget-capped. Do not read a VAM number beating the 10-minute SmolVLA as “VAM beats SmolVLA.”

---

## Headline rows (add below, never replace)

| date          | arm                                                           | metric                    | RMSE °        | steps / budget               | W&B                                                                                    | notes                                                                                                                                                                                                                                                                                      |
| ------------- | ------------------------------------------------------------- | ------------------------- | ------------- | ---------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2026-08-18    | SmolVLA retry4 (pretrained vision)                            | frozen-protocol           | 15.00         | 5,000 steps / ~10 min        | [c597bb29](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/c597bb29) | Held-out 0–31. The short SmolVLA reference.                                                                                                                                                                                                                                                |
| 2026-08-19    | SmolVLA 1h (pretrained vision)                                | reported val / comparison | 14.83         | 29,200 steps / ~1 h          | [424c1831](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/424c1831) | Longer SmolVLA. Still not trained to plateau. Frozen-protocol re-score of this exact ckpt was not written as a separate JSON.                                                                                                                                                              |
| 2026-08-19    | state_repeat baseline                                         | frozen-protocol           | 18.86         | n/a                          | —                                                                                      | Repeat last observed joint state.                                                                                                                                                                                                                                                          |
| 2026-08-19    | mean_action baseline                                          | frozen-protocol           | 30.15         | n/a                          | —                                                                                      | Train-split mean action.                                                                                                                                                                                                                                                                   |
| 2026-08-20    | Frozen Cosmos 2B, full grid (runD)                            | trainer val               | 16.59         | 4,500 / 14 h                 | [v50tr0dk](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/v50tr0dk) | Cached, transform `none`, 19,200 tokens.                                                                                                                                                                                                                                                   |
| 2026-08-20    | Frozen Cosmos 2B, full grid (runD)                            | frozen-protocol           | 16.92         | same ckpt                    | [v50tr0dk](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/v50tr0dk) | Standalone evaluator.                                                                                                                                                                                                                                                                      |
| 2026-08-20    | Frozen Cosmos 2B, pool2 (30 min ablation)                     | trainer val               | 23.32         | 808 / 30 min                 | [8ncbyble](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/8ncbyble) | Best connector arm at equal wall-clock.                                                                                                                                                                                                                                                    |
| 2026-08-20–21 | Frozen Cosmos 2B, pool2 batched (to plateau)                  | trainer val               | **14.51**     | 19,500 / overnight           | [kn787l8b](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/kn787l8b) | Best VAM so far. Beats 10-min SmolVLA (15.00). Does **not** license a claim against the 1 h SmolVLA (14.83) — different metric, and SmolVLA was not converged. Last val 14.82 at step 24,600. Frozen-protocol eval of this ckpt not run (evaluator lacked context transforms at the time). |
| 2026-08-21    | Random-init Cosmos pool4                                      | trainer val               | 23.94         | 887 / ~30 min                | [n6uwog60](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/n6uwog60) | Pretrained pool4 30 min was 24.04 — no pretrained benefit at that budget.                                                                                                                                                                                                                  |
| 2026-08-21    | SmolVLA vision-randomized (invalid first attempt)             | frozen-protocol           | 14.83         | same as pretrained twin      | [de1efc75](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/de1efc75) | Hook never ran (stale site-packages). Byte-identical to pretrained. Do not use as a control.                                                                                                                                                                                               |
| 2026-08-21    | SmolVLA vision-randomized (verified rerun)                    | frozen-protocol           | 17.43         | ~30 min queue                | [4c6b8c2e](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/4c6b8c2e) | 197/197 vision tensors actually reinitialized. Worse than pretrained 15.00, so vision prior helps SmolVLA.                                                                                                                                                                                 |
| 2026-08-21    | VLA-JEPA fused AdamW                                          | train loss only           | loss 0.244    | 2,401 / ~29 min              | [W&B](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/a85055ba)      | Ran. Action horizon 7, so **no** frozen-protocol 30-step RMSE. Not comparable on this board yet.                                                                                                                                                                                           |
| 2026-08-21    | FastWAM                                                       | no score                  | —             | construction OK; train OOM   | —                                                                                      | Loads on 24 GB in bf16 (6.02B params). Backward OOMs at batch 1 (~23.5 GiB). No real run, no RMSE.                                                                                                                                                                                         |
| 2026-08-22    | LoRA co-train Arm A (action grads → LoRA, λ_video=0.5, pool2) | trainer val               | 24.37         | 2,983 / 8 h                  | [er7bi2bl](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/er7bi2bl) | Joint video+action. Worse than frozen pool2 14.51. Best at step 2,700; last val 27.54 at 2,900.                                                                                                                                                                                            |
| 2026-08-22    | LoRA co-train Arm B (video-only LoRA, λ_video=0.5, pool2)     | trainer val               | 56.42 (early) | 200 / 3 h cap, still running | [s59gpitx](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/s59gpitx) | First val only. Update this row when the run finishes.                                                                                                                                                                                                                                     |

---

## Connector ablation (30 min each, 2026-08-20)

All cached, sigma 80, trainer val RMSE. W&B project same as above.

| arm                   | tokens | best val RMSE | W&B                                                                                    |
| --------------------- | ------ | ------------- | -------------------------------------------------------------------------------------- |
| runD @30 min (`none`) | 19,200 | 45.81         | [v50tr0dk](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/v50tr0dk) |
| pool2                 | 4,800  | 23.32         | [8ncbyble](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/8ncbyble) |
| cond_frames           | 2,400  | 24.28         | [rmcpaxtc](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/rmcpaxtc) |
| pool4                 | 1,280  | 24.04         | [8eflb9j3](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/8eflb9j3) |
| gen_frames_pool2      | 4,200  | 26.24         | [nn8sicpy](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/nn8sicpy) |
| global_mean           | 1      | 25.66         | [x3g4ptaw](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/x3g4ptaw) |
| frame_mean            | 16     | 26.94         | [gyiep8ro](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/gyiep8ro) |

---

## Changelog

- 2026-08-22: file created. Seeded with the cube-out-of-box runs through LoRA Arm A and in-progress Arm B.
  | 2026-08-24 | SmolVLA converged (29.2k steps, batch 8) | frozen protocol | 14.83 | curve 15.38/15.05/14.88/14.87/14.83 at 7k/14k/21k/28k/29.2k | see 20260819_1hr run | Plateaued; retires the "not converged" caveat on the 5k-step entry. |
  | 2026-08-24 | SmolVLA batch 64 (8k steps) | frozen protocol | 14.82 | 8,000 / ~65 min | smolvla_batch64_cube_out_of_box | Batch size was not the limiter; matches batch-8 plateau. |
  | 2026-08-24 | VLA-JEPA chunk-30 retrain | frozen protocol | 19.19 | 30-step horizon | vla-jepa-chunk30 | Worse than state_repeat 18.86. |
  | 2026-08-24 | FastWAM LoRA co-train (4.2k steps) | frozen protocol | 20.36 | 4,200 / 1 h | fastwam-lora | Worse than state_repeat 18.86. |
  | 2026-08-24 | Cosmos pool2 + pretrained SmolVLA expert (~100M + 2M adapter) | trainer fixed-probe val | **14.14** | best at 6k / ~50 min | [wvltqpfl](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/wvltqpfl) | New best. Same features as pool2+World2Action (14.51), same expert as SmolVLA (14.83): pretrained small head wins. Standalone frozen-protocol eval pending (needs smolexpert backend in evaluator). |
  | 2026-08-24 | Cosmos pool2 + SmolVLA expert, CONVERGED (~26k steps, ~3.5 h) | trainer fixed-probe val | **13.65** (best, step 22k) | settled 13.8-13.9 | run smolexpert-on-cosmos-pool2-converge (oy0an3ag) | New overall best. Improved past the 1-h cap result (14.14); best ckpt train-converge/best.safetensors. Frozen-protocol standalone eval still pending (needs smolexpert eval backend). |
  | 2026-08-25 | Cosmos pool2 + pretrained SmolVLA expert, prefix VAE, 32 episodes | trainer fixed-probe val | **13.81** | converged; best step 36k | smolexpert-prefix-retrain | Prefix VAE is 2.1x faster than legacy extraction; legacy-feature best was 13.65. |
  | 2026-08-25 | Cosmos pool2 + expert, prefix VAE, 16 episodes | trainer fixed-probe val | **20.13** | 1 h cap; best step 20k | smolexpert-prefix-ep16 | Better than SmolVLA 25.85; not formally converged. |
  | 2026-08-25 | Cosmos pool2 + expert, prefix VAE, 8 episodes | trainer fixed-probe val | **20.58** | 1 h cap; best step 21k | smolexpert-prefix-ep8 | Better than SmolVLA 25.24; not formally converged. |
  | 2026-08-25 | Cosmos pool2 + expert, prefix VAE, 4 episodes | trainer fixed-probe val | 32.81 | converged; best step 1k | smolexpert-prefix-ep4 | Both methods fail at 4 episodes; SmolVLA was 28.66. |

## Reading the RMSE scale (2026-08-25)

Action space on this dataset: per-joint ranges 146-197 deg for the four main joints, global action
std 45.26 deg, mean motion 0.89 deg/frame (so the arm travels ~27 deg across a 30-step / 3 s chunk).

Reference points on the same masked-chunk metric: mean_action 30.15, state_repeat 18.86,
1-NN state retrieval 23.61, best learned policy 13.65. The mean L2 distance from a validation
state to its nearest training state is 17.51 deg, so retrieval is weak and the learned policies are
genuinely generalizing. 13.65 is 0.30 action std and 28 percent better than holding still - real
signal, not near-optimal.

Main caveat: the score averages all 30 steps equally, but under chunked replanning only the first
few are executed, and those late steps are the least predictable. Differences of ~1 deg should not
be treated as a ranking without rollouts.

**Metric we are adding:** executed-prefix RMSE (h=1 and first-5) reported next to the full-30
number. It is free to compute from the same predictions and measures what the robot actually
performs.
| 2026-08-26 | LTX-2.5 pool2 + pretrained SmolVLA expert | trainer fixed-probe val | **14.03** (h1 4.50; first-5 mean 6.28) | best step 61,000; plateau step 71,000 / 1.68 h | [z1jp21cs](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/z1jp21cs) | tokens=640, train stride=3; confound: none; same stride-3/stride-20 anchors as the LTX World2Action arm. |

## Dry-run findings that qualify every row above (2026-08-26)

Source: `scripts/video_vam/dry_run_rollout.py`, open-loop replay of held-out episode 32, no robot. Contract doc: `docs/video_vam_rollout_path.md`. JSON under `/home/anton/.cache/video-vam/dry-runs/`.

1. **SmolVLA rows have normalization leakage.** The checkpoint's serialized normalizer is dataset-global (`action.count` 6,536, stats through episode 39) while training used episodes 0-31. SmolExpert normalizers are train-only (1,572 anchors, episodes 0-31). The leak favors SmolVLA, so VAM-vs-SmolVLA gaps above are if anything conservative. A train-only-stats SmolVLA retrain is needed for a clean baseline.
2. **The RMSE unit is mixed, not pure degrees.** Five arm joints are degrees; the gripper is `range_0_100`. The aggregate therefore averages degrees with a percentage. Per-joint numbers remain interpretable; the scalar aggregate is a convention, not a physical angle.
3. **SmolVLA does not execute a 30-step chunk.** It predicts `chunk_size=50` and queues `n_action_steps=10` (1.0 s at 10 Hz). Our 30-step protocol scores a horizon it never executes, which is another reason to prefer executed-prefix error.
4. **Chunk seams are large for the video-feature policies.** Replacement-chunk seam RMSE: SmolVLA 3.26 deg (max joint jump 5.83), Cosmos 18.17 (25.86), LTX 23.49 (36.87). Unsafe on hardware without seam blending or shorter replan intervals, and a signal that frozen-feature policies are much less temporally consistent than the VLA at equal aggregate RMSE.
5. **Measured steady inference (under GPU contention, so pessimistic):** SmolVLA 0.265 s, Cosmos 1.288 s, LTX 2.384 s per chunk against a 3.0 s chunk budget. Chunked execution closes the real-time gap for all three; LTX has the least margin.
6. All three checkpoints load outside their trainers and are exactly deterministic under fixed noise. Physical joint limits are still unverified: no calibration or limit file exists on abakus.

## The split is regime-shifted: every number above is partly measuring session drift (2026-08-26)

Found while verifying the SmolVLA normalization leak. Per-episode mean actions show a **changepoint at episode 24**. Joints 2 and 3 flip sign across it:

| regime         | frames | mean action (6 joints)                    |
| -------------- | ------ | ----------------------------------------- |
| episodes 0-23  | 3,202  | `[12.5, -27.2, 23.5, 55.5, -53.0, 26.2]`  |
| episodes 24-39 | 3,334  | `[12.2, +5.2, -20.8, 67.5, -48.1, 18.1]`  |
| difference     |        | `[-0.4, +32.5, -44.3, +11.9, +4.9, -8.1]` |

Episode length also jumps (~100 frames before ep 20, 165-320 after), consistent with a re-mount, re-calibration, or moved box/cube partway through collection.

Consequence for our train 0-31 / val 32-39 split: training is 3,202 early-regime frames + 1,614 late-regime frames, while **validation is 100% late-regime**. So held-out RMSE measures generalization across a session shift, not in-distribution held-out accuracy. Train-vs-val mean action differs by +19.96 deg on joint 2 and -29.47 deg on joint 3.

This plausibly explains why every method above plateaus in a narrow 13.6-15.1 band regardless of backbone, connector, or head, and why `state_repeat` at 18.86 is not far behind: a shared irreducible floor set by the shift rather than by model capacity. It also means the SmolVLA normalization leak was more consequential than it would be in an i.i.d. split, since dataset-global statistics leaked late-regime centering.

NOT yet established: how much of the ~14 deg floor is the shift. Testing that requires a within-regime split (e.g. train 24-35 / val 36-39) and a cache rebuild.

Action for the next dataset: keep recording conditions fixed, and use an interleaved/random episode split rather than a trailing contiguous block. Verify train/val action statistics match before trusting any number.

## Correction to finding 4 above: the large seams were our scheduler, not the policies (2026-08-26)

Finding 4 in the dry-run section reported seam RMSE of 18.17 (Cosmos) and 23.49 (LTX) versus 3.26 (SmolVLA), and interpreted it as a temporal-consistency failure of frozen-feature policies. **That interpretation was wrong.** The harness installed each replacement chunk at action index 0, instead of skipping the actions whose timestamps elapsed during inference. Because Cosmos and LTX are slower, they were penalized in proportion to their latency. Re-indexed correctly on episode 32:

| policy              | reported seam | delay-aligned seam |
| ------------------- | ------------- | ------------------ |
| SmolVLA             | 3.257         | 2.806              |
| Cosmos + SmolExpert | 18.170        | 2.986              |
| LTX + SmolExpert    | 23.488        | 4.325              |

So there is no 6x temporal-consistency gap. The video-feature policies are modestly less consistent than the VLA, not catastrophically so, and the hardware-safety concern is much smaller than stated.

Fixed-context flow uncertainty (8 draws, same context and state, mean pairwise RMSE): SmolVLA 1.498, Cosmos 2.249, LTX 1.958. Sampling noise therefore contributes to but does not explain seams. With noise held fixed, adjacent-frame action change is SmolVLA 3.258, Cosmos 3.900, LTX 4.612; for the video policies, feature-driven change dominates state-driven change (Cosmos 3.355 vs 0.328; LTX 3.854 vs 0.413).

RTC (LeRobot's existing guided implementation, wired to `SmolExpertActionDecoder`) at the delays we measured. Seam reduction, and cost in executed-prefix RMSE:

| policy  | s_min=10                            | s_min=30                            |
| ------- | ----------------------------------- | ----------------------------------- |
| SmolVLA | 3.714 → 1.280 (-65.5%), cost +0.616 | 3.714 → 1.423, cost +0.742          |
| Cosmos  | 3.013 → 2.786 (-7.5%), cost +2.288  | 3.013 → 0.983 (-67.4%), cost +1.365 |
| LTX     | 4.325 → 2.213 (-48.8%), cost +0.452 | 4.325 → 1.038 (-76.0%), cost -0.068 |

At `s_min=10` the slow video policies have too little soft overlap left after inference delay for guidance to act on. `s_min=30` works much better for them.

**Caveat: all seam and RTC numbers here come from a single held-out transition on one episode (anchors 4820/4821).** They are directional, not estimates. Do not quote them as measured effect sizes until repeated over many transitions and episodes.

Full report: `docs/video_vam_temporal_consistency_report.md`.
| 2026-08-26 | LTX-2.5 unpooled + pretrained SmolVLA expert | trainer fixed-probe val | **13.84** (h1 4.55; first-5 mean 6.24) | best step 45,000; plateau step 55,000 / 3.68 h | [3rypmjug](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/3rypmjug) | tokens=2,400, train stride=3; confound: none; disk guard retained the full stride-3 train cache. |

<!-- ltx-layer-probe-20260826 -->

| 2026-08-26 | LTX multi-depth learned scalar mix, blocks 8/14/20/26/34/40 | trainer fixed-probe val | **15.09** | best 19,200 / stopped 25,200 | [run](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/lwvtjptm) | Full-30 global masked mixed-unit RMSE; h=1 6.79, first-5 8.77; per-joint: shoulder_pan=20.493, shoulder_lift=14.006, elbow_flex=14.221, wrist_flex=17.948, wrist_roll=8.160, gripper=12.599. pool2 (640 tokens) + World2Action; weights are a hypothesis. |
| 2026-08-26 | LTX top-1 confirmation, block 40 only | trainer fixed-probe val | **16.03** | best 5,100 / stopped 8,100 | [run](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/tz8nhgdg) | Same pool2 + World2Action; h=1 8.36, first-5 9.87; per-joint: shoulder_pan=22.341, shoulder_lift=15.020, elbow_flex=14.859, wrist_flex=19.694, wrist_roll=8.072, gripper=11.960. |
| 2026-08-26 | LTX top-2 confirmation, blocks 34/40 | trainer fixed-probe val | **15.29** | best 11,100 / stopped 14,100 | [run](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/jdod8g03) | Same pool2 + World2Action; h=1 8.36, first-5 9.99; per-joint: shoulder_pan=21.252, shoulder_lift=13.844, elbow_flex=15.495, wrist_flex=17.473, wrist_roll=8.228, gripper=12.081. |
| 2026-08-28 | Cosmos video-LoRA (step 6k) T=16 pool2 + SmolExpert | trainer fixed-probe val | **13.06** (h1 4.76; first-5 mean 6.19) | best step 38,000; plateau 48,000 / 2.42 h | [ixzworl4](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/ixzworl4) | Generic Cosmos + video-only LoRA rank/alpha 16. tokens=4,800. Offline best on this split. |
| 2026-08-28 | Cosmos video-LoRA (step 6k) `state_t=2` unpooled + SmolExpert | trainer fixed-probe val | **13.74** (h1 4.78; first-5 mean 6.55) | best step 27,000 / 46 min; plateau 37,000 / 1.06 h | [jri7vehq](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action/runs/jri7vehq) | Same LoRA as 13.06. DiT on 2 observed VAE latents only (2,400 tokens, transform none). Cache extract 5.7x vs T=16 LoRA pool2 (204 vs 1163 ms/entry). +0.68 vs 13.06; beats prefix-pool2 13.81. Peaked 11k steps earlier. |
