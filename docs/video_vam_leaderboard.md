# Video-VAM leaderboard (single source of numbers)

**Append-only.** Never delete or edit a historical number; add a validity tag and a note. Every new row needs: date, dataset@revision, split, seed(s), run dir, validity.
Metric: mixed-unit aggregate RMSE over a 30-step chunk (5 arm joints in degrees + gripper in [0, 100]); **H1** = offset 0, **F5** = mean of offsets 0–4. Contract: [`video_vam_action_rmse_protocol.md`](./video_vam_action_rmse_protocol.md). W&B: [video-vam-world2action](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action).

**Validity tags**

- `VALID` — Protocol-conformant, leakage-free, features match training.
- `VALID*` — conformant but weights lost / not re-scored with sampling rev 1.1, or pre-dates a later extractor fix; keep as reference only.
- `PROVISIONAL` — conformant code, but dataset integrity unresolved (v2 snapshot, see [status §2](./video_vam_status.md#2-datasets)).
- `UNVERIFIED` — reported in the diary without a local artifact to check.
- `INVALID` — bug affects the number (reason given). `NON-COMPARABLE` — different setup.

**All rows are single-seed** unless stated. Differences below ~0.5 should be treated as noise until seed variance is measured.

---

## 1. Cube-out-of-box v1, Protocol 1.0

`hubnemo/cube_out_of_box_dataset@243370c3`, train eps 0–31, val eps 32–39 (88 anchors, stride 20).

| Date  | Model                                             | Features / tokens                |   Full-30 |    H1 |    F5 | Latency | Validity / note                                                                                                                                         |
| :---- | :------------------------------------------------ | :------------------------------- | --------: | ----: | ----: | :------ | :------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 08-28 | Cosmos-2B video-LoRA + SmolExpert                 | T=16, L20, pool2, 4,800          | **13.06** |  4.76 |  6.19 | ~1.2 s  | `VALID*` weights on HF (`cube-out-of-box-cosmos-pool2-smolexpert`); not re-scored at rev 1.1                                                            |
| 08-28 | Cosmos-2B teacher `cond_frames` reference         | T=16 first 2 latent slots, 2,400 |     13.08 |  4.97 |  6.40 | —       | `VALID*` empirical reference, **not a ceiling**; weights lost                                                                                           |
| 08-29 | Cosmos-2B T=2 direct-distilled student            | T=2, L20, 2,400                  |     13.15 |  4.58 |  6.04 | ~204 ms | `VALID*` weights lost; "89.4 % gap closure" is within seed noise until re-run                                                                           |
| 08-24 | Cosmos-2B frozen pretrained                       | T=16, L20, pool2                 |     13.65 |     — |     — | ~1.2 s  | `VALID*`                                                                                                                                                |
| 08-28 | Cosmos-2B T=2 undistilled                         | T=2, L20, 2,400                  |     13.74 |  4.78 |  6.55 | ~204 ms | `VALID*` weights lost                                                                                                                                   |
| 09-09 | Cosmos-2B T=2 direct-distilled student (re-train) | T=2, L20, 2,400                  |     13.48 |  4.07 |  5.75 | ~204 ms | `VALID` weights on abakus `cosmos2b-t2-distilled-smolexpert` (+ LoRA `cosmos2b-t2-direct-distilled-lora`)                                               |
| 09-09 | Cosmos-2B T=2 undistilled (re-train)              | T=2, L20, 2,400                  |     13.65 |  4.23 |  6.20 | ~204 ms | `VALID` weights on abakus `cosmos2b-t2-undistilled-smolexpert`. Re-train gap to distilled: 0.16 (orig. 0.59)                                            |
| 09-28 | Cosmos 3 Edge **PEFT video-LoRA** + SmolExpert, online, no aug | 600, L20 | 13.63 | 4.26 | 6.02 | ~80 ms | `VALID` best @ 18k/25k (`outputs/train/c3peft_v1_unaug` on abakus; PEFT adapter `cosmos3-edge-v1-peft-lora`, val loss 0.3771) |
| 09-2x | Cosmos 3 Edge legacy video-LoRA + SmolExpert, online, no aug | 600, L20 | 13.73 | 4.30 | 6.05 | ~80 ms | `VALID` best @ 16.5k/25k (`outputs/train/v1-cosmos3-edge-lora-online-unaugmented-smolexpert` on abakus). Legacy vs PEFT diff 0.10 = within noise |
| 08-25 | Cosmos-2B prefix-VAE                              | T=16 prefix VAE, pool2           |     13.81 |     — |     — | —       | `VALID*`                                                                                                                                                |
| 09-12 | Cosmos 3 Edge video-LoRA, online + legacy aug     | 600, L20                         |     13.82 |  4.64 |  6.26 | ~80 ms  | `VALID` (Eval-2 on v2: 24.66, see §2)                                                                                                                   |
| 08-26 | LTX-2.5 22B                                       | block 34 unpooled, 2,400         |     13.84 |  4.55 |  6.24 | ~1.2 s  | `VALID*` weights on HF                                                                                                                                  |
| 08-26 | LTX-2.5 22B                                       | block 34 pool2, 640              |     14.03 |  4.50 |  6.28 | ~1.2 s  | `VALID*` weights on HF                                                                                                                                  |
| 09-04 | Cosmos 3 Edge base (zero-shot)                    | 600, L20                         |     14.26 |  4.67 |  6.34 | ~92 ms  | `VALID*` pre-dates the und/gen routing fix (audit §2.1)                                                                                                 |
| 08-20 | Cosmos-2B World2Action decoder                    | L20, pool2                       |     14.51 |     — |     — | —       | `VALID*`                                                                                                                                                |
| 09-2x | Cosmos-2B T=2 video-LoRA, online, no aug          | 2,400, L20, σ 80                 |     14.52 |  5.15 |  6.92 | ~204 ms | `INVALID` B2/B3: LoRA injected at alpha 32 (trained 16); eval cache built at alpha 16. Run `outputs/train/v1-cosmos2b-t2-online-unaugmented-smolexpert` |
| 08-24 | SmolVLA 450M, 1 h (29.2k steps)                   | SigLIP, 64                       |     14.83 |  4.85 |  6.64 | ~120 ms | `VALID` baseline (no augmentation, 1 h budget — not matched to VAM arms)                                                                                |
| 09-06 | Cosmos 7B base, re-benchmark stage 1              | L14+20, 64                       |     15.98 |  5.58 |  7.77 | —       | `VALID*` real VAE, but unstandardized intermediate cache; retired pending clean run                                                                     |
| 09-26 | FLUX 3 Action SO-101 + LoRA (r/α 32, BFL `lora.json`), wrist duplicated (W1), 10 Hz contract, EMA @ 2500 updates | 7B WAM, 8-step history + past commands | 15.74 | 0.89 | 3.44 | ~1.0 s (base) | `VALID` 88 anchors, single seed, not plateaued (26.39 / 19.96 / 17.37 / 15.74 @ 625 / 1250 / 1875 / 2500 updates). H1/F5 not comparable (conditions on past commands; command-repeat = 18.19 / 1.53 / 4.46). [plan §7](./plan_flux3_action.md#7-results-2026-09-25) |
| 09-26 | FLUX 3 Action SO-101, zero-shot, 10 Hz contract, cube quantiles | same | 30.40 | 0.82 | 3.46 | ~1.0 s | `VALID`; worse than holding the last command |
| 08-19 | state-repeat                                      | —                                |     18.86 |  0.89 |  4.45 | —       | trivial baseline                                                                                                                                        |
| 08-19 | mean action                                       | —                                |     30.15 | 12.40 | 18.20 | —       | trivial baseline                                                                                                                                        |

## 2. Scale-100 (cube v2)

`Orellius/cube_out_of_box_v2@5d0325cc`, train eps 0–31 + 40–89. **Eval-1** = v1 val eps 32–39 (88 anchors), **Eval-2** = eps 90–99 (51 anchors). All `PROVISIONAL` because of the v2 metadata/row mismatch.

| Date  | Model                                                 | Train data       | Eval-1 (H1)      | Eval-2 (H1)      | Validity / note                                                      |
| :---- | :---------------------------------------------------- | :--------------- | :--------------- | :--------------- | :------------------------------------------------------------------- |
| 09-11 | Cosmos 3 Edge video-LoRA + SmolExpert (offline cache) | v2 0–31, 40–89   | **13.49** (3.62) | **17.20** (4.92) | `PROVISIONAL`                                                        |
| 09-12 | Cosmos 3 Edge video-LoRA, online + legacy aug         | v2 0–31, 40–89   | 15.48 (4.60)     | 17.23 (5.08)     | `PROVISIONAL`                                                        |
| 09-12 | Cosmos 3 Edge video-LoRA, online + legacy aug         | **v1** 0–31 only | 13.82 (4.64)     | 24.66 (10.00)    | `VALID` for Eval-1; Eval-2 is cross-dataset                          |
| 09-11 | Cosmos 14B base, L18+30, FP8 streaming                | v2               | 14.02 (4.57)     | 19.41 (6.08)     | `PROVISIONAL`                                                        |
| 09-11 | FLUX.2 klein base, junction tap                       | v2               | 15.75 (5.18)     | 20.45 (6.56)     | `PROVISIONAL`; verify cache `vae_sha256` (B1 follow-up)              |
| 09-11 | FLUX.2 klein video-LoRA                               | v2               | 15.89 (5.07)     | 21.02 (6.77)     | `PROVISIONAL`; verify cache `vae_sha256`                             |
| 09-13 | Cosmos-2B T=2 video-LoRA, online + aug                | v2               | 15.57 (4.84)     | 18.71 (6.32)     | `INVALID` B2/B3 (alpha 32 vs 16; eval cache from different features) |
| 09-13 | FLUX.2 klein base, online + aug                       | v2               | 18.42 (10.93)    | 21.59 (10.74)    | `INVALID` B1 (bilinear pseudo-latents in online path)                |
| 09-11 | Cosmos 14B video-LoRA (FP8 QLoRA)                     | v2               | —                | —                | never ran (was "QUEUED")                                             |

## 3. Sort-cubes (different task — not comparable to §1/§2)

`Orellius/so101_sort_cubes_no_top@<unpinned>`, train eps 0–67, held-out 68–76. All runs before 09-24 did not record a dataset revision.

| Date  | Model                                                                | Recipe                    |                 Full-30 |    H1 |    F5 | Validity / note                                                                                                              |
| :---- | :------------------------------------------------------------------- | :------------------------ | ----------------------: | ----: | ----: | :--------------------------------------------------------------------------------------------------------------------------- |
| 09-2x | Cosmos 3 Edge video-LoRA, online, joint actions                      | BS 8 (4×2), 3 Euler steps |                   15.45 |  4.46 |  6.11 | `VALID` (revision unpinned). Run `outputs/train/sort-cubes-cosmos3-edge-lora-online-unaugmented`                             |
| 09-2x | Cosmos 3 Edge, online, effective BS 64                               | 8×8, LR 1e-3              |                   14.81 |  4.31 |  5.79 | `VALID` (revision unpinned; artifact verified on abakus `outputs/train/sort-cubes-cosmos3-edge-optimized-bs64`, best @ 8.5k/10k, used cube-v1 LoRA + prompt) |
| 09-2x | Cosmos 3 Edge, online, 5D Cartesian (xyz+pitch+roll), single-step IK | BS 8                      |                   42.74 | 32.58 | 41.67 | `VALID` (revision unpinned); first iteration, yaw drift. Local run `outputs/train/sort-cubes-cosmos3-edge-lora-cartesian-ik` |
| 09-2x | Cosmos 3 Edge, online, 7D Cartesian (xyz+rotvec), iterative IK       | BS 8                      |         24.58 (55.8 mm) |     — |     — | `UNVERIFIED` no local artifact (overwrote the same run dir on abakus?). Still far worse than joint space                     |
| 09-2x | SmolVLA, fresh                                                       | lerobot-train             | 17.27 @50k, 17.73 @155k |     — |     — | `UNVERIFIED` locally                                                                                                         |
| —     | `Orellius/..._smolvla_base_100k` (HF)                                | random 8 % frame holdout  |                   6.515 |     — |     — | `NON-COMPARABLE` trained on the held-out episodes                                                                            |
| 09-26 | Cosmos 3 Edge **sort-cubes PEFT video-LoRA** (10 fps clips, task prompt) + SmolExpert, online, no aug | BS 8 (4×2), 3 Euler steps | 15.54 | 4.35 | 6.02 | `VALID` @`a7c156de`, best @ 17.5k/25k. vs 15.45 with the cube-v1 adapter → task-matched adapter gives **no gain** (within noise). Run `outputs/train/c3peft_sort_cubes` |
| 09-25 | **FLUX 3 Action SO-101 + LoRA** (r/α 32, BFL `lora.json`), wrist duplicated (W1), EMA @ 1875 updates | batch 8, 4 Euler, guidance 3 | **14.20** | 0.47 | 1.56 | `VALID` @`a7c156de`, single seed; final @2500 = 14.54. H1/F5 not comparable (conditions on past commands). [plan §7](./plan_flux3_action.md#7-results-2026-09-25) |
| 09-25 | FLUX 3 Action SO-101, **zero-shot**, wrist duplicated (W1), state quantiles recalibrated | 4 Euler, guidance 3 | 19.24 | 0.51 | 2.14 | `VALID` @`a7c156de`, 405 anchors. Conditions on the 8 recorded past commands → compare H1/F5 to `command_repeat` (19.69 / 1.40 / 4.11), not to the rows above. [plan §7](./plan_flux3_action.md#7-results-2026-09-25) |
| 09-25 | FLUX 3 Action SO-101, zero-shot, checkpoint quantiles            | same                      |                   23.71 |  1.18 |  3.53 | `VALID`; our calibration is outside the checkpoint's state range (clipped)                                                   |
| 10-31 | Cosmos 3 Edge (frozen base) + SmolExpert, 30m fast recipe (stride 1, cached VAE, Triton) | BS 16, 3 Euler steps, 2,580 steps | 16.64 | 6.28 | 8.01 | `VALID` @`a7c156de`, Val Flow Loss `0.0688`, Arm `17.81°`, Grip `8.61`. HF: `Orellius/so101_sort_cubes_no_top_cosmos3_edge_smolexpert` |
| 10-31 | Cosmos 3 Edge **joint DoRA** (`r=16, α=32`, gen tower) + SmolExpert, 30m warm-start | BS 16, 3 Euler steps, 936 steps | **16.34** | **6.12** | **7.78** | `VALID` @`a7c156de`, Val Flow Loss `0.0712` (`0.0684` @ step 600; H1 `6.10` @ step 936), Arm `17.48°`, Grip `8.60`. HF: `Orellius/so101_sort_cubes_no_top_cosmos3_edge_dora_smolexpert` |
| 10-31 | **SmolVLA 450M** (`sort-cubes-smolvla-100k-exact`), evaluated on canonical val `68..72` (200 anchors, stride 5) | 100k steps, native 30 Hz (and 10 Hz subsampled) | **6.06** (`6.47` @10Hz) | **5.33** | **5.02** (`5.67` @10Hz) | `NON-COMPARABLE` @`a7c156de`: evaluated at 30 Hz (30 steps = `1.0 s` physical horizon vs `3.0 s` @ 10 Hz for VAM rows) and only on 5-ep slice `68..72` (dominated by static holding frames). See row below for exact 405-anchor 10 Hz comparison |
| 11-06 | **SmolVLA 450M** (`sort-cubes-smolvla-100k-exact`), **strict Protocol 1.0** (`68..76`, 405 anchors, 10 Hz physical horizon matched) | 35k steps (BS 32, ~1.12M samples), subsampled every 3rd step to match 10 Hz (`1.0 s` = 10 steps) | 16.96 (`1.0 s` max) | 6.09 | 11.51 | `VALID` @`a7c156de`: exact apples-to-apples against Cosmos 3 Edge BS 64 (`H1 = 4.31°`, `F5 @ 10Hz = 5.79°`, `1.0 s = 9.42°`, `3.0 s Full-30 = 14.81°`). Median latency `215.4 ms` (`2.7x` slower than compiled Cosmos 3 Edge `~80 ms`) |
| 11-06 | Cosmos 3 Edge **BS 64 + Joint Action-LoRA Warm-Start** (`gen_only`, `r=16, α=32`) | BS 32 (16×2), LR 5e-5, 3 Euler steps, warm-start from `sort-cubes-cosmos3-edge-optimized-bs64` | 14.87 | **4.27** | **5.73** | `VALID` @`a7c156de`, best @ 250/389 (`25m` wall-clock), Pooled F5 `5.81°`, Arm `15.87°`, Grip `8.22`. **New best short-horizon (`H1 = 4.27°`, `F5 = 5.73°`) on sort-cubes.** Run `outputs/train/sort-cubes-cosmos3-edge-bs64-joint-lora-warmstart` |
| 11-06 | Cosmos 3 Edge **BS 64 + Physics Photometric/Shadow Aug Warm-Start** | BS 32 (16×2), LR 5e-5, Planckian + cast shadow + gamma aug, warm-start from `sort-cubes-cosmos3-edge-optimized-bs64` | 14.86 | 4.32 | 5.79 | `VALID` @`a7c156de`, best @ 250/637 (`25m` wall-clock), Pooled F5 `5.87°`, Arm `15.86°`, Grip `8.12`, Val Flow Loss `0.0744`. Retains exact unaugmented accuracy while invariant to lighting/shadows. Run `outputs/train/sort-cubes-cosmos3-edge-bs64-photometric-aug-warmstart` |
| 11-06 | Cosmos 3 Edge **BS 64 + Real-Data Warm-Started EXPO Residual (`K=6`)** | Marek Score-Centered `ExpoResidualCritic` (`3,000` steps on real `0..67` anchors) | **14.78** | 4.67 | 6.04 | `VALID` @`a7c156de`, improves multi-candidate `Full-30` from `14.84° -> 14.78°` and `H1` from `4.72° -> 4.67°` (@ step 900). Artifact `outputs/sparse_seq_rl/expo_real_warmstart_bs64.pt` |
| 11-06 | `hubnemo/..._smolvla_lora_rank64_bs32_lr1e-3_steps10000_schedulerdecaylr1e-4` (and identical `..._lora_main_...`) | SmolVLA LoRA (`r=64, α=8`), BS 32, 10k steps, no aug, **3 Euler steps** (vs 10 Euler) | 12.58 (`12.26` @10Hz 1.0s) | 5.38 | **6.96** (`7.02` pooled; `9.41` @10Hz) | `NON-COMPARABLE` @`a7c156de`: **trained on full dataset (all 77 eps incl. `68..76`)**. 10 Euler steps degrades to `13.33 / 6.20 / 7.60` (`7.64` pooled). Even with train-on-val leakage, H1 (`5.38°`) is worse than clean held-out Cosmos 3 Edge BS 64 (`4.31°`) |
| 11-06 | `hubnemo/so101_smolvla_exp01_bs16_steps20k_imgtf` | SmolVLA LoRA (`r=64, α=8`), BS 16, 20k steps, LR 1e-4, ColorJitter+Sharpness+Affine, **3 Euler steps** (vs 10 Euler) | 16.43 (`15.96` @10Hz 1.0s) | 6.63 | 8.53 (`8.61` pooled; `11.80` @10Hz) | `NON-COMPARABLE` @`a7c156de`: **trained on full dataset (all 77 eps incl. `68..76`)** + image transforms. 10 Euler steps degrades to `17.15 / 7.56 / 9.22` (`9.28` pooled). Worse than clean held-out unaugmented SmolVLA (`8.01` F5) |

## 3b. Reward Detector & Sparse-Reward RL Post-Training (`SO101CubeBoxSparse-v0`, 10 Spawns)

### Real Dataset Success Detector (`Orellius/so101_sort_cubes_no_top`, 3-Way Split: Train `0..57`, Val `58..67`, Test `68..72` & `68..76`)

| Detector Architecture & Training Recipe | Val (`58..67`) Acc / Prec / Rec / F1 | Held-Out Test (`68..72`) Acc / Prec / Rec / F1 | Full Test (`68..76`) Acc / Prec / Rec / F1 | Empty-Box Look False-Positive Rate |
| :--- | :---: | :---: | :---: | :---: |
| `SingleFrameResNet50Detector` (Baseline) | — | `61.4%` / `28.6%` / `80.0%` / `0.426` | — | `60.0%` |
| `MultiFrameSuccessDetector` (Initial Heuristic) | — | `80.0%` / `68.8%` / `55.0%` / `0.611` | — | `15.0%` |
| `MultiFrameSuccessDetector` (**Boosted: Pan>32° Drop Window + 6-View Aug + MixView + 50% Proprio Dropout + Val Early Stop**) | **`100.0%` / `100.0%` / `100.0%` / `1.000`** | **`100.0%` / `100.0%` / `100.0%` / `1.000`** | **`99.2%` / `97.3%` / `100.0%` / `0.986`** | **`0.0%`** (`68..72`), `2.78%` (`68..76`) |
| *— Boosted Detector Evaluated 100% Blind to Joints (`seq_states = 0`)* | `99.3%` / `97.6%` / `100.0%` / `0.988` | **`100.0%` / `100.0%` / `100.0%` / `1.000`** | **`99.2%` / `97.3%` / `100.0%` / `0.986`** | **`0.0%`** |
| `SharedFeatureSequenceDetector` on **Cosmos 3 Edge LoRA** Tokens (`[24, 2048]`, 3 seeds) | **`100.0%` / `100.0%` / `100.0%` / `1.000`** | **`100.0%` / `100.0%` / `100.0%` / `1.000`** | **`98.15%` / `93.93%` / `100.0%` / `0.969`** | **`0.0%`** (`68..72`), `3.71%` (`68..76`) |
| `SharedFeatureSequenceDetector` on **SmolVLA SigLIP** Tokens (`[24, 960]`, 3 seeds) | **`100.0%` / `100.0%` / `100.0%` / `1.000`** | **`98.57%` / `95.38%` / `100.0%` / `0.976`** | **`98.94%` / `96.54%` / `100.0%` / `0.982`** | **`0.0%`** (`68..72` & `68..76`) |

### 10-Spawn Closed-Loop RL Post-Training Ablation: Vanilla EXPO vs. Score-Centered EXPO (Dr. GRPO)

Evaluated across 10 randomized cube spawns ($\pm 3.5\text{ cm}$) from a 60-episode exploration buffer (`6` rollouts/spawn):

| Policy / Post-Training Method | Reward Signal | Training Steps | **Closed-Loop Success (`10` Spawns)** | Successful Spawn IDs | Mean Min EE-to-Cube (`cm`) | Mean Max Cube Lift (`cm`) | Mean Shaped Reward |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **Zero-Shot Base Policy** (`Cosmos 3 Edge LoRA + SmolExpert`) | — | `0` | **`0/10 (0%)`** | `[]` | `3.79 cm` | `0.91 cm` | `0.231` |
| **Vanilla EXPO** (Uncentered AWR, $w_i = \exp(\beta R_i)$) | **100% Learned Sequence Detector** | `2,500` (`1x`) | **`7/10 (70%)`** | `[0, 2, 3, 5, 7, 8, 9]` | `1.95 cm` | `16.88 cm` | `0.811` |
| **Score-Centered EXPO (Dr. GRPO)** ($A_{g,i} = R_{g,i} - \bar{R}_g$) | **100% Learned Sequence Detector** | `2,500` (`1x`) | **`7/10 (70%)`** | `[0, 1, 3, 4, 5, 7, 8]` | `2.06 cm` | `15.49 cm` | `0.768` |
| **Vanilla EXPO** (Uncentered AWR, $w_i = \exp(\beta R_i)$) | **Sparse Oracle (`0/1`)** | `2,500` (`1x`) | **`7/10 (70%)`** | `[0, 2, 3, 5, 7, 8, 9]` | `1.98 cm` | `15.96 cm` | `0.834` |
| **Score-Centered EXPO (Dr. GRPO)** ($A_{g,i} = R_{g,i} - \bar{R}_g$) | **Sparse Oracle (`0/1`)** | `2,500` (`1x`) | **`8/10 (80%)`** | `[0, 1, 2, 3, 4, 5, 7, 8]` | `1.68 cm` | `17.38 cm` | `0.853` |
| **Vanilla EXPO** (Uncentered AWR, $w_i = \exp(\beta R_i)$) | **100% Learned Sequence Detector** | `5,000` (`2x`) | **`7/10 (70%)`** | `[0, 1, 2, 3, 4, 5, 9]` | `2.07 cm` | `17.80 cm` | `0.846` |
| **Score-Centered EXPO (Dr. GRPO)** ($A_{g,i} = R_{g,i} - \bar{R}_g$) | **100% Learned Sequence Detector** | `5,000` (`2x`) | **`9/10 (90%)`** | `[0, 1, 2, 3, 4, 5, 7, 8, 9]` | **`1.68 cm`** | **`19.32 cm`** | **`0.922`** |
| **Vanilla EXPO** (Uncentered AWR, $w_i = \exp(\beta R_i)$) | **Sparse Oracle (`0/1`)** | `5,000` (`2x`) | **`8/10 (80%)`** | `[0, 1, 2, 3, 5, 7, 8, 9]` | `1.83 cm` | `18.72 cm` | `0.884` |
| **Score-Centered EXPO (Dr. GRPO)** ($A_{g,i} = R_{g,i} - \bar{R}_g$) | **Sparse Oracle (`0/1`)** | `5,000` (`2x`) | **`9/10 (90%)`** | `[0, 1, 2, 3, 4, 5, 7, 8, 9]` | **`1.53 cm`** | **`19.64 cm`** | **`0.922`** |
| **Cosmos 3 Edge + Marek Score-Centered EXPO** (`arXiv:2609.20807` sampler score subtraction) | **Shared Sequence Detector** | `5,000` (`2x`) | **`8/10–9/10 (80–90%)`** | `[0, 1, 2, 3, 4, 5, 7, 8, 9]` | **`1.70 cm`** | **`17.50 cm`** | **`0.880`** |
| **SmolVLA Zero-Shot Base Policy** (`450M`) | — | `0` | **`0/10 (0%)`** | `[]` | `3.35 cm` | `1.02 cm` | `0.245` |
| **SmolVLA + Marek Score-Centered EXPO** (`arXiv:2609.20807` sampler score subtraction) | **SmolVLA Shared Sequence Detector** | `5,000` (`2x`) | **`9/10 (90%)`** | `[0, 1, 2, 3, 4, 5, 7, 8, 9]` | **`1.79 cm`** | **`18.98 cm`** | **`0.915`** |

## 4. Video prediction (PSNR/SSIM) — diagnostic only

PSNR/SSIM reward blurry, conservative predictions and were shown to disagree with task plausibility (diary 08-18). Not a backbone-quality metric.

| Model                                              | Frames | PSNR / SSIM           | Validity                                                                                        |
| :------------------------------------------------- | :----- | :-------------------- | :---------------------------------------------------------------------------------------------- |
| Cosmos 14B base / QLoRA (ep 32, 161 frames)        | 161    | 18.28 / 0.7803 (both) | `INVALID` B4: renderer used a one-step heuristic latent update; base and LoRA outputs identical |
| Cosmos 7B (17 frames)                              | 17     | 28.21 / 0.877         | `INVALID` pseudo-latent renderer                                                                |
| FLUX.2 klein (17 frames)                           | 17     | 12.56 / 0.491         | `INVALID` pseudo-latent renderer                                                                |
| Cosmos-2B, 5-frame conditioning vs static baseline | 16     | 14.55 vs 24.11 (ep 0) | `VALID` diagnostic (diary 08-18)                                                                |

## 5. Retraction ledger (pre-audit, 2026-09-05/06)

| Date  | Run                                                                    | Reported | Reason                                                                |
| :---- | :--------------------------------------------------------------------- | :------- | :-------------------------------------------------------------------- |
| 09-05 | Cosmos 7B base (L14+20)                                                | 9.46     | `INVALID` frame-level `random_split` leak + bilinear pseudo-latents   |
| 09-05 | Cosmos 7B video-LoRA                                                   | 9.75     | `INVALID` same                                                        |
| 09-05 | Cosmos 14B base (L18+30)                                               | 9.97     | `INVALID` frame-level leak; ignored `action_is_pad`                   |
| 09-05 | FLUX.2 video-LoRA                                                      | 10.33    | `INVALID` leak + pseudo-latents                                       |
| 09-05 | FLUX.2 base junction                                                   | 11.45    | `INVALID` leak + pseudo-latents                                       |
| 09-06 | Cosmos 14B base "converged"                                            | 10.99    | `INVALID` leak                                                        |
| 09-06 | Cosmos 14B QLoRA                                                       | 11.08    | `INVALID` leak                                                        |
| 08-20 | Early unadapted Cosmos-2B                                              | 26.12    | `NON-COMPARABLE` uncalibrated early config; features not reproducible |
| 09-08 | `grand_evaluation_summary.json` from `run_full_autonomous_pipeline.sh` | —        | `INVALID` hard-coded placeholder numbers                              |
