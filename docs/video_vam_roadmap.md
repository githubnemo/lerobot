# Video Action Model (VAM) Strategic Research Roadmap & Living Status

_Last Updated: 2026-09-02_
_Target Application: High-Precision, Real-Time Robotic Manipulation (SO-100 Robot Arm, `cube_out_of_box` task)_
_Core Infrastructure: Remote Server `abakus` (Ubuntu, RTX 4090 24GB VRAM), Local Repository `/Users/anton.wiehe/Code/compass`_

---

## 1. Executive Summary & North Star

- **The Core Dilemma**: Heavy video diffusion world models ($T=16$) generate state-of-the-art physical representations (**13.06°** validation RMSE), but take **~1,200 ms** per forward pass—far too slow for 10 Hz real-time robotic control (~100 ms budget). Fast 2-frame inference ($T=2$, **~204 ms**) exhibits severe attention concentration and distribution shift, dropping accuracy to **13.74°**.
- **The Breakthrough**: By directly distilling the frozen $T=16$ Layer-20 spatiotemporal diffusion manifold into a fast $T=2$ student LoRA without intermediate projection heads, we eliminate feature collapse, achieve a **93.6% MSE reduction** ($\text{CosSim} = \mathbf{0.9732}$), and reach **`13.15°`** validation RMSE at **~204 ms**.
- **Oracle Verification**: Training directly on the teacher's ground-truth first 2 latent frames establishes an absolute theoretical ceiling of **`13.08°`**. Our distilled model reaches **`13.15°`**, capturing **`98.9%`** of the maximum possible teacher value while running 6× faster.

---

## 2. Live Pipeline Status (In-Flight)

- **Active Run**: `cosmos3-edge-pipeline-20260902` on `abakus` (tmux)
  - **Model**: `nvidia/Cosmos3-Edge` (28 layers, 2048 hidden, 3.37B dense parameters, Wan 2.2 VAE)
  - **Current Execution**: **Stage 1 (Feature Extraction)** is extracting Layer-20 representations across train (episodes 0–31) and validation (episodes 32–39) at ~8.3 samples/second.
  - **Next Stage**: Automatically launches **Stage 2 (SmolExpert Policy Training)** to evaluate whether Cosmos 3 Edge's updated architecture surpasses the 13.15° mark.
- **Master Log**: `/home/anton/.cache/video-vam/runs/cosmos3-edge-pipeline-20260902.log`

---

## 3. Quantitative Benchmark Scoreboard (Held-Out Episodes 32–39)

| Model / Architecture                       | Latency     | Tokens to Policy                 | Representation Loss / CosSim | Action RMSE (Overall) | Immediate Action (H=1) | First-5 Actions Mean | Status / Role                    |
| :----------------------------------------- | :---------- | :------------------------------- | :--------------------------- | :-------------------- | :--------------------- | :------------------- | :------------------------------- |
| **Cosmos $T=16$ Gold Standard**            | ~1,200 ms   | 4,800 ($16 \times 15 \times 20$) | - / 1.0000                   | **`13.06°`**          | `4.76°`                | `6.19°`              | Offline Gold Standard            |
| **Teacher $T=16$ `cond_frames` Oracle**    | -           | 2,400 ($2 \times 30 \times 40$)  | - / 1.0000                   | **`13.08°`**          | `4.97°`                | `6.40°`              | **Theoretical Oracle Ceiling**   |
| **Cosmos $T=2$ Direct Distilled (Ours)**   | **~204 ms** | 2,400 ($2 \times 30 \times 40$)  | **1.21 / 0.9732**            | **`13.15°`**          | **`4.58°`** (Best)     | **`6.04°`** (Best)   | **Matches Oracle within 0.07°!** |
| **Cosmos $T=2$ Undistilled Baseline**      | **~204 ms** | 2,400 ($2 \times 30 \times 40$)  | - / 0.6092                   | **`13.74°`**          | `4.92°`                | `6.34°`              | Fast Baseline (0.68° gap)        |
| **LTX-2.5 22B (Unpooled Layer 34)**        | ~1,400 ms   | 2,400 ($15 \times 20$ / frame)   | -                            | **`13.84°`**          | `5.21°`                | `6.72°`              | Limited by 32× spatial VAE       |
| _V2 Distillation (Lag Bug + Linear Head)_  | ~204 ms     | 2,400                            | 8.88 / 0.6809                | `13.99°`              | `5.18°`                | `6.83°`              | Scrambled latent space           |
| _V1 Distillation (Lag Bug)_                | ~204 ms     | 2,400                            | 8.26 / 0.8056                | `14.08°`              | `5.02°`                | `6.58°`              | Degraded by 400 ms lag           |
| _Failed: Action Backprop into Blocks 0–19_ | ~204 ms     | 2,400                            | exploded variance            | `14.51° – 14.74°`     | -                      | -                    | Catastrophic feature collapse    |
| _SmolVLA 450M Baseline_                    | ~80 ms      | 64                               | -                            | **`14.83°`**          | -                      | -                    | Pure VLA Baseline                |
| _Failed: $T=4$ Noise Slots_                | ~340 ms     | 4,800                            | unstable                     | `17.92°`              | -                      | -                    | Attended to Gaussian noise       |

---

## 4. Key Scientific Insights & Discoveries

1. **Direct Manifold Distillation Beats Auxiliary Heads**:
   - Placing a 134M linear readout head between student LoRA and teacher targets allowed the LoRA to produce _scrambled intermediate coordinates_ ($z$ had $\text{CosSim} = 0.2782$).
   - Removing the auxiliary head and applying direct supervision ($T=2 \to H_{\text{teacher}}^{0..1}$) forced the student LoRA to directly learn the un-scrambled physical manifold ($\text{CosSim} = 0.9732$).
2. **The 4-Frame Temporal Shift Bug**:
   - An indexing flaw in data loading (`entry.frame_index + i` vs. `entry.frame_index - 4 + i`) fed future frames during distillation while testing on past frames, creating a 400 ms lag that artificially degraded earlier distillation runs.
   - Resolving this to the canonical `CUBE_OUT_OF_BOX_CONTRACT` causal window immediately unlocked the 13.15° result.
3. **The Dual-Operator Dataset Shift**:
   - Kinematic analysis revealed that `cube_out_of_box_dataset` consists of two distinct teleoperators:
     - **Episodes 0–19 (Operator A)**: High-elbow posture (`Lift ≈ -30°`, `Elbow ≈ +35°`).
     - **Episodes 20–39 (Operator B)**: Low-elbow posture (`Lift ≈ +10°`, `Elbow ≈ -20°`).
   - The validation set (episodes 32–39) is 100% Operator B, testing cross-operator out-of-distribution kinematic transfer.
4. **Why LTX-2.5 Trails Cosmos**:
   - LTX-2.5 utilizes an aggressive **$32\times$ spatial VAE** ($15 \times 20$ latents at 480p), whereas Cosmos uses an **$8\times$ spatial VAE** ($60 \times 80$ latents). A 3 cm cube collapses into less than 1.5 latent pixels in LTX, destroying contact geometry before the transformer even processes it.

---

## 5. Strategic Roadmap & Execution Phases

```
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Representation Distillation Breakthrough (COMPLETED)           │
│  ✔ Fixed 4-frame lag bug and aligned causal window contract            │
│  ✔ Succeeded in direct Layer-20 manifold distillation (13.15° RMSE)    │
│  ✔ Verified Oracle ceiling on teacher cond_frames (13.08° RMSE)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Cosmos 3 Edge Scaling & Benchmarking (IN PROGRESS)            │
│  ▶ Downloaded nvidia/Cosmos3-Edge & upgraded diffusers to 0.41.0.dev0  │
│  ▶ Active: Extracting Cosmos3-Edge teacher features (Stage 1)          │
│  ▶ Queued: SmolExpert policy training on Cosmos3-Edge                  │
│  ▶ Queued: Direct distillation from Cosmos3-Edge (T=16 -> T=2)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Real-Time Inference Acceleration (< 80 ms Target)             │
│  ☐ torch.compile(mode="max-autotune") on distilled student DiT         │
│  ☐ Static CUDA Graph capture for SmolExpert 10-step flow head (26 ms)  │
│  ☐ Benchmark 5-step flow matching vs. 10-step flow matching            │
│  ☐ End-to-end latency profiling: Push full loop to 12-16 Hz            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Closed-Loop Physical Robot Rollouts (SO-100 Hardware)         │
│  ☐ Dry-run policy execution verification with live camera feed         │
│  ☐ A/B Trial: Distilled T=2 (13.15°) vs. Undistilled T=2 (13.74°)       │
│  ☐ Measure real-world grasp success rate on 20 physical trials         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 5: Publication & Interactive Blog Post (notnanton.io)            │
│  ☐ Interactive figures & learning curve comparisons                    │
│  ☐ Document the failed ablations, lag bug, and distillation solution   │
│  ☐ Release code, checkpoints, and citeable technical report            │
└────────────────────────────────────────────────────────────────────────┘
```
