# Video Action Model (VAM) Strategic Research Roadmap & Living Status

**Source of Truth:** Strategic Priorities, Research Bets, and Implementation Plan
**Repository:** `lerobot-video-vam` (Remote: `/home/anton/lerobot-video-vam` on `abakus`)
**Target Task:** High-Precision Robotic Manipulation (SO-101 Open-Source Arm, `cube_out_of_box` task)
**Dataset:** `hubnemo/cube_out_of_box_dataset` (6,536 frames, 40 episodes, 10 fps, revision `243370c3c08bcbd860133c4a0d658ea7c1d2e77e`)

---

## 1. Executive Summary & Core Research Questions

The central question of Video-VAM is whether generative video diffusion and flow-matching backbones provide privileged visual-spatial representations for physical robot control compared to standard 2D vision encoders.

### Key Empirical Findings to Date:

- **Direct Manifold Distillation vs. Full Inference:**
  - Full-window video diffusion ($T=16$, Cosmos-Predict2-2B with 4x temporal downsampling) achieved **13.06** validation RMSE under Protocol 1.0 (at ~1,200 ms latency).
  - Fast 2-frame inference ($T=2$, ~204 ms uncompiled) dropped accuracy to **13.74**.
  - Direct student distillation from the $T=16$ Layer-20 spatiotemporal manifold into a $T=2$ student LoRA without intermediate linear projection heads achieved **13.15** validation RMSE.
- **Oracle & Gap Closure Clarifications:**
  - An empirical baseline trained directly on the teacher's ground-truth `cond_frames` reached **13.08**. This is an empirical reference point, **not a mathematical ceiling**.
  - Relative to the 13.74 baseline and the 13.08 empirical reference, the distilled model (13.15) achieves an **89.4% gap closure** ($\frac{13.74 - 13.15}{13.74 - 13.08} = 0.894$), subject to evaluation seed uncertainty (earlier claims of 98.9% gap closure were mathematically inaccurate).
- **Scale of Representation Gains:**
  - Comparing frozen SmolExpert (**13.65**) against legacy prefix (**13.81**) and adapted Video-LoRA (**13.06**) shows an empirical advantage of approximately **0.6–0.8 mixed RMSE units** on this benchmark. This represents a meaningful, but confounded, incremental gain—**not an error halving**.
  - The historical unadapted 26.12 baseline was trained under different early configurations and is **not directly comparable**.

---

## 2. Current Implementation & Verification Status

> **Operational Retraining Plan:** For the exact executable overnight sequence to retrain missing T=2 models and execute the non-placeholder queue, see [`docs/video_vam_execution_plan.md`](./video_vam_execution_plan.md).

- **Status:** **In Review / Correctness Implementation Active**.
  - Agents are currently implementing and standardizing the unified `BaseVAMExtractor`, native causal VAE pipelines, and strict Protocol 1.0 guards.
  - **No training or evaluation jobs are currently running or scheduled.**
  - All historical runs involving frame-level random splits or bilinear pseudo-latents are formally marked invalid.
- **Next Operational Milestone:** Execute formal unit-test suite verification, followed by a clean, automated re-benchmarking queue across Cosmos 3, Cosmos 7B, Cosmos 14B, and FLUX.2 backbones under strict Protocol 1.0.

---

## 3. Reference Quantitative Scoreboard (Protocol 1.0 Held-Out Episodes 32–39)

> **Important Metric Definition:** All RMSE values report the standard mixed aggregate metric: 5 arm joints in degrees ($^\circ$) and the gripper in range $[0, 100]$. This is a mixed-unit aggregate convention, **not pure degrees**.
>
> - **Horizon-1 (H1):** Action prediction error at timestep offset 0 ($t$).
> - **First-5:** Mean action prediction error across timestep offsets $0 \dots 4$ ($t \dots t+4$).
> - **Full-30:** Aggregate RMSE across the full 30-step prediction chunk (offsets $0 \dots 29$).

| Model / Architecture                                      | Tokens to Policy                 | Feature Representation              | Full-30 Mixed RMSE | Immediate (H1, $t$) | First-5 ($t..t+4$) | Status / Integrity                                                       |
| :-------------------------------------------------------- | :------------------------------- | :---------------------------------- | :----------------- | :------------------ | :----------------- | :----------------------------------------------------------------------- |
| **Cosmos-Predict2-2B (Video-LoRA, $T=16$, pool2)**        | 4,800 ($16 \times 15 \times 20$) | Layer 20, 2048-dim                  | **13.06**          | 4.76                | 6.19               | Historical reported result; matched re-score pending                     |
| **Cosmos-Predict2-2B (Teacher `cond_frames` Ref)**        | 2,400 ($2 \times 30 \times 40$)  | Layer 20, 2048-dim                  | **13.08**          | 4.97                | 6.40               | Empirical Reference (not ceiling)                                        |
| **Cosmos-Predict2-2B ($T=2$ Direct Distilled)**           | 2,400 ($2 \times 30 \times 40$)  | Layer 20, 2048-dim                  | **13.15**          | 4.58                | 6.04               | Historical reported result; matched re-score pending (89.4% gap closure) |
| **Cosmos-Predict2-2B (Frozen Pretrained, pool2)**         | 4,800 ($16 \times 15 \times 20$) | Layer 20, 2048-dim                  | **13.65**          | —                   | —                  | Historical reported result; matched re-score pending                     |
| **Cosmos-Predict2-2B ($T=2$ Undistilled Baseline)**       | 2,400 ($2 \times 30 \times 40$)  | Layer 20, 2048-dim                  | **13.74**          | 4.92                | 6.34               | Historical reported result; matched re-score pending                     |
| **LTX-Video-2.5 22B (Unpooled Layer 34)**                 | 2,400 ($15 \times 20$ / frame)   | Block 34, 4096-dim                  | **13.84**          | 4.55                | 6.24               | Historical reported result; matched re-score pending                     |
| **Cosmos 3 Edge (Pure 600 Visual Tokens)**                | 600 ($2 \times 15 \times 20$)    | Layer 20 `und_seq`, 2048-dim        | **14.26**          | 4.67                | 6.34               | Protocol 1.0 (Zero-shot base; LoRA rerun pending)                        |
| **SmolVLA 450M Baseline (Pretrained Vision)**             | 64 tokens                        | SigLIP 2D vision                    | **14.83**          | —                   | —                  | Pretrained VLA Reference                                                 |
| _Cosmos 7B / 14B / FLUX.2 Historical Runs (9.46°–10.99°)_ | Various                          | Random Frame Split / Pseudo-Latents | _INVALID_          | —                   | —                  | **RETRACTED (Data Leakage / No VAE)**                                    |

---

## 4. Key Hypotheses and Physical Insights

1. **Direct Manifold Distillation vs. Linear Projection Heads:**
   In the recorded runs, supervising student LoRA features directly against teacher intermediate layers ($T=2 \to H_{\text{teacher}}^{0..1}$) improves direct coordinate alignment ($\text{CosSim} = 0.9732$), whereas the auxiliary-head run had low direct cosine alignment. Low cosine alone does not prove information loss: invertible coordinate changes preserve information; compare downstream decodability.
2. **Dataset Shift & Changepoint Analysis:**
   - Analysis of `hubnemo/cube_out_of_box_dataset` demonstrates a marked kinematic shift around Episode 20–24.
   - Reported early-versus-late regime differences are +32.5° on Joint 2 and -44.3° on Joint 3. The separate train-versus-validation differences are +19.96° and -29.47°.
   - Validation (episodes 32–39) is entirely in the late regime. **This is a hypothesis of operator or hardware recalibration during data collection**, not an intentional planned domain shift.
3. **LTX-2.5 Spatial Resolution Hypothesis:**
   - LTX-2.5 uses a $32\times$ spatial compression VAE ($15 \times 20$ latents at 480p) compared to Cosmos $8\times$ spatial compression ($60 \times 80$).
   - The hypothesis that severe spatial compression degrades fine contact resolution is plausible, but **remains a research hypothesis**, not an established theorem.
4. **Cosmos 3 Edge Architectural Specifics:**
   - Local configuration: 2048 hidden dim, 16 attention heads, 28 transformer layers, 3.37B measured dense parameters (vs public 4B label).
   - Generative tower vs. understanding tower: LoRA fine-tuning on `gen_seq` with flow-matching velocity loss introduces a representation mismatch when extracting un-noised `und_seq` tokens at inference.

---

## 5. Strategic Roadmap & Execution Priorities

```
┌────────────────────────────────────────────────────────────────────────┐
│ Priority 1: Correctness, Rebuild & Protocol 1.0 Re-Benchmarking         │
│  ▶ Unify extractors under BaseVAMExtractor with genuine causal VAEs    │
│  ▶ Enforce per-sample seed batch invariance and fail-closed validation │
│  ▶ Re-extract and re-benchmark Cosmos 3, 7B, 14B, and FLUX.2 under P1.0 │
│  ▶ Verify code via unit tests (main verification pending)              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Priority 2: Multi-Frame Student Architecture (Primary Research Bet)    │
│  ■ Design compact multi-frame student DiTs (not merely T reduction)    │
│  ■ Distill multi-step spatio-temporal dynamics into edge-sized models  │
│  ■ Maintain high-capacity spatial grounding at < 50 ms latency        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Priority 3: Targeted Controlled Experiments                            │
│  ■ Action-conditioned latent prediction (controlled test of dynamics)  │
│  ■ Exploratory gated: Temporal recurrent memory mechanisms             │
│  ■ Action sampler distillation (reducing flow-matching ODE steps)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Priority 4: Physical Data Collection & Recovery Demonstration          │
│  ■ Collect targeted recovery trajectories and physical failure modes   │
│  ■ Re-calibrate SO-101 hardware joint limits and coordinate frames     │
│  ■ RL fine-tuning: only after reward formulation and recovery data    │
│    (explicit failure labels helpful but practical reward signal key)   │
└────────────────────────────────────────────────────────────────────────┘
```

### Strategic Bet Definitions:

1. **Primary Bet — Smaller Multi-Frame Student Architecture:** Rather than simply truncating teacher time horizons ($T=16 \to T=2$), design compact, dedicated multi-frame student backbones that natively compress temporal dynamics into lightweight architectures.
2. **Exploratory Bet — Temporal Memory:** Investigate recurrent or state-space memory layers to maintain long-horizon context across chunk transitions, gated behind empirical verification.
3. **Controlled Experiment — Action-Conditioned Latent Prediction:** Evaluate whether explicit action conditioning during world-model latent rollout improves future frame prediction and downstream policy stability.
4. **Data Strategy — Recovery Demos:** Collect explicit physical recovery and disturbance demonstrations to provide the policy with out-of-distribution correction capabilities.
5. **Downstream Policy — Sampler Distillation:** Distill the 10-step flow matching action generation head into 1–2 step deterministic generators.
6. **Reinforcement Learning:** Introduce RL exploration only after establishing well-behaved reward signals and physical recovery demonstration distributions.

---

## 6. Historical Roadmap Archive (Preserved for Provenance)

> _The sections below document earlier roadmap snapshots preserved for historical traceability. Active priorities are governed by Sections 1–5 above._

### Historical Snapshot: Early September 2026 Exploration

```
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Representation Distillation Breakthrough (COMPLETED)           │
│  ✔ Fixed 4-frame lag bug and aligned causal window contract            │
│  ✔ Succeeded in direct Layer-20 manifold distillation (13.15° RMSE)    │
│  ✔ Evaluated teacher cond_frames reference (13.08° RMSE)                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Cosmos 3 Edge Initial Exploration (SUPERSEDED BY AUDIT)       │
│  ▶ Explored nvidia/Cosmos3-Edge pure 600 vision tokens (14.26° Base)   │
│  ▶ Video-LoRA on gen_seq evaluated; revealed tower mismatch with und   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Real-Time Inference Optimization Target                       │
│  ☐ torch.compile and RPC deployment pipelines                          │
│  ☐ End-to-end latency profiling across SO-101 control loops           │
└────────────────────────────────────────────────────────────────────────┘
```
