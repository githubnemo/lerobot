# Video-VAM: Technical Report on World Model Backbones for Physical AI Policy Learning

**Status:** Authoritative Comprehensive Synthesis & Integrity Audit
**Date:** September 2026
**Repository:** `lerobot-video-vam` (`/home/anton/lerobot-video-vam` on `abakus`)
**Dataset:** `hubnemo/cube_out_of_box_dataset` (SO-101 Open-Source 6-DoF Arm, 6,536 Frames, 40 Demonstration Episodes at 10 FPS, Revision `243370c3c08bcbd860133c4a0d658ea7c1d2e77e`)
**Hardware Target:** Single Node (NVIDIA GeForce RTX 4090 24GB, Ubuntu Linux)

---

## 1. Executive Summary

The **Video-VAM (Video Action Model)** research initiative investigates the utility of large-scale pretrained video diffusion and flow-matching backbones as foundational visual-spatial representations for robotic manipulation. Rather than relying solely on 2D vision encoders (e.g. SigLIP) or end-to-end vision-language-action (VLA) training, Video-VAM taps intermediate spatiotemporal feature manifolds extracted from Diffusion Transformers (DiTs).

This technical report provides the architectural taxonomy, conditioning geometries, representation extraction mechanics, and downstream policy benchmarks across five model backbones:

1. **Cosmos-Predict2-2B** (Bidirectional DiT with causal video VAE)
2. **Cosmos 3 Edge (3.37B Dual-Pathway Mixture-of-Transformers)**
3. **Cosmos-1.0-Diffusion-7B** (3D Spatiotemporal DiT)
4. **FLUX.2 [klein] (9B)** (Multi-reference hybrid double/single-stream transformer)
5. **Cosmos-1.0-Diffusion-14B** (Memory-constrained block streaming & FP8 execution)

### Crucial Methodological Retraction & Scientific Audit Notice

In early September 2026, preliminary experiments reported sub-11° validation RMSE scores for Cosmos 7B (9.46°), Cosmos 14B (9.97°), and FLUX.2 (10.33°). **A rigorous methodological audit has proven these metrics to be invalid**:

- **Frame-Level Data Leakage:** Training scripts applied in-memory random splitting (`torch.utils.data.random_split(85%, 15%)`) on sequential video clips (stride 3), placing adjacent frames ($t$ and $t+3$, separated by only 300 ms) across training and validation sets. The models performed trivial trajectory interpolation rather than held-out generalization across the session changepoint.
- **Pseudo-Latent Bypassing:** Extraction scripts bypassed genuine 3D VAEs, replacing latent encoding with bilinear spatial downsampling and zero-padding.
- **Current Action:** All sub-11° numbers have been retracted. The codebase is currently being standardized under `BaseVAMExtractor` and strict Protocol 1.0; clean re-benchmarking is pending.

---

## 2. World Model Architectural Taxonomy

```
+----------------------------------------------------------------------------------------------------+
|                                      VIDEO-VAM MODEL TAXONOMY                                      |
+----------------------------------------------------------------------------------------------------+
| Model              | Parameters | Transformer Type        | Hidden Dim | Heads | Layers | Latent Dim  |
+--------------------+------------+-------------------------+------------+-------+--------+-------------+
| Cosmos-Predict2-2B | 2.0B       | 3D Spatiotemporal DiT   | 2048       | 16    | 28     | 16x(T=2/16) |
| Cosmos 3 Edge      | 3.37B      | Dual-Pathway MoT        | 2048       | 16    | 28     | 600 visual  |
| Cosmos 7B          | 7.0B       | 3D Spatiotemporal DiT   | 4096       | 32    | 28     | 16x(T=4)    |
| FLUX.2 [klein]     | 9.0B       | Hybrid Double/Single DiT| 6144       | 48    | 8+48   | 128x(8x8)   |
| Cosmos 14B         | 14.25B     | Streaming 3D DiT (FP8)  | 5120       | 40    | 36     | 16x(T=4)    |
+--------------------+------------+-------------------------+------------+-------+--------+-------------+
```

### 2.1 Cosmos-Predict2-2B

- **Architecture:** 28 DiT blocks operating over causal continuous 3D video latents.
- **Latent Compression:** Causal continuous 3D VAE with $8\times 8$ spatial downsampling and **$4\times$ temporal downsampling**.
- **Operating Regimes:**
  - $T=16$ latent frames (offline gold standard, 4,800 tokens at `pool2`, ~1,200 ms latency).
  - $T=2$ latent frames (fast observed-only prefix, 2,400 tokens unpooled, ~204 ms latency).
- **Connector:** Layer 20 intermediate representations ($D=2048$).

### 2.2 Cosmos 3 Edge: Dual-Pathway Mixture-of-Transformers (MoT)

- **Architecture:** 3.37B measured dense parameters (2048 hidden dim, 16 attention heads, 28 layers).
- **Dual-Pathway Mechanism:**
  - Native video path: und_seq contains the text prefix, with causal attention.
  - gen_seq contains both clean observed visual tokens and noisy future tokens, with full attention.
  - The historical custom extractor incorrectly placed VAE visual tokens in und_seq; corrected extraction uses gen_seq and selects clean observed tokens.
- **Implementation Insight:** Fine-tuning Video-LoRA on `gen_seq` with flow-matching loss while extracting features from un-noised `und_seq` creates a representation mismatch. Patchification must strictly follow native `(p, p, C)` spatial packing.

### 2.3 Cosmos-1.0-Diffusion-7B & 14B

- **7B Model:** 7.0B parameter DiT with 28 blocks, 32 attention heads, 4096 hidden dim. Uses 3D mRoPE with `rope_scale=(2.0, 1.0, 1.0)`.
- **14B Model:** 14.25B parameter DiT with 36 blocks, 40 attention heads, 5120 hidden dim.
- **Block Streaming & FP8 Engine:** Linear layer weights quantized on-the-fly to `torch.float8_e4m3fn`. Blocks stream on-demand between host CPU and CUDA memory, with an earlier reported **< 2.7 GB** extraction measurement; full native VAE/training peak memory has not been reprofiled.

### 2.4 FLUX.2 [klein] (9B)

- **Architecture:** 8 multimodal double-stream blocks and 48 single-stream blocks.
- **Conditioning:** Multi-reference observations ($t-2, t-1, t$) with 4D Coordinate RoPE $(T, H, W, L)$.

---

## 3. Historical downstream policy results (not revalidated in this repair)

Evaluated on held-out Episodes 32–39 (88 fixed validation anchors). Actions are scored across the 30-step chunk (offsets $0 \dots 29$ at 10 Hz) using the standard **mixed-unit aggregate RMSE** (5 joints in degrees + gripper in range $[0, 100]$):

| Model Backbone                             | Feature Extraction Setup    | Full-30 Mixed RMSE | Immediate (H1, $t$) | First-5 ($t..t+4$) | Status / Integrity                             |
| :----------------------------------------- | :-------------------------- | :----------------- | :------------------ | :----------------- | :--------------------------------------------- |
| **Cosmos-Predict2-2B (Video-LoRA)**        | $T=16$, Layer 20, pool2     | **13.06**          | 4.76                | 6.19               | Historical reported result (Offline Best)      |
| **Cosmos-Predict2-2B (Teacher Ref)**       | $T=16$ `cond_frames`        | **13.08**          | 4.97                | 6.40               | Empirical Reference Point                      |
| **Cosmos-Predict2-2B (Direct Distilled)**  | $T=2$ Student LoRA          | **13.15**          | 4.58                | 6.04               | Historical reported result (89.4% gap closure) |
| **Cosmos-Predict2-2B (Frozen Pretrained)** | $T=16$, Layer 20, pool2     | **13.65**          | —                   | —                  | Historical reported result                     |
| **Cosmos-Predict2-2B (Undistilled Base)**  | $T=2$, Layer 20 unpooled    | **13.74**          | 4.92                | 6.34               | Historical reported result                     |
| **LTX-Video-2.5 22B**                      | Layer 34 unpooled           | **13.84**          | 4.55                | 6.24               | Historical reported result                     |
| **Cosmos 3 Edge (Pure 600 Visual Tokens)** | Layer 20 `und_seq` (600t)   | **14.26**          | 4.67                | 6.34               | Historical reported result (Zero-Shot Base)    |
| **SmolVLA 450M Baseline**                  | Pretrained SigLIP vision    | **14.83**          | —                   | —                  | Pretrained VLA Reference                       |
| _Cosmos 7B / 14B / FLUX.2 Historical_      | Random Frame Split / No VAE | _INVALID_          | —                   | —                  | **RETRACTED (Data Leakage)**                   |

---

## 4. Key Scientific Insights

1. **Direct Manifold Distillation Closes the Fast-Inference Gap:**
   Distilling the frozen $T=16$ Layer-20 representation directly into a $T=2$ student LoRA achieves **13.15**, capturing **89.4%** of the gap between the fast baseline (13.74) and the teacher reference (13.08).
2. **Representation Gains are Incremental (0.6°–0.8°):**
   Comparing frozen features (13.65) to adapted Video-LoRA (13.06) demonstrates an improvement of ~0.6°–0.8° on this benchmark. This represents a valuable domain adaptation gain, but not an error halving.
3. **Dataset Recalibration Changepoint:**
   Episodes 0–23 and Episodes 24–39 exhibit a major kinematic mean shift (+32.5° on Joint 2, -44.3° on Joint 3; the separate train-versus-validation differences were +19.96° and -29.47°). The validation split (episodes 32–39) is entirely in the late regime, testing generalization across this shift.
4. **Current Status:**
   Correctness fixes across all extractors and training pipelines are implemented and in review. Formal re-benchmarking reruns are pending.
