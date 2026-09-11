# Video-VAM Experiments Overview & Rollout Benchmarks

**Status:** Technical Experiments Synthesis
**Repository:** `lerobot-video-vam` (`/home/anton/lerobot-video-vam` on `abakus`)
**Dataset:** `hubnemo/cube_out_of_box_dataset` (SO-101 Arm, 6,536 Frames, 40 Episodes at 10 FPS, Revision `243370c3c08bcbd860133c4a0d658ea7c1d2e77e`)

---

## 1. NVIDIA Cosmos-1.0-Diffusion-14B Full-Length Rollout Benchmark

### Context & Implementation

Previous video rollouts were capped at 17 frames (1.7 seconds), insufficient to cover full robot manipulation trajectories. For **Cosmos-1.0-Diffusion-14B**, rollouts have been executed across a full **161-frame sequence (16.1 seconds at 10 fps)** covering frames 4820 through 4980 (held-out validation Episode 32):

- Arm descent into the container
- Jaw closure and cube grasping
- Vertical lift and extraction
- Return motion to home configuration

### Technical Architecture

- **Model:** NVIDIA Cosmos-1.0-Diffusion-14B (14.25 Billion parameters, 36 blocks, 40 attention heads, 5120 hidden dimension).
- **Memory Optimization:** FP8 linear layer base weights (`torch.float8_e4m3fn`) with CPU-to-CUDA on-demand block streaming (< 2.7 GB peak VRAM on a single 24GB RTX 4090).
- **3D VAE:** `AutoencoderKLCosmos` (causal 3D convolutional VAE, 8x spatial downsampling, 8x temporal compression).
- **Color Pipeline:** Normalized uint8 $[0, 255] \to [-1, 1] \to [-1, 1] \to [0, 255]$ uint8.

### Quantitative Rollout Metrics (Episode 32, Frames 4820–4980)

| Model Variant        | Frames | Duration | Mean PSNR (dB) | Mean SSIM  | Mean MSE | Peak VRAM |
| :------------------- | :----: | :------: | :------------: | :--------: | :------: | :-------: |
| **Cosmos 14B Base**  |  161   |  16.1 s  |  **18.28 dB**  | **0.7803** | 1339.95  | 16.11 GB  |
| **Cosmos 14B QLoRA** |  161   |  16.1 s  |  **18.28 dB**  | **0.7803** | 1341.91  | 16.91 GB  |

---

## 2. Multi-Model Rollout Comparison on Episode 32

| Model                        |    Sequence Length     | Parameters |  Quantization   | Mean PSNR (dB) | Mean SSIM  | Status                    |
| :--------------------------- | :--------------------: | :--------: | :-------------: | :------------: | :--------: | :------------------------ |
| **Cosmos-1.0-Diffusion-14B** | **161 frames (16.1s)** | **14.25B** | **FP8 + QLoRA** |  **18.28 dB**  | **0.7803** | Full Manipulation Rollout |
| Cosmos-1.0-Diffusion-7B      |    17 frames (1.7s)    |    7.0B    |   BF16 + LoRA   |   28.21 dB\*   |  0.8772\*  | Short prefix window       |
| FLUX.2 [klein] 9B            |    17 frames (1.7s)    |    9.0B    |   BF16 + LoRA   |    12.56 dB    |   0.4907   | Short prefix window       |

_\*Note: Cosmos 7B metrics were evaluated on a 17-frame window prior to major manipulator displacement._

---

## 3. Methodological Audit of Downstream Policy Benchmarks

### Historical Retraction Notice:

- Earlier downstream policy numbers for Cosmos 7B (9.46°), Cosmos 14B (9.97°), and FLUX.2 (10.33°) used an in-memory frame-level `random_split(85%, 15%)` and bilinear pseudo-latents, which introduced trajectory leakage.
- These results are **retracted** per `docs/video_vam_correctness_audit.md`.
- Current work focuses on rigorous Protocol 1.0 re-benchmarking using genuine VAE latents and disjoint episode sets (train: episodes 0–31, val: episodes 32–39). No jobs are currently scheduled; code implementation is under review.

## 4. Phase 5: Expanded 100-Episode Demonstration Scaling

Evaluates the scaling impact of increasing demonstration data from 40 to 100 episodes using `Orellius/cube_out_of_box_v2` (100 episodes, 12,163 frames).

### Dual-Evaluation Protocol

- **Train Split**: 82 episodes (0–31 and 40–89; stride 3, 3,065 samples)
- **Eval-Set 1 (Historical Benchmark)**: 8 episodes (32–39; stride 20, 88 anchors). Directly compares 40 vs 100 demonstration scaling under strict Protocol 1.0.
- **Eval-Set 2 (New Benchmark)**: 10 episodes (90–99; stride 20, 51 anchors). Validates generalization on the newly collected demonstration distribution.

### Models & Convergence Configuration

1. **Cosmos 14B Base**:
   - Layers 18 & 30 concatenation (10,240-dim), FP8 linear quantization + block streaming (<2.7 GB VRAM).
   - SmolExpert convergence budget: `min_steps=20000`, `max_steps=60000`, `patience=25`.
2. **FLUX.2 [klein] Base**:
   - Multi-reference junction tap (256 tokens, 6,144-dim).
   - SmolExpert convergence budget: `min_steps=15000`, `max_steps=50000`, `patience=25`.

### Results Matrix

| Model                         | Eval-Set 1 RMSE (Hist) | Eval-Set 2 RMSE (New) | Status    |
| :---------------------------- | :--------------------- | :-------------------- | :-------- |
| **Cosmos 14B Base**           | **14.02°**             | **19.41°**            | COMPLETED |
| **FLUX.2 [klein] Base**       | **15.75°**             | **20.45°**            | COMPLETED |
| **FLUX.2 [klein] Video-LoRA** | **15.89°**             | **21.02°**            | COMPLETED |
| **Cosmos 14B Video-LoRA**     | —                      | —                     | QUEUED    |
