# AGENTS.md: Developer & Agent Guide for `lerobot-video-vam`

This document serves as the **definitive instruction manual, architectural contract, and evaluation guide** for any AI coding agent or engineer working in this repository (`/home/anton/lerobot-video-vam`). All conventions, contracts, and evaluation protocols defined here are strictly binding.

---

## 1. Project Mission & Core Overview

`lerobot-video-vam` couples generative video diffusion and flow-matching foundation models (Diffusion Transformers / DiTs) with robotics action decoders (specifically SmolVLA / SmolExpert action heads) for physical robot manipulation.

### Supported World Model Backbones

- **NVIDIA Cosmos-Predict2-2B** (Bidirectional DiT with causal video VAE; 4x temporal downsampling)
- **NVIDIA Cosmos 3 Edge** (3.37B measured dense parameters, 2048 hidden dim, 16 heads, 28 layers, Dual-Pathway MoT)
- **NVIDIA Cosmos-1.0-Diffusion-7B** (Diffusers 3D DiT)
- **NVIDIA Cosmos-1.0-Diffusion-14B** (Diffusers DiT with FP8 linear weights & CPU-CUDA block streaming)
- **Lightricks LTX-Video-2.5-22B** (Distilled DiT + causal 3D VAE)
- **Black Forest Labs FLUX.2 [klein] 9B** (Multi-reference hybrid double/single-stream DiT)

### Primary Benchmark

- **Standard Dataset**: `hubnemo/cube_out_of_box_dataset` (SO-101 robotic arm, 6-DoF, 40 episodes, 6,536 frames at 10 FPS, revision `243370c3c08bcbd860133c4a0d658ea7c1d2e77e`).
- **Scaling Dataset**: `Orellius/cube_out_of_box_v2` (100 episodes, 12,163 frames).

---

## 2. Tech Stack & Environment Commands

- **Environment**: Python 3.12+, PyTorch 2.x, CUDA (BF16 / FP8 support).
- **Packaging**: Managed strictly with [`uv`](https://github.com/astral-sh/uv). Always prefix commands with `uv run`.

```bash
# Environment sync
uv sync --locked --extra test --extra dev   # Base + testing + dev tools
uv sync --locked --extra all                # Full dependency suite

# Testing
uv run pytest tests/policies/test_vam_base_abstractions.py -svv   # Base abstraction test suite
uv run pytest tests/policies/test_cosmos3_features.py -svv        # Cosmos 3 feature test suite
uv run pytest tests/policies/test_cosmos3_lora.py -svv            # Cosmos 3 LoRA test suite
uv run pytest tests -svv --maxfail=10                              # Unit test suite

# Formatting and Linting
uv run pre-commit run --all-files
```

---

## 3. System Architecture & Base Abstractions (`src/lerobot/policies/vam/base/`)

All Video Action Model (VAM) components must inherit from and conform to the standardized abstractions in `src/lerobot/policies/vam/base/`.

### 3.1 `BaseVAMExtractor` (`src/lerobot/policies/vam/base/extractor.py`)

#### Core Contract Guarantees:

1. **True VAE Latent Encoding**:
   Subclasses must implement `encode_latents(rgb_frames: Tensor) -> Tensor`. RGB inputs `[B, 3, T, H, W]` must pass through the backbone's true VAE (e.g. `AutoencoderKLCosmos` or LTX causal VAE).
   - **STRICT PROHIBITION**: Heuristic spatial downsampling or bilinear mock padding (e.g. `F.interpolate(..., size=(32,32))` zero-padded to 16 channels) is **strictly forbidden**. It bypasses the latent representation space and raises `VAEContractViolationError`.
   - **Architectural Reality**: Base abstractions alone do **not** magically ensure correctness; each concrete child extractor must correctly invoke its native VAE, apply per-channel latent normalization, and respect posterior modes.
2. **Intermediate Block Tapping**:
   Subclasses must implement `forward_transformer_blocks(latents, text_conditioning, timestep) -> dict[int, Tensor]` to return tapped hidden representations keyed by layer index.
3. **Standardized 3D Spatiotemporal Grid & Pooling**:
   `BaseVAMExtractor.extract()` automatically executes:
   - Latent encoding via true VAE.
   - Transformer forward pass through targeted layers.
   - Spatiotemporal 3D grid alignment via `compute_grid_shape(latents)`.
   - Spatial token pooling via `pool_spatial_tokens(flat_tokens, grid_shape, factor)`.
   - 3D grid coordinate calculation `compute_3d_grid_coordinates(grid_shape, batch_size)`.
4. **Unified Output Container**:
   Returns `VAMExtractionOutput` containing `features`, `features_by_layer`, `grid_coords`, `grid_shape`, and `provenance`.

---

### 3.2 `BaseLoRALinear` & Video-LoRA (`src/lerobot/policies/vam/base/lora.py`)

Unified parameter-efficient fine-tuning (PEFT) framework:

- Formulated as $y = W_{\text{base}} x + \frac{\alpha}{r} (B \cdot A) x$ with $B=0$ at initialization.
- Fully compatible with FP32, BF16, and FP8 (`torch.float8_e4m3fn`) linear base weights.

---

### 3.3 Protocol Split Guard (`src/lerobot/policies/vam/base/split_guard.py`)

- Verifies that train and validation episode sets are **strictly disjoint**.
- Validates that manifests comply with episode boundaries.
- Blocks execution if in-memory frame-level random splitting is detected.

---

## 4. Strict Protocol 1.0 Evaluation Rules

### 4.1 Prohibition of Frame-Level Random Splits

> **CRITICAL RULE**: **NEVER** use `torch.utils.data.random_split` or frame-level random splitting across robotic trajectory datasets.

In demonstration datasets, extraction at stride 3 produces temporally adjacent observation windows (e.g. frame $t$ and frame $t+3$, separated by only 300 ms). Splitting frames randomly between training and validation allows trivial trajectory memorization rather than testing out-of-distribution physical generalization. Historical runs that used random frame splits reported artificial numbers (9.46°–10.99°) and are permanently retracted.

### 4.2 Canonical Protocol 1.0 Specification

- **Training Set**: Strictly **Episodes 0 to 31** (32 demonstrations, ~1,572 extraction windows at stride 3).
- **Validation Set**: Strictly **Episodes 32 to 39** (8 held-out demonstrations, exactly 88 fixed validation anchors at stride 20).
- **Dataset Shift**: Episodes 32–39 exhibit an operator/setup changepoint (+19.96° on Joint 2, -29.47° on Joint 3). True generalizable models must bridge this shift.

---

## 5. Standard CLI Conventions

### 5.1 Standard Extraction: `build_vam_feature_cache.py`

```bash
uv run python scripts/video_vam/build_vam_feature_cache.py \
    --backbone cosmos14b \
    --checkpoint-path /home/anton/.cache/video-vam/cosmos-14b \
    --dataset-repo-id hubnemo/cube_out_of_box_dataset \
    --train-episodes 0-31 \
    --val-episodes 32-39 \
    --train-stride 3 \
    --val-stride 20 \
    --pool-spatial 2 \
    --output-dir /home/anton/.cache/video-vam/features/cosmos14b_protocol1
```

_(Note: Always use `--dataset-repo-id` as the canonical CLI argument)._

### 5.2 Standard Training: `train_smolexpert.py`

```bash
uv run python scripts/video_vam/train_smolexpert.py \
    --backbone cosmos14b \
    --train-manifest /home/anton/.cache/video-vam/features/cosmos14b_protocol1/train/manifest.json \
    --val-manifest /home/anton/.cache/video-vam/features/cosmos14b_protocol1/val/manifest.json \
    --context-transform auto \
    --batch-size 8 \
    --grad-accum-steps 1 \
    --lr 1e-4 \
    --lr-scheduler cosine \
    --warmup-steps 1000 \
    --patience 20 \
    --val-every 1000 \
    --max-steps 50000 \
    --output-dir /home/anton/.cache/video-vam/runs/smolexpert-cosmos14b
```

### 5.3 Action Masking (`action_is_pad`) & Normalization

1. **Normalizer Derivation**: `SmolVLANormalizer` must be computed strictly from unpadded actions in training episodes (0–31). Validation episodes must never contaminate normalizer statistics.
2. **Loss & Metric Masking**: Flow-matching loss and evaluation RMSE must strictly mask padded tokens (`valid = ~paddings`). Padded tokens contribute zero error and zero denominator weight.

---

## 6. Evaluation Rigor & Metric Reporting

### 6.1 Standard Metrics

All action errors are reported in the benchmark's **mixed-unit aggregate convention**:

- Dimensions 0–4 (Arm Joints 1–5): Physical degrees ($^\circ$).
- Dimension 5 (Gripper): Stored percentage / position ($[0, 100]$).

| Metric                 | Definition                                                                         | Purpose                                 |
| :--------------------- | :--------------------------------------------------------------------------------- | :-------------------------------------- |
| **Full-30 Mixed RMSE** | Global RMSE across all valid scalar positions over 30 steps (offsets $0 \dots 29$) | Primary rank metric                     |
| **H1 RMSE**            | RMSE at immediate step offset 0 ($t$)                                              | Measures single-step reactive precision |
| **First-5 RMSE**       | Mean RMSE across offsets $0 \dots 4$ ($t \dots t+4$, first 500 ms)                 | Measures trajectory smoothness          |
| **Per-Joint RMSE**     | Individual RMSE values for each of the 6 robot DoFs                                | Identifies specific joint regressions   |

### 6.2 Evaluation Invariants

- **Per-Sample Seed Batch Invariance**: Each sample receives a deterministic seed derived via stable hash `int(hashlib.sha256(f"{evaluation_seed}:{sample_id}".encode()).hexdigest()[:8], 16) % (2**31 - 1)`. Output is invariant to batch size.
- **Global Error Summation**: Accumulate squared errors globally across the entire evaluation set before taking the square root. **Never average per-batch RMSEs**.
- **Optimizer Step Alignment**: Learning rate warmup and early stopping count actual optimizer steps when gradient accumulation is used.
- **Fail-Closed Validation**: Reject manifests with mismatched dataset revision hashes or overlapping episode splits.
- **Actual Checkpoint Resume**: Resume operations must restore full optimizer, scheduler, and step states.

---

## 7. Core Documentation References

- **`docs/video_vam_roadmap.md`**: Strategic priorities, research bets, and execution roadmap.
- **`docs/video_vam_correctness_audit.md`**: Authoritative source of truth for bugs, data leakage, VAE bypasses, and validation integrity.
- **`docs/video_vam_action_rmse_protocol.md`**: Frozen Protocol 1.0 metrics, horizons, and evaluation equations.
- **`docs/video_vam_cube_out_of_box_leaderboard.md`**: Append-only results ledger with explicit retraction annotations.
- **`docs/video_vam_technical_report.md`**: Deep-dive technical synthesis of world models for physical robot control.
- **`docs/video_vam_research_diary.md`**: Historical research log.

---

## 8. Rules of Engagement for Coding Agents

1. **Do Not Reintroduce Pseudo-Latents or Frame Splits**: Never use `F.interpolate` mocks or `torch.utils.data.random_split`.
2. **Preserve Historical Integrity**: Never delete historical ledger entries; annotate invalid runs with clear retraction notices.
3. **Verify via Tests**: Run `uv run pytest tests/policies/test_vam_base_abstractions.py` and component tests before declaring edits complete.
4. **No Unverified Claims**: Do not invent test pass counts or benchmark numbers.
