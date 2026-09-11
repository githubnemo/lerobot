# Video VAM Action-RMSE Comparison Protocol

**Split/metric contract:** Protocol 1.0 (frozen historical anchors and aggregation)
**Unified evaluator sampling revision:** 1.1 (stable identity-hashed seeds; re-score checkpoints before comparisons)
**Status:** Canonical Source of Truth for Physical Action Metrics and Evaluation Invariants
**Repository:** `lerobot-video-vam`
**Purpose:** Standardized, Native-Systems Physical-Action Comparison Across Foundation Models

---

## 1. Dataset & Split Specifications

- **Dataset Repository:** `hubnemo/cube_out_of_box_dataset`
- **Dataset Revision:** Exact immutable revision `243370c3c08bcbd860133c4a0d658ea7c1d2e77e`
- **Dataset Composition:** 6,536 total frames across 40 continuous episodes recorded at **10 FPS** on the SO-101 6-DoF robot arm.
- **Training Episodes:** Strictly Episodes `0..31` (32 demonstration trajectories, ~1,572 extraction windows at stride 3).
- **Validation Episodes:** Strictly Episodes `32..39` (8 held-out demonstration trajectories, exactly 88 fixed validation anchors at stride 20).
- **Split Guard:** Train and validation episode sets must be **strictly disjoint**. In-memory frame-level random splitting is **strictly forbidden**.

---

## 2. Action Targets, Units, and Masking Rules

### 2.1 Action Space & Mixed Units

The SO-101 action vector contains 6 dimensions:

- **Dimensions 0–4 (Joints 1–5):** Shoulder Pan, Shoulder Lift, Elbow Flex, Wrist Flex, Wrist Roll recorded in **physical degrees ($^\circ$)**.
- **Dimension 5 (Joint 6 / Gripper):** Gripper motor position recorded in range **$[0, 100]$** (percentage open).

> **Mixed-Unit Notice:** The aggregate scalar RMSE reports a weighted combination of 5 rotational degree coordinates and 1 gripper percentage coordinate. It is a standard comparison convention across this benchmark, **not pure physical degrees**. Per-joint RMSEs must also be recorded to track individual joint errors.

### 2.2 Prediction Horizons

- **Full-30 Horizon:** Evaluates all 30 predicted timesteps across offsets $0 \dots 29$ ($t \dots t+29$ at 10 Hz, representing a 3.0-second chunk).
- **Horizon-1 (H1):** Evaluates the immediate next timestep at offset 0 ($t$), measuring single-step reaction accuracy.
- **First-5 Horizon:** Evaluates the mean error across the first 5 timesteps (offsets $0 \dots 4$, corresponding to $t \dots t+4$, the initial 500 ms executed segment).

### 2.3 Action Masking (`action_is_pad`)

- Demonstrations have variable lengths. Action chunks that extend beyond the episode termination are padded.
- **Masking Mandate:** The boolean tensor `action_is_pad` must mask padded tokens. Padded action values must contribute **zero error** to the numerator and **zero weight** to the denominator.
- Normalizer statistics (`SmolVLANormalizer`) must be derived **strictly from training episodes** with padded actions excluded. Validation samples must never contaminate normalizer statistics.

---

## 3. Deterministic Sampling & Aggregation Mathematics

### 3.1 Sampling revisions

Historical Protocol 1.0 uses its recorded per-anchor probe seeds (or evaluation seed plus ordered anchor position in the standalone evaluator). Keep those artifacts unchanged.

The repaired unified trainer uses sampling revision 1.1: deterministic SHA-256-derived seeds from dataset identity, sample ID and evaluation seed; see evaluate_validation in scripts/video_vam/train_smolexpert.py for the exact encoding. Noise, flow epsilon and flow time are independent of training RNG and batch/order. GPU floating-point kernels may still introduce small numerical differences across batch sizes. Re-evaluate all comparison checkpoints with the same evaluator revision; do not silently compare old and new sampling numbers.

### 3.2 Global Error Aggregation

- **Global Summation:** Squared errors must be accumulated globally across the entire evaluation set before taking the square root:
  $$\text{RMSE}_{\text{global}} = \sqrt{\frac{\sum_{i=1}^{N} \sum_{h=0}^{H-1} \sum_{d=0}^{D-1} (\hat{a}_{i,h,d} - a_{i,h,d})^2 \cdot (1 - \text{pad}_{i,h})}{\sum_{i=1}^{N} \sum_{h=0}^{H-1} \sum_{d=0}^{D-1} (1 - \text{pad}_{i,h})}}$$
- **Prohibition:** **NEVER** calculate per-batch RMSEs and average them. Batch-mean averaging produces mathematically flawed metric distortions.

---

## 4. Native Policy Configurations

1. **SmolExpert (VAM Decoders):**
   - Context input: visual features extracted from designated backbone layers.
   - Denormalization: inverted using train-only mean/std normalization (SmolVLANormalizer).
   - Flow-matching decoder runs 10 deterministic sampling Euler steps.

2. **Legacy World2Action:** train-window-only min/max normalization, excluding padded actions. This is not the SmolExpert normalizer.

3. **SmolVLA Reference:**
   - Standard LeRobot checkpoint.
   - Native chunk size: 50 steps; only indices `0..29` (offsets $0 \dots 29$) are evaluated.
   - Denormalization: inverted using checkpoint-native statistics.

4. **FastWAM:** native 32-action chunk, score first 30; checkpoint-native preprocessing and cached UMT5 conditioning.

5. **Baselines:**
   - `state_repeat`: repeats the current 6-DoF joint state proprioception across all 30 steps.
   - `mean_action`: predicts the static train-split mean action vector across all 30 steps.

---

## 5. Fail-Closed Validation & Protocol Invariants

Evaluator scripts must enforce fail-closed checks before computing metrics:

- Verify that `val_manifest.json` conforms to `MANIFEST_SCHEMA_VERSION = 1`.
- Verify that every validation sample belongs to Episodes 32–39.
- Verify that the dataset commit matches `243370c3c08bcbd860133c4a0d658ea7c1d2e77e`.
- Reject caches with un-normalized latent representations or missing provenance metadata.

---

## 6. Protocol Versioning Policy

- **Protocol 1.0 Freeze:** The definitions and validation anchors specified here are permanently frozen.
- **Future Revisions:** Any changes to sampling seeds, validation anchors, or horizon masks must increment the protocol version (e.g. **Protocol 1.1**) and be explicitly labeled. Never mix or silently compare numbers across different protocol versions.
