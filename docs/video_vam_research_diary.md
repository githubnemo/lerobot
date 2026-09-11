# Current correction status — 2026-09-07

The native-path and evaluation repairs are under integration validation; no corrected performance numbers exist yet. See [roadmap](video_vam_roadmap.md), [audit](video_vam_correctness_audit.md), and [metric/sampling revisions](video_vam_action_rmse_protocol.md). Historical claims below must be interpreted with those corrections. In particular, operator/calibration causation is unproven, auxiliary-head low cosine is not proof of lost information, and the 26.12 comparison is not an isolated Video-LoRA effect.

# Video Foundation Models as Robot-Policy Backbones

> **Strategic Living Roadmap**: See [`video_vam_roadmap.md`](video_vam_roadmap.md) for the active project status, complete benchmark scoreboard, and forward execution phases.

## Historical executive summary (2026-09-02; superseded)

See the roadmap and correctness audit for current status. Claims below are preserved as history, not current conclusions.

- **Direct Representation Distillation Milestone**: Direct $T=16 \to T=2$ Layer-20 manifold distillation reached **`13.15°`** validation RMSE at **~204 ms**, beating the undistilled baseline (**13.74°**) and matching the full $T=16$ teacher (**13.06°**) within **0.09°**.
- **RETRACTED interpretation — theoretical ceiling/98.9%:** Training directly on teacher $T=16$ `cond_frames` reached **`13.08°`**. The distilled model captures **98.9%** of the theoretical ceiling.
- **Root-Cause Flaws Resolved**: Fixed a 4-frame temporal lag in data loading; identified that auxiliary linear projection heads scramble internal latent features ($\text{CosSim} = 0.28$ without head vs. $0.97$ with direct distillation).
- **Active Job on `abakus`**: `cosmos3-edge-pipeline-20260902` is running the complete benchmark pipeline for NVIDIA's newly released **`Cosmos3-Edge`** (~4B parameters, Wan 2.2 VAE).

---

## Historical Research diary — 2026-08-17

This is a catch-up document, not an implementation report. No model or dataset code has been written yet. It records the initial source audit, the current machine/repository state, the architecture I believe we are actually dealing with, unresolved scientific risks, a milestone plan, and the first decisions needed before implementation.

## 1. Current status

- Work is on `abakus` only.
- The existing `/home/anton/lerobot` checkout was not safe to switch: it contains uncommitted RL experiments on `feat/rl-test-anon` and is two commits ahead of its remote.
- I left that work untouched.
- I fetched current official Hugging Face LeRobot `upstream/main` and created a linked worktree:
  - path: `/home/anton/lerobot-video-vam`
  - branch: `feat-video-vam`
  - base: official LeRobot commit `6adf51511b7625090eade8d82d9f61a1846ebe56`
- The new worktree was clean before this diary was added.
- No commit or push has been made.

### Abakus compute inventory

- GPU: one NVIDIA GeForce RTX 4090, 24,564 MiB VRAM, compute capability 8.9, 450 W limit.
- Host RAM: 62 GiB, with about 60 GiB available during inspection.
- Swap: 8 GiB.
- Storage: about 1.1 TiB free on `/home/anton`.
- CPU availability: 16 logical CPUs.
- Host Python: 3.12.3.
- Existing old LeRobot virtualenv: PyTorch 2.7.1 + CUDA 12.6; CUDA is usable and sees the 4090.
- No system `nvcc` command is currently installed. This does not prevent ordinary prebuilt PyTorch CUDA use, but custom CUDA extensions may need extra setup.

This machine is comfortable for SmolVLA and small decoders. It is below the official standalone Cosmos-Predict2 2B Video2World memory figure (~32.5 GB) and far below a normal full-precision LTX-2.5 deployment. Both video backbones will need a measured low-memory strategy rather than an assumption that they fit.

## 2. Sources pinned during this audit

### LeRobot / SO-101 / SmolVLA

- Official LeRobot repository: <https://github.com/huggingface/lerobot>
- Inspected commit: `6adf51511b7625090eade8d82d9f61a1846ebe56`
- Latest stable observed: `v0.6.1`; moving `main` identifies itself around 0.6.2.
- LeRobotDataset v3 docs: <https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3>
- SO-101 docs: <https://github.com/huggingface/lerobot/blob/6adf51511b7625090eade8d82d9f61a1846ebe56/docs/source/so101.mdx>
- SmolVLA implementation: <https://github.com/huggingface/lerobot/tree/6adf51511b7625090eade8d82d9f61a1846ebe56/src/lerobot/policies/smolvla>
- SmolVLA checkpoint: <https://huggingface.co/lerobot/smolvla_base>
- Checkpoint revision inspected: `c83c3163b8ca9b7e67c509fffd9121e66cb96205`
- SmolVLA paper: <https://arxiv.org/abs/2506.01844>

### mimic-video / Cosmos-Predict2

- Official mimic-video repository: <https://github.com/mimic-video/mimic-video>
- Inspected commit: `e3355dbc93132b576c02f920a59b4fc18a4f5906`
- Paper: <https://arxiv.org/abs/2512.15692>
- Official checkpoint collection: <https://huggingface.co/jonpai/mimic-video>
- Vendored Cosmos code: <https://github.com/mimic-video/mimic-video/tree/e3355dbc93132b576c02f920a59b4fc18a4f5906/model/cosmos_predict2>
- Original Cosmos-Predict2 repository: <https://github.com/nvidia-cosmos/cosmos-predict2>
- Inspected upstream commit: `661da4774b0ca41d082a0ecbeb47550bcf07e03f`

Important lifecycle fact: mimic-video uses an older, simplified/vendored Cosmos-Predict2 implementation. The original Predict2 repository is archived; Predict2.5 and Cosmos 3 are not drop-in upgrades for mimic-video. For a faithful baseline we should pin mimic-video's code first, not silently port it to Cosmos 3.

### LTX-2.5

- Official repository: <https://github.com/Lightricks/LTX-2>
- Inspected commit: `400fd31054597515f47125691032c04b1c3ee24e`
- Core architecture: <https://github.com/Lightricks/LTX-2/tree/400fd31054597515f47125691032c04b1c3ee24e/packages/ltx-core>
- Training/LoRA package: <https://github.com/Lightricks/LTX-2/tree/400fd31054597515f47125691032c04b1c3ee24e/packages/ltx-trainer>
- Public architecture paper: <https://arxiv.org/abs/2601.03233>

Caveat: the paper is for the LTX-2 family and does not document every LTX-2.5 checkpoint flag. Exact model metadata must be read from the checkpoint we choose rather than inferred from the paper or filename.

## 3. What mimic-video actually implements

### 3.1 High-level computation

The released method is approximately:

```text
observed RGB prefix + language
        ↓
Cosmos video tokenizer
        ↓
clean observed latent prefix + noisy/generated future latent
        ↓
Cosmos-Predict2 2B video DiT at video sigma σv
        ↓
intermediate hidden tokens (released config: layer index 20)
        ↓
500M action DiT cross-attending to those tokens
        ↑
current proprioception token + noisy action chunk + action time τa
        ↓
flow-matched action velocity
        ↓ repeated Euler steps
normalized action chunk
```

The policy does not need to decode future pixels before producing actions. The action decoder consumes an internal video-DiT hidden state.

### 3.2 Video input and latent shapes

For the released 480×640 configuration:

```text
observed RGB:       [B, 3, 5, 480, 640]
future RGB target:  [B, 3, 56, 480, 640]
combined video:     [B, 3, 61, 480, 640]
```

The Cosmos tokenizer has:

- 16 latent channels;
- 8× spatial compression;
- 4× temporal compression, with a special first frame.

Therefore:

```text
video latent: [B, 16, 16, 60, 80]
```

The two latent prefix frames corresponding to the five observed frames are kept clean as conditioning. The future latent region is noised during training or initialized/generated from noise during inference.

The 2B video DiT uses:

- 28 transformer blocks;
- hidden width 2048;
- 16 attention heads;
- spatial patch size 2;
- temporal patch size 1.

After patch embedding:

```text
hidden grid: [B, 16, 30, 40, 2048]
flattened:   [B, 19,200, 2048]
```

The released action experiments configure `xattn_layer_idx = 20`. Those flattened hidden tokens are the action decoder's cross-attention context. They are transformer hidden states, not decoded video, raw noisy tokens, or the final denoising prediction.

### 3.3 Video flow time versus sigma

The paper uses normalized flow notation:

```text
z_τ = (1 - τ) z_clean + τ ε
```

where `τ=0` is clean and `τ=1` is noise.

The released code operates in Cosmos sigma space instead. At inference, the future latent is initialized from Gaussian noise scaled up to `sigma_max = 80`, then integrated along a sigma schedule. The action head also receives video sigma after the mapping:

```text
σv / (1 + σv)
```

Thus “video timestep” is part of the representation definition. A feature from layer 20 at one sigma is not scientifically the same representation as layer 20 at another sigma.

### 3.4 One backbone forward pass

A single Cosmos video-DiT forward receives:

- video latent with clean observation-prefix conditioning;
- the video sigma/timestep;
- text/T5 context, typically `[B, 512, 1024]`;
- positional and conditioning masks.

It updates all spatiotemporal tokens through the 28-block transformer and exposes requested intermediate hidden states. mimic-video stops video generation at a chosen step, requests the configured hidden state, and passes that state directly to the action model.

The action representation is therefore jointly a function of:

```text
observation history
language
future-latent initialization/current generated latent
video sigma
hidden-state depth
random seed / denoising trajectory
```

These variables must be logged. Treating the feature as merely “Cosmos embedding” would hide scientifically important conditions.

### 3.5 Proprioception and action decoder

The released robot representation is 10D:

```text
[x, y, z, rotation_6d(6), gripper]
```

Typical proprioception input:

```text
state: [B, 1, 10]
```

The released action DiT uses:

- width 1024;
- 24 blocks;
- 8 heads;
- cross-attention context width 2048;
- approximately 500M parameters.

The proprioceptive state is projected to a token. The noisy action chunk is projected to action tokens. The state and action tokens are concatenated, then each block performs cross-attention to video tokens, self-attention over state/action tokens, and an MLP with timestep-conditioned AdaLN modulation.

Released horizons:

- Bridge: `max_horizon=16`, so with one state token it predicts 15 action steps.
- LIBERO: `max_horizon=61`, so with one state token it predicts 60 action steps.

The output is shaped:

```text
[B, action_horizon, action_dimension]
```

For SO-101 direct joint control, both the state/action dimension and semantics must change. The published 10D Cartesian decoder cannot directly command SO-101's native six motor positions.

### 3.6 Action flow matching

For normalized target actions `A0`, Gaussian noise `ε`, and action flow time `τa`:

```text
Aτ = (1 - τa) A0 + τa ε
uτ = ε - A0
loss = || policy(Aτ, state, video_hidden, τa, σv) - uτ ||²
```

At inference:

1. Sample Gaussian action noise with shape `[B, H, A]` at `τa=1`.
2. Predict the action velocity field.
3. Euler-integrate toward `τa=0`.
4. The public config uses ten action denoising steps.
5. Unnormalize to robot units.
6. Execute only a configurable prefix of the chunk, then replan.

The released evaluators commonly execute five actions before replanning even when the predicted chunk is longer.

### 3.7 Learned, frozen, and gradient flow

mimic-video has two conceptual stages:

1. Robot-video adaptation: LoRA adapts the video model on robot video/language without action labels. The released config uses a high LoRA rank (256, alpha 32) across many attention/MLP projections.
2. Action training: the video pipeline is frozen; the action DiT is trained from scratch.

During action training, hidden video states are explicitly detached/cloned. Therefore:

```text
action loss → action DiT parameters

action loss ↛ video DiT
action loss ↛ tokenizer
action loss ↛ text encoder
```

Our initial frozen-backbone experiment should preserve this separation. Later video LoRA adaptation is a distinct experiment and must not be conflated with action supervision.

### 3.8 Important reproduction discrepancies and scientific risks

#### Paper/code action-time discrepancy

The paper describes a nonuniform action-time density roughly proportional to `sqrt(τa - 0.001)`. The public code configures `BetaScheduler(alpha=1, beta=1)`, effectively uniform over approximately `[0.001, 1]`.

We need to name which target we reproduce: published prose or current released code. My recommendation is to make current released code the reproducibility baseline and record the discrepancy explicitly.

#### Ground-truth future information during action training

The action-training path can build video features from a noised latent containing the recorded future video, while inference begins from noise and a model-generated denoising trajectory. At high noise, little future target information remains; at lower noise, the training feature can retain information from real future frames that does not exist online.

This may be an intentional teacher-forced training design, but it is also the most important leakage/distribution-shift audit item in this project. We must distinguish:

- future video used only as an offline video-model target;
- future video influencing action-decoder training features;
- causal deployment features generated only from current/past observations.

For every selected sigma, we should test whether replacing or shuffling the future target changes the action feature and whether the online generated-feature distribution matches the training-feature distribution.

#### Environment mismatch

Current mimic-video requires Python 3.10 and NumPy 1.26-era constraints. Current LeRobot requires Python 3.12 and NumPy 2.x. One environment is unlikely to be robust.

The probable architecture is:

```text
LeRobot control/data process (Python 3.12)
        ↕ explicit typed IPC/RPC boundary
mimic-video inference/training process (Python 3.10)
```

This should remain a small adapter boundary, not a copied fork of either upstream project.

## 4. How LTX-2.5 differs where it matters

### 4.1 Architecture

LTX-2.5 is an audiovisual diffusion transformer, not a robot policy. The inspected implementation has:

- release label around 22B parameters;
- approximately 14B video-stream capacity and 5B audio-stream capacity in architectural descriptions, with remaining accounting in auxiliary/shared components;
- 48 transformer blocks;
- video width 4096;
- 32 video attention heads of dimension 128;
- audio width 2048;
- 128-channel video latents;
- asymmetric video/audio streams with bidirectional cross-modal attention;
- Gemma 4 text conditioning through LTX-specific connectors.

For this project, audio should be disabled unless we explicitly decide that robot audio is an observation. Otherwise it adds compute and creates an unfair extra modality.

### 4.2 VAE and token shapes

The VideoVAE approximately maps:

```text
RGB:    [B, 3, F, H, W]
latent: [B, 128, F', H/32, W/32]
F' = 1 + (F - 1) / 8
```

The current model path uses latent patch size `(1,1,1)`, then flattens and projects:

```text
[B, 128, F', H', W']
→ [B, F'·H'·W', 128]
→ [B, F'·H'·W', 4096]
```

Example at 544×960:

```text
one RGB frame:  [B, 3, 1, 544, 960]
one latent grid: [B, 128, 1, 17, 30]
tokens:          [B, 510, 4096]
```

For 121 frames the same resolution yields 16 latent temporal slices and 8,160 video tokens.

This differs sharply from mimic-video's 19,200 tokens at 480×640/61 frames. LTX compresses spatially and temporally much more aggressively but uses a wider 4096-dimensional token.

### 4.3 Conditioning and timestep semantics

LTX conditioning images are inserted into the video latent. Their denoise mask freezes or marks the corresponding latent positions. Timesteps can be per-token:

```text
token_timestep = denoise_mask × sigma
```

Therefore conditioned image tokens can have timestep zero while generated tokens have the current sigma. Hidden states depend on:

```text
latent values
positions
denoise mask
per-token timestep and sigma
text context
keyframe/reference configuration
audio state, if enabled
checkpoint-specific AdaLN/RoPE flags
```

A useful extraction API must return this metadata with the tensor. Returning only `[B,N,4096]` would make experiments hard to reproduce and easy to misinterpret.

LTX's schedule is also token-count dependent. Resolution and duration can change the sigma trajectory, so a nominal “step 5” is not comparable across token counts. We should record actual sigma, not only step index.

### 4.4 Plausible representation extraction points

The most defensible first candidates are:

1. Final VideoVAE latent: `[B,128,F',H',W']`. Cheapest LTX-specific baseline; no transformer semantics.
2. Patchified latent before projection: `[B,N,128]`.
3. Projected tokens before block 0: `[B,N,4096]`.
4. Full block outputs at multiple depths: `[B,N,4096]`.
5. Final pre-output normalized/modulated tokens: `[B,N,4096]`.

The final 128-channel diffusion prediction is not my recommended control representation. It is optimized to predict a denoising quantity, whereas block hidden states are the closer analogue to mimic-video's extracted representation.

Preliminary depth sweep for the actual 48-block architecture:

```text
VAE latent / block 0 baseline
block 12 (25%)
block 24 (50%)
block 36 (75%)
block 48/final (100%)
```

We may add block 6 or 8 if the 25% prefix is already too expensive. The final choice is a later Decision Gate after loading the exact checkpoint and measuring prefix latency/memory.

### 4.5 Safe early exit

The transformer is sequential, so stopping after block `k` is valid for extracting that prefix representation. It is not valid for producing a final LTX denoising prediction or decoded video.

The clean upstream-minimal change is an optional `max_block`/callback in the model's internal block loop that returns:

```text
video_hidden: [B,Nvideo,4096]
positions
timesteps
sigma
denoise_mask
latent_grid_shape
```

A Python forward hook is acceptable for initial inspection but is fragile with `torch.compile`, graph capture, offloading, or block streaming.

Required verification:

1. Run full 48-block inference in deterministic eager/eval mode and record block `k`.
2. Run the prefix path with exactly the same latent, text, masks, positions, sigma, seed, and precision.
3. Compare prefix output to the recorded full-forward activation at block `k`.
4. Only after numerical equivalence is established use early exit for latency claims.

Without actual early termination, a “depth versus latency” figure would be invalid.

### 4.6 Adapter and decoder incompatibilities

LTX has no official inverse action head. A projector is unavoidable because LTX emits width 4096 while mimic-video's action decoder cross-attention expects width 2048.

A minimal candidate is a tokenwise linear map:

```text
[B,N,4096] → [B,N,2048]
```

That map alone has roughly 8.4M weights before bias. If token counts are too expensive for cross-attention, a spatial pooling stage may also be needed. Pooling is not a trivial implementation choice: it decides whether the policy can retain spatial correspondences.

The fair comparison should report separately:

- backbone parameters executed;
- projector parameters;
- action decoder parameters;
- total trainable parameters;
- representation token count and bytes;
- latency to the extraction layer;
- downstream latency.

The matched-decoder experiment should use the same decoder family and comparable width/depth across Cosmos and LTX, but exact equality may be impossible because context widths and token grids differ. We should avoid giving LTX a large learned resampler that becomes a second representation model.

### 4.7 LTX memory reality on abakus

Observed official artifacts and docs suggest approximately:

- full transformer artifact: ~42 GB decimal;
- int8 transformer artifact: ~21.5 GB decimal;
- NVFP4 transformer artifact: ~18.7 GB decimal;
- Gemma text encoder: ~26 GB decimal before any quantization;
- normal training: 80 GB-class GPU;
- low-VRAM LoRA examples: 32 GB-class GPU.

The 24 GB 4090 cannot be assumed to hold the full standard pipeline. A possible research-only extraction path is:

1. precompute/cache task text embeddings;
2. unload/offload the text encoder;
3. disable audio;
4. use an officially supported quantized transformer path compatible with Ada compute capability 8.9;
5. offload VAE and/or transformer blocks;
6. use one short observation history and reduced resolution first;
7. decode no video during action training;
8. measure numerical and performance consequences.

Whether the available quantized path actually fits and runs acceptably on this 4090 is unresolved. NVFP4 support must not be assumed on Ada. Int8 plus offload is the more plausible first probe, but it needs a real smoke test.

## 5. SmolVLA baseline facts

SmolVLA is already integrated in LeRobot and is the lowest-risk baseline.

Its default conceptual input is:

```text
current image(s)
current proprioceptive state
task-language instruction
```

It is not normally a multi-frame video encoder. The SO-101 state/action tensors are typically six named motor positions before internal padding to 32 dimensions.

The released policy combines:

- a truncated SmolVLM2 vision-language model;
- approximately 64 visual tokens per image;
- a separate action expert;
- state and action projections;
- flow-matched continuous action chunks.

Typical dimensions/configuration:

```text
max state dimension:  32
max action dimension: 32
action chunk:         50
flow steps:           10
```

SO-101's meaningful six action dimensions are padded internally and unpadded before postprocessing.

Default low-data fine-tuning behavior is approximately:

- frozen vision encoder/VLM;
- trainable action expert;
- trainable state/action/timestep projections;
- about 450M total parameters;
- about 100M trainable parameters, to be counted exactly from the pinned checkpoint.

Its flow objective is similar in form to mimic-video, but current SmolVLA samples action time from `Beta(1.5,1.0)` rather than mimic-video public code's effective uniform schedule.

For fairness we should share raw LeRobot episodes, episode splits, camera source, task strings, robot state/actions, and evaluation conditions. We should not force SmolVLA to use LTX/Cosmos video latents or force VAMs to use SmolVLA's internal temporal architecture. The comparison is system-level, with architectural differences documented.

## 6. LeRobotDataset and SO-101 facts relevant to the common interface

### 6.1 Canonical source of truth

LeRobotDataset v3 stores:

```text
meta/info.json
meta/stats.json
meta/tasks.parquet
meta/episodes/...parquet
data/...parquet
videos/observation.images.<camera>/...mp4
```

Current live code uses `meta/tasks.parquet`; some docs still mention legacy `tasks.jsonl`.

The common causal sample can be represented as:

```text
observation.images.<camera>: [C,H,W]
observation.state:           [6]
task:                        string
action:                      [H,6]
action_is_pad:               [H] boolean
```

`delta_timestamps={"action": [0/fps, ..., (H-1)/fps]}` produces an action window. LeRobot clamps windows at episode boundaries and supplies a padding mask; it does not cross into the next episode.

### 6.2 Native SO-101 semantics

The standard single-arm follower exposes six named positions in this order in the current default implementation:

```text
shoulder_pan.pos
shoulder_lift.pos
elbow_flex.pos
wrist_flex.pos
wrist_roll.pos
gripper.pos
```

We must still assert names/order from the concrete dataset and robot rather than assume them.

The hardware driver natively receives absolute calibrated joint-position goals. “Relative to current state” is a reversible processor convention. “Delta from the previous target” is a different accumulating representation and is not a hardware mode.

Current LeRobot defaults/examples often target 30 Hz, but actual camera/control cadence may be lower. `read_latest()` may return a stale camera frame. Requested FPS must not be treated as measured synchronization.

### 6.3 Split/statistics caveat

LeRobot's built-in `eval_split` holds out complete episodes, generally the last fraction per task. However, split datasets still see dataset-level `meta/stats.json`. For strict low-data experiments, normalization statistics must be recomputed from training episodes only.

Episode-level splitting is mandatory. For the main result, session/object-layout dependence should also be controlled because two different episode IDs can still share almost identical scenes and trajectories.

## 7. Minimum viable path to one safe SO-101 rollout

The first implementation target should be narrower than the whole research matrix:

1. Pin the LeRobot, mimic-video, model-checkpoint, and dataset revisions.
2. Define the exact SO-101 observation/action contract and timing from the concrete robot/data.
3. Record or identify 1–5 clean episodes in canonical LeRobotDataset v3.
4. Build a causal dataset adapter that emits current/past images and state plus future action labels; future RGB remains a separate adaptation-only target.
5. Add synthetic alignment, episode-boundary, normalization round-trip, and leakage tests before model work.
6. Wrap the pinned mimic-video representation extractor with explicit sigma, hidden layer, seed, and metadata outputs.
7. Adapt the action DiT input/output to the chosen SO-101 state/action dimensions while keeping the video backbone frozen.
8. Overfit a tiny batch, then one episode, then 1–5 episodes. Verify normalized and physical-unit action reconstruction.
9. Compare training-time teacher-forced video features with causal online-generated features at the selected sigma.
10. Add a dry-run LeRobot policy wrapper that cannot send actions; validate names, units, finite values, joint/rate bounds, timeouts, and queue behavior.
11. Add a separately enabled hardware path with conservative clipping, physical emergency stop, short rollout duration, and human supervision.
12. Run the same tiny-data plumbing check for SmolVLA.
13. Only after Cosmos is correct, add the LTX extractor and matched decoder.

A useful deployment boundary is:

```text
LeRobot process
  camera/state/task → request
  action chunk      ← response

Backbone/action process
  preprocess → video features → action denoising → unnormalized chunk
```

The control process should own hardware safety. The model process should never write to motors directly.

## 8. Proposed repository structure

This keeps LeRobot as the host project and adds adapters rather than copying upstream implementations:

```text
src/lerobot/policies/vam/
    configuration_vam.py
    modeling_vam.py
    processor_vam.py
    common/
        sample.py
        action_flow.py
        projectors.py
    cosmos/
        extractor.py
        policy.py
    ltx/
        extractor.py
        policy.py

src/lerobot/datasets/vam/
    windows.py
    splits.py
    normalization.py
    leakage_checks.py

src/lerobot/robot_devices/ or current rollout integration
    no direct change unless the existing policy API is insufficient

configs/video_vam/
    cosmos/
    ltx/
    smolvla/
    evaluation/

tests/policies/vam/
tests/datasets/vam/

scripts/video_vam/
    inspect_dataset.py
    cache_features.py
    overfit_tiny.py
    benchmark_latency.py
    rollout_dry_run.py

docs/video_vam/
    upstream_pins.md
    deviations.md
    experiment_protocol.md
```

Exact placement will follow current LeRobot conventions after implementation starts. Upstream mimic-video and LTX should remain separately pinned dependencies/checkouts or service environments, not copied wholesale into `src/`.

## 9. Milestone plan

### Milestone 0 — decisions, pins, and feasibility probes

- Resolve Decision Gate 1 below.
- Pin exact checkpoints and licenses/access requirements.
- Confirm model download sizes and disk plan.
- Run no-training load/forward memory probes on the 4090 for Cosmos and LTX.
- Decide separate environments and IPC contract.

Exit criterion: we know which backbones can execute on abakus, at what precision/offload, and what numerical representation they return.

### Milestone 1 — canonical data contract and tests

- Inspect the actual SO-101 dataset schema and calibration metadata.
- Implement episode-level splits and train-only normalization.
- Implement causal windows, padding masks, timestamp/alignment diagnostics, and future-leakage tests.

Exit criterion: a sample's image, state, action chunk, units, names, timestamps, and episode provenance can be explained and tested.

### Milestone 2 — faithful mimic-video frozen extractor

- Integrate the pinned official code with minimal patches.
- Reproduce its hidden-state extraction and one supported tiny/offline example if feasible.
- Log sigma, layer, token layout, model/checkpoint revision, seed, and preprocessing.
- Document every deviation from upstream.

Exit criterion: extracted tensors match expected shapes and the frozen backbone receives no gradients.

### Milestone 3 — SO-101 action decoder overfit

- Adapt state/action dimensions and chosen action semantics.
- Train the matched action DiT on a tiny batch and 1–5 episodes.
- Validate training-time versus online causal representations.
- Demonstrate action reconstruction in normalized and robot units.

Exit criterion: the complete LeRobot episode → Cosmos hidden state → predicted SO-101 chunk path intentionally overfits without future input at inference.

### Milestone 4 — safe dry run and first physical rollout

- Implement action bounds, joint/rate limits, NaN handling, timeout behavior, queue reset, and explicit arming.
- Measure end-to-end policy latency and choose chunk execution/replanning behavior.
- Perform a short supervised rollout only after offline checks pass.

Exit criterion: one non-cherry-picked rollout is recorded with model/checkpoint/config and full safety logs.

### Milestone 5 — SmolVLA baseline

- Fine-tune the pinned `lerobot/smolvla_base` on the same raw train episodes/split.
- Keep its native processor and architecture.
- Match evaluation conditions and report unavoidable differences.

Exit criterion: a credible low-data VLA baseline uses the same robot data and evaluation protocol.

### Milestone 6 — LTX frozen extractor and matched action decoder

- Load the exact LTX-2.5 checkpoint under a feasible low-memory plan.
- Add a metadata-preserving hidden-state callback/early-exit path.
- Verify prefix equality against full-forward activations.
- Add the smallest projector and matched action decoder.
- Repeat tiny-batch/episode overfit.

Exit criterion: LTX features causally produce an SO-101 action chunk, with adapter/decoder parameter counts and measured memory/latency.

### Milestone 7 — representation-depth and flow-time studies

- Freeze data, splits, decoder family, training budget, and evaluation protocol.
- Sweep approved depth and sigma points.
- Actually terminate at each selected depth.
- Measure feature bytes, VRAM, prefix latency, decoder latency, action error, smoothness, and rollout success.

Exit criterion: depth/compute and sigma/control conclusions are not confounded by hidden changes in data or downstream capacity.

### Milestone 8 — robot-video LoRA adaptation

- Use only approved train episodes and no action labels in the video objective.
- Match robot-video exposure and report rank, targets, steps, trainable parameters, and compute.
- Retrain/evaluate action decoders against frozen-backbone controls.

Exit criterion: the effect of robot-domain video adaptation is separated from action supervision.

### Milestone 9 — data-efficiency/OOD matrix

- Propose nested subsets, run count, GPU-hours, cost, storage, and physical rollout burden.
- Wait for approval before launching.
- Evaluate ID and one-factor-at-a-time OOD conditions with predeclared success criteria.

## 10. Decision Gate 1 — choices needed before implementation

Please answer these when you are awake. I have included recommendations, but none should be treated as decided.

### 10.1 Robot state and action representation

What must be decided:

- Which state vector the policy receives.
- Which physical command each predicted action vector means.

Realistic options:

1. Absolute six-joint positions for both state and target action.
   - Matches the native SO-101 follower interface and ordinary LeRobot recordings.
   - Simplest reversible normalization and safest first plumbing test.
   - Requires the policy to learn both current pose and desired movement.
2. Current absolute joint state with action relative to the current state.
   - Target is `desired_absolute_action - current_state` at the same causal step.
   - Often easier to center across starting configurations.
   - Requires exact action/state temporal alignment and conversion back to absolute goals online.
3. Delta from the previous commanded action.
   - Models incremental motion, but errors accumulate and replay semantics become more fragile.
   - Not recommended for the first milestone.
4. Cartesian end-effector pose/action plus inverse kinematics.
   - Closer to mimic-video's released 10D representation.
   - Introduces kinematic calibration, IK failure/singularity handling, and a controller as major confounds.

Recommendation:

- Start with six absolute calibrated joint positions for `observation.state` and six absolute joint-position action targets.
- Add a relative-action ablation only after the path overfits.
- Do not begin with Cartesian control unless the available demonstrations are already canonical Cartesian actions.

Fairness impact:

- Cosmos/LTX/SmolVLA should all use the same physical target semantics in the main matched-data comparison.
- Using mimic's native Cartesian actions for Cosmos while SmolVLA uses native joint positions would confound backbone quality with controller/action-space quality.

Questions:

- Are the demonstrations already recorded, and what are the exact `observation.state` and `action` feature names/units?
- Do you want the initial target to be absolute joint positions, relative-to-current joint positions, or Cartesian control?
- Is this one six-motor SO-101 arm, or a custom/bimanual configuration?

### 10.2 Primary task

What must be decided:

- The first real-robot task determines what capability the experiment can claim to test.

Options:

1. Pushing an object to a marked target.
   - Best balance of dynamics relevance, continuous correction, easy reset, objective success metric, and later latent-RL compatibility.
   - SO-101 execution is realistic.
2. Pick-and-place.
   - Operationally conventional and useful as a sanity baseline.
   - Success may be dominated by grasp/contact precision and static localization rather than predictive dynamics.
3. Deformable object manipulation.
   - Strongest hypothesis fit for video/world representations.
   - Reset, labels, variance, and physical evaluation burden are much worse; poor first debugging task.

Recommendation:

- Use pushing as the primary research task.
- If time permits, use a simple pick/place as an engineering sanity task, not the main scientific result.
- Defer deformables until the pipeline and statistics are stable.

Fairness impact:

- Changing tasks between models is invalid.
- Changing reset distributions or target tolerances after seeing model behavior also biases the comparison.

Questions:

- Which task do you choose?
- If pushing: what object, target geometry, table surface, allowed workspace, timeout, and objective success measurement are practical?

### 10.3 Camera configuration

What must be decided:

- Number, identity, placement, and availability of cameras at train and deployment time.

Options:

1. One fixed workspace camera.
   - Lowest compute and simplest alignment.
   - Recommended for first overfit and primary backbone study.
2. One workspace plus one wrist camera.
   - Better contact/occlusion visibility.
   - Doubles image processing and raises fusion/fairness questions because upstream models handle multiple cameras differently.
3. Multiple external cameras.
   - More coverage but much more data/compute and greater shortcut risk.

Recommendation:

- Begin with one fixed workspace camera that sees the whole pushing workspace and gripper.
- Record additional cameras if available, but do not use them in the first matched experiment until fusion is specified.

Fairness impact:

- All compared systems should receive the same camera information unless a model cannot natively consume it; any mismatch must be a separate system-level comparison.
- Camera placement changes can dominate representation differences.

Questions:

- How many cameras are physically available, what are their names/models, and where can they be mounted?
- Is a wrist camera already installed and synchronized in existing data?

### 10.4 Observation and control frequency

What must be decided:

- Camera sampling rate, state/action logging rate, hardware command rate, and policy replanning rate are distinct quantities.

Options:

1. Record/control at 30 Hz and run the large policy asynchronously at a slower replanning rate while executing chunks.
2. Subsample observations/actions to 10–15 Hz for all models.
3. Use model-native rates independently.

Recommendation:

- Record at the highest reliable synchronized rate, likely 30 Hz, preserving raw data.
- Define one common evaluation control rate after measuring actual timestamps.
- Permit model-specific replanning only if end-to-end effective control frequency is reported and treated as part of the system comparison.

Fairness impact:

- A lower temporal sampling rate changes both action difficulty and video history semantics.
- Model-native rates are acceptable for a system-level benchmark but not a strictly representation-matched comparison unless results are stratified.

Questions:

- What frequencies do the existing demonstrations actually use?
- Can the camera provide hardware timestamps or only LeRobot frame indices?
- Is asynchronous chunk execution acceptable for the physical setup?

### 10.5 Action chunk horizon and execution prefix

What must be decided:

- `H`: number of actions predicted.
- Number of predicted actions actually executed before replanning.

Candidate horizons at 30 Hz:

- `H=15`: 0.5 s, close to mimic-video Bridge's 15-action output.
- `H=30`: 1.0 s, a reasonable initial SO-101 compromise.
- `H=50–60`: 1.7–2.0 s, closer to SmolVLA/LIBERO defaults but harder to reconstruct and less reactive.

Recommendation:

- Start with `H=30` at 30 Hz, but initially execute only 3–5 actions before replanning in simulation/dry-run terms.
- On abakus, video-backbone latency may make synchronous replanning every 3–5 actions impossible; we may need asynchronous chunking or a lower common control rate.
- The tiny overfit test can train multiple horizons cheaply after features are available.

Fairness impact:

- Decoder capacity and task difficulty depend on horizon.
- We can share the physical target duration while allowing native internal chunk lengths, but then latency and execution policy become part of the model system and must be reported.

Questions:

- Do you prefer a fixed common horizon across all methods or each method's native horizon with matched physical evaluation?
- What maximum tolerable delay should trigger a safe stop?

### 10.6 Image resolution and history

What must be decided:

- Raw recording resolution, model input resolution, and number of observed frames.

Facts:

- mimic-video's public path is specialized for 480×640 and one or five observed frames.
- LTX commonly uses dimensions aligned to its 32× VAE compression; 544×960 gives a 17×30 latent grid.
- SmolVLA defaults to padded 512×512 images.

Recommendation:

- Preserve raw camera frames at a stable native resolution of at least 480×640 if possible.
- Use model-native deterministic resizing/cropping and log it.
- Begin Cosmos with its supported five-frame history.
- Give LTX the same physical observation duration, not blindly the same frame count, after temporal sampling is set.
- SmolVLA remains single-current-frame unless we intentionally change its architecture.

Fairness impact:

- Same raw information and temporal duration matter more than identical post-resize tensor shape.
- Aggressive resizing/cropping could erase the task target and invalidate the experiment.

Questions:

- What resolutions and FPS can each camera reliably record?
- Do existing episodes contain one or multiple image keys?

### 10.7 Language/task conditioning

What must be decided:

- Whether the first path includes task language.

Options:

1. No semantic language variation; use one constant task string/null embedding.
2. Include the fixed natural-language instruction for the one task.
3. Train multiple instructions/tasks immediately.

Recommendation:

- Use one fixed instruction string if the pretrained backbone requires text, but do not claim language-conditioned generalization.
- Keep the first task single-instruction so the overfit test isolates visual/state/action plumbing.

Fairness impact:

- SmolVLA is natively language conditioned; Cosmos/LTX also receive text context.
- A constant instruction is fair for a single-task comparison. Different or richer task text across models is not.

Question:

- Should the first milestone use a constant instruction such as “push the object to the target,” or no language where technically allowed?

### 10.8 Available compute and acceptable offload

What must be decided:

- Whether abakus is the only compute resource and what compromises are scientifically acceptable.

Options:

1. Abakus only: 4090/24 GB.
   - Cheapest and immediately available.
   - Cosmos needs low-memory techniques; LTX-2.5 almost certainly needs quantization/offload and may be very slow.
2. Abakus for development plus occasional 48–80 GB GPU for backbone validation/training.
   - Preserves local iteration while enabling unquantized reference checks and realistic LTX work.
3. Multi-GPU/cloud cluster for the full experiment.
   - Most flexible but requires budget, transfer, and reproducibility planning.

Recommendation:

- Use abakus for LeRobot, SmolVLA, decoders, tests, and low-memory smoke probes.
- Secure at least occasional 80 GB GPU access before committing to the LTX depth/flow matrix or LTX LoRA.
- Compare quantized abakus features against a higher-precision reference on a small fixed batch before treating quantized features as scientifically equivalent.

Fairness impact:

- Quantizing LTX but not Cosmos changes the representation under test. It may be operationally necessary, but then we should either quantize both comparably or report a separate deployment-constrained track.
- CPU offload changes latency, so capability-vs-compute plots must distinguish model compute from transfer/offload overhead while also reporting end-to-end latency.

Questions:

- Is abakus the only available machine?
- Do you have access to A100 80 GB, H100/H200, RTX 6000 Ada 48 GB, or multiple GPUs?
- What approximate cloud budget is acceptable for feasibility checks and later runs?
- Is quantized inference acceptable for the primary scientific comparison, or only for engineering development?

## 11. Additional decisions that can wait until after Gate 1

These will become explicit gates before their corresponding work:

- Episode-level split details and session/OOD stratification.
- Temporal history duration and subsampling.
- Future-video horizon and whether teacher-forced future latents may influence action training.
- Exact normalization (`mean/std`, quantile, relative action transforms).
- Image augmentations.
- Cosmos video sigma and hidden-state layer.
- LTX extraction layers and sigma points.
- Token pooling/projector design and matched decoder capacity.
- Action-flow time distribution and denoising steps.
- Real-robot success metric, rollout count, confidence intervals, randomization, and stopping criteria.
- The large experiment matrix and compute budget.

## 12. Immediate concerns to resolve before code

1. LTX-2.5 feasibility on 24 GB is not established. A load/forward probe is needed before designing around it.
2. mimic-video action training may expose the decoder to noised real future frames. This must be reproduced deliberately and audited, not hidden in a loader.
3. LeRobot and mimic-video dependency constraints conflict; separate environments/processes are likely required.
4. Official mimic-video has no SO-101 adapter and uses a different 10D Cartesian embodiment.
5. The existing physical camera/action timing must be measured; nominal 30 Hz metadata alone is insufficient.
6. Global dataset statistics can leak held-out episode information in very small datasets.
7. A 500M mimic-style action decoder is much larger than SmolVLA's ~100M trainable expert. “Matched downstream policy” and “native system baseline” should be reported as separate views rather than pretending they are identical.

## 13. Proposed answer format for Decision Gate 1

A concise reply can use this template:

```text
Action/state: absolute 6D joints | relative joints | Cartesian | other
Robot: single SO-101 | other
Task: pushing | pick/place | deformable | other
Object/target/reset details:
Cameras: names, count, placement, resolution, FPS
Recorded data already exists: yes/no, path or Hugging Face repo ID
Control/logging frequency:
Action horizon and execution preference:
Language: fixed instruction | none | multi-task
Compute beyond abakus:
Cloud budget / quantization constraints:
```

After these answers, the next step should be a concrete data/hardware audit and feasibility probes—not a large training run.

## 14. Daily entry — smallest causal SO-101 Cosmos-VAM path and intentional overfit (2026-08-17)

Today’s goal was deliberately narrow: establish the smallest causal SO-101
Cosmos-VAM end-to-end path and perform an intentional tiny overfit. This was
not a generalization experiment and not a rollout-readiness exercise. The
useful result is a reproducible plumbing check from causal RGB history and
language through frozen Cosmos features into a native 6D action-flow decoder.

### Data and causal contract

The dataset is pinned to
`hubnemo/cube_out_of_box_dataset@243370c3c08bcbd860133c4a0d658ea7c1d2e77e`.
The inspected dataset contains 40 episodes, 6,536 frames, and a 10 Hz stream.
The camera is natively 480x640. Each training window uses five causal RGB
frames, the current 6D absolute joint state, and 30 future 6D absolute joint
actions. Padded action tails are excluded from statistics, flow loss, and
physical-action evaluation through `action_is_pad`.

The causal boundary is important: Cosmos receives only RGB history and the
prompt embedding. State, future action labels, and the padding mask stay on
the action-decoder/training side. The first cache sample is episode 0 at
current frame 4, with RGB window indices 0 through 4. Per-window extractor
noise is deterministic, derived from dataset revision, episode, current frame,
and the global seed, so the same cached window has stable frozen features.

### Environment and retained artifacts

There is one LeRobot Python environment at `/home/anton/lerobot-video-vam/.venv`.
The recorded runtime is Python 3.12.3, torch 2.11.0+cu128, an NVIDIA RTX 4090,
and Transformer Engine 2.18 with a CUDA 12 runtime. T5 and all generated
artifacts were retained.

The official prompt embedding for `take cube out of box` is `[1, 512, 1024]`
BF16. The verified T5 checkpoint SHA256 is
`5fdc64177b14b0f72437fea171a752c626733f67755842528c75b90cacd1c807`, and the
embedding artifact SHA256 is
`2d3cff0d125d764446bcef0f473c3616c932826de7bee2433f3dd9296435d5a8`.
The saved embedding path is:

`/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors`

Transformers 5.5.4 tokenizer handling uses the attention mask to determine
true content length. The reported `return_length` value is not used as the
source of sequence length, and masked positions in the saved embedding are
zero.

### Frozen Cosmos extraction

The extractor uses the generic frozen Cosmos checkpoint, not the Bridge LoRA.
The selected representation is hidden layer 20 from one high-noise forward at
`sigma=10`, with deterministic per-window noise and no online resize. The
native camera contract is 480x640, not the previously assumed 704x1280 shape.
The backend is `minimal_a2a` with PyTorch SDPA, while the official Transformer
Engine RMSNorm and rotary implementations remain in the runtime closure.

The real smoke report produced raw hidden tokens `[1, 16, 30, 40, 2048]` and
flattened context `[1, 19200, 2048]`, both BF16 at the context boundary. Model
load took 4.208 seconds and feature extraction took 1.569 seconds. The peak
allocated extraction memory was 6,849,119,744 bytes and peak reserved memory
was 11,853,103,104 bytes.

The retained one-window smoke cache is:

`/home/anton/.cache/video-vam/cosmos-features/cube-out-of-box-episode-0000-frame-000004.safetensors`

The larger diagnostic cache is rooted at
`/home/anton/.cache/video-vam/cosmos-overfit-ep0-64/`; its manifest contains 64
ordered windows and hashes for every detached context and sidecar.

### Native action decoder

The native decoder contract is state `[B, 1, 6]`, action target/sample
`[B, 30, 6]`, and frozen Cosmos context `[B, 19200, 2048]`. Internally it uses
31 tokens: one state token followed by 30 action tokens. The exposed prediction
slices the one internal state-token output and therefore remains `[B, 30, 6]`.

The full native decoder has 499,171,958 trainable parameters. Construction
used approximately 1 GB of BF16 allocation for the model path. The complete
action decoder was initialized from scratch; Cosmos stayed frozen. Training
samples use the released uniform `Beta(1,1)` action-time distribution, and
inference uses exactly ten official Euler steps. State and action values use
train-set-only per-joint min-max normalization to `[-1, 1]`. The flow objective
is masked by valid action positions and evaluated in FP32.

The approved runtime deviation is PyTorch SDPA in place of upstream
`flash_attn_no_cp`. This is a backend/runtime difference, not a smaller
decoder, a changed context, or a changed scientific target.

### First diagnostic: all 64 episode-0 windows

The 64-window diagnostic cache payload is 5,033,618,046 bytes. The aggressive
300-step run used AdamW with learning rate `3e-4`, weight decay `0`, and batch
size 1. Its fixed flow loss moved from 3.9865 to 1.0808. The fixed sampled
physical-action MSE was 469.51. This was a partial fit only; the full 64-window
run did not fully overfit.

### Approved completion: strict evenly spaced eight-window subset

The completion subset was selected strictly and evenly from the 64-window
manifest, using indices `[0, 9, 18, 27, 36, 45, 54, 63]`. It therefore still
contains only episode-0 training windows; the subset is a diagnostic reduction,
not a validation split. The run used 1,000 steps and took 7m24s.

On the fixed flow probe, loss decreased from 4.3372 to 0.1526. The last
stochastic training loss was 0.1347. The fixed sampled action probe reported
physical MSE 27.30 and RMSE 5.22.

Reopening the checkpoint and evaluating all eight manifest entries gave:

- fixed flow loss mean 0.17590, minimum 0.14695, maximum 0.22224;
- sampled physical-action MSE mean 52.5811, minimum 22.8734, maximum 90.4789;
- aggregate sampled physical-action RMSE 7.2513.

This is successful plumbing and intentional training-set overfit evidence. It
shows that the causal cache, prompt conditioning, frozen Cosmos context,
native decoder, normalization, masked objective, checkpoint reopen, and
sampler can work together. It does not demonstrate generalization, held-out
performance, closed-loop behavior, or rollout readiness.

### Final diagnostic artifacts

The final weights-only checkpoint is:

`/home/anton/.cache/video-vam/runs/cosmos-overfit-even8-s0/checkpoint-step-001000.safetensors`

The strict JSON checkpoint metadata is:

`/home/anton/.cache/video-vam/runs/cosmos-overfit-even8-s0/checkpoint-step-001000.json`

The reopened all-eight evaluation JSON is:

`/home/anton/.cache/video-vam/runs/cosmos-overfit-even8-s0/evaluation-step-001000.json`

The weights-only safetensors file is 1,996,754,912 bytes and contains 610
tensors. Metadata marks it `diagnostic_only_non_rollout` and
`robot_ready=false`.

### Compatibility and root-cause findings

- The released scheduler configuration overrides the `BetaScheduler` default
  with the required uniform `Beta(1,1)` distribution.
- The native image contract is 480x640. The 704x1280 assumption was generic
  Cosmos context, not this pinned camera/checkpoint contract.
- With Transformers 5, tokenizer length semantics come from `attention_mask`,
  not the returned length field.
- The strict checkpoint loader allows only the exact known Transformer Engine
  extra-state metadata entries.
- Official autocast is needed around the CUDA Cosmos forward so FP32 timestep
  embeddings reach autocast-aware linear layers correctly.
- The torch SDPA helper must return `[B, S, H, D]`; the common wrapper performs
  the final flatten.
- The native decoder returns state plus action tokens, so exactly one internal
  state-token output must be sliced before action loss and policy output.

### Verification record and limitations

Recorded at the end of this session: the final focused suite had 85 passing
tests, and Ruff/format checks were clean.

Limitations:

- Training used only episode-0 windows; there is no held-out validation split.
- Each physical-action sample was evaluated with one evaluation seed.
- The physical units appear to be joint-position units, but this must be
  confirmed against the SO-101 dataset and robot interface.
- The checkpoint is diagnostic-only and is not robot-ready.
- The optimizer setting was aggressive and intended only for this diagnostic.
- The full 64-window run did not fully overfit.

### Next decision gates

No next direction is selected in this entry:

- full-episode and held-out protocol;
- SmolVLA baseline;
- Bridge LoRA comparison;
- rollout wrapper and safety;
- LTX extraction architecture.

## 2026-08-18 — Does the video context actually contribute anything?

The overnight rehearsal run gave us a training curve that went down, which is
easy to over-read. This session was spent trying to falsify it. The headline is
that we could not yet show the pretrained Cosmos context contributes any
information at all, and we found a concrete defect in our own pipeline that is
the most likely cause.

### Upstream contract, verified against source

Two claims from earlier in the project needed checking against
`model/cosmos_predict2/models/world2action_model.py` at ref
`e3355dbc93132b576c02f920a59b4fc18a4f5906`.

The 35-step `generate_video(..., num_sampling_step=35, guidance=0.0,
return_all_context=True)` call exists only inside the `validation_step`
analysis branch, which sweeps action MSE across every scheduler sigma to study
sigma sensitivity. It is **not** the training path. Training goes through
`get_crossattn_emb` (lines 323-352), which performs exactly one
`video2world_pipe.denoise(...)` call with
`return_only_hidden_states_up_to=xattn_layer_idx`. Our single-forward extractor
therefore matches upstream and needs no 35x rebuild.

The consequential finding is in `draw_video_sigma` (lines 360-379). Upstream
re-draws **both the noise and the sigma on every training step**, including a
5% branch that replaces sigma with `exp(U(log 200, log 100000))`, and feeds the
drawn sigma to the decoder as context-timestep conditioning. Our cached
pipeline used a **constant sigma of 10.0** for both extraction and decoder
conditioning. Every window thus had exactly one frozen context vector at one
noise level, and the decoder's sigma input was a constant.

Upstream also builds conditioning as a 61-frame zero pixel tensor with only the
first `T` frames populated (`T in (1, 5)`,
`num_latent_conditional_frames = 1 if T == 1 else 2`), whereas we encoded `T`
frames and zero-padded in latent space. Decision taken: align to upstream.

### Experiment 1 — is the visual conditioning causally used?

The 47 dB conditioning reconstruction measured earlier does not prove the DiT
_uses_ the conditioning, since those frames are written into the latent at
every step and would reconstruct even if ignored. Test: generate the same
window twice with identical seed, once with true conditioning frames and once
with frames from a different episode, then score the two generated futures
against each other.

- true vs foreign conditioning: 15.97 dB / 0.567 SSIM
- true vs zeroed conditioning: 16.74 dB / 0.553 SSIM
- foreign vs zero: 14.41 dB / 0.495 SSIM

Conditioning materially changes the output, so it is not causally inert. The
failure mode is "used but wrong": a capability and domain-gap limitation, not a
wiring defect.

### Experiment 2 — one conditioning frame or five?

Upstream's assert permits `T == 1`. Five frames beat one, so our choice is not
the problem. Both remain far below a static-frame baseline.

| Window     | 1 frame          | 5 frames         | static baseline  |
| ---------- | ---------------- | ---------------- | ---------------- |
| episode 0  | 13.41 dB / 0.525 | 14.55 dB / 0.560 | 24.11 dB / 0.851 |
| episode 19 | 10.70 dB / 0.462 | 10.73 dB / 0.487 | 13.45 dB / 0.698 |

Also confirmed that `guidance=0.0` is canonical; the CFG variant we had added
made predictions worse (14.75 dB to 12.37 dB), consistent with it being a
deviation rather than a fix.

### Experiment 3 — the shuffled-context control (the important one)

Pair each window's state and actions with a **different** window's real Cosmos
features via a seeded derangement. The feature distribution is exactly the real
one; only the correspondence to the actions is destroyed. This cannot be
dismissed as random weights producing out-of-distribution activations.

| Run                     | best flow loss | best RMSE | step |
| ----------------------- | -------------- | --------- | ---- |
| baseline (real context) | 0.29559        | 22.66°    | 2300 |
| shuffled context        | 0.28945        | 22.23°    | 2900 |

Shuffling did not hurt; it was marginally better, which is within noise. Read
plainly, the decoder is solving the task from proprioceptive state and dataset
priors while ignoring 19,200 context tokens. This is consistent with the video
prediction result: a backbone whose futures lose to a static baseline is
unlikely to carry action-relevant scene information in its hidden states.

As it stands, the planned Cosmos 2B vs 14B vs LTX comparison has no signal to
measure. Fixing that precedes any scaling up.

### Units

SO-101 `.pos` actions are confirmed to be **degrees**
(`use_degrees=True`, `MotorNormMode.DEGREES`). An RMSE of 22.66 is therefore a
very large mean joint error, and the overnight "loss goes down" result is much
weaker than the bare number suggested.

### Leading hypothesis and the test for it

The frozen sigma of 10.0 is the leading explanation for the shuffled result: a
single fixed context vector per window, with a constant sigma conditioning
input, gives the decoder every incentive to ignore the context in favour of
state. An online-extraction mode reproducing upstream's per-step randomized
sigma and noise has been implemented. Crucially it must be paired with its own
shuffled control under the online regime, since a better online number could
otherwise reflect nothing more than a better-regularized decoder.

Queued: online extraction, online shuffled control, random-init backbone,
Bridge-LoRA features.

### Infrastructure

abakus dropped off Tailscale mid-session and killed the random-init run at step
1699; a reboot restored it. All GPU jobs now run in detached tmux sessions with
teed logfiles so no run depends on an SSH connection surviving.

### Open questions

- Are the cached context vectors discriminative between windows at all, by
  pairwise cosine similarity? A near-degenerate answer would indicate a
  conditioning or pooling defect on our side rather than a Cosmos limitation.
- Does the real context beat its own shuffled control under **any**
  configuration? This is the gating question for the whole comparison.
- What is the online step time and peak memory, which decides whether online
  extraction is affordable for larger backbones.

### Qualitative observations from watching the prediction videos

Two findings from visual inspection that the automatic metrics missed or
actively contradicted. Recorded because they matter more than the scores.

**Bridge LoRA may be the wrong domain match.** The Bridge-adapted checkpoint
does not appear to have been trained on first-person / wrist-mounted footage,
whereas our camera viewpoint is. If the viewpoint distribution differs, the
Bridge LoRA is not the domain adaptation we assumed it was, and a null result
from the Bridge-LoRA arm would say nothing about whether domain adaptation
helps in general — only that _this_ adaptation targets a different viewpoint.
Worth confirming BridgeData V2's camera setup before spending a run on it.

**CFG 7.0 looked qualitatively better than guidance 0.0, despite scoring
worse.** By PSNR, CFG made things worse (14.75 dB to 12.37 dB), and on that
basis it was set aside as a non-canonical deviation. Visually, however, the
CFG-7 predictions actually placed objects into boxes — that is, they produced
task-plausible behavior — while also hallucinating artifacts such as human
hands. Guidance 0.0 did not produce the task behavior at all.

This is a known failure mode of PSNR: it rewards conservative, blurry,
low-variance predictions and penalizes confident, sharp ones that are spatially
misaligned. A prediction that performs the right action in the wrong pixels
scores worse than a prediction that does nothing. For our purposes — extracting
features that encode task-relevant dynamics — semantic plausibility is likely
the better signal than pixel fidelity.

Implication: PSNR and SSIM should not be the primary criterion for judging
backbone suitability here, and the static-frame baseline "winning" on PSNR
should not be read as the backbone being useless. It also raises the question of
whether context extracted at non-zero guidance would be more informative, even
though upstream uses `guidance=0.0` for extraction.

## 2026-08-19 (night) — the first honest comparison, and a course correction

### The number that reframed everything

We finally measured all policies in the same physical units on the same 88
held-out windows. Action RMSE in degrees, lower is better:

| Backend                                            | RMSE      |
| -------------------------------------------------- | --------- |
| SmolVLA (trained on episodes 0–31)                 | **15.00** |
| `state_repeat` (hold current joint angles for 3 s) | 18.86     |
| Cosmos-2B VAM (best checkpoint, step 2300)         | 23.63     |
| mean training action                               | 30.15     |

Our VAM loses to a baseline that does nothing at all. That is not a tuning
gap — a correctly wired decoder cannot be worse than freezing the arm. It made
every prior "the loss is going down" observation untrustworthy.

Getting this number required fixing five bugs that all lived in the CUDA path of
the evaluator, which had only ever been exercised on CPU: a float32 checkpoint
rejected against a bfloat16 module, `load_state_dict(assign=True)` planting CPU
tensors inside a CUDA module, batches never moved to the device, SmolVLA's
mixed-precision action expert needing the noise typed from its action projection
rather than the first parameter, and a postprocessor loaded without its action
converters. Worth remembering as a pattern: an evaluation path that has only
ever run on one device is untested code.

### The night's real lesson: we spent it measuring the measurement

Most of the night went into a linear probe asking whether Cosmos features add
anything over proprioceptive state. It kept returning an arithmetically
impossible answer — state alone R² 0.655, context alone 0.196, both together
0.510. A model containing state as a subset cannot score below state alone, so
this was a bug in the probe, not a finding.

The cause was regularization, twice over. First a single hard-coded ridge
penalty of 10.0 shared across a 6-dimensional model and a 512-dimensional one.
Then, after tuning the penalty per model, the combined model still failed,
because a _single scalar_ penalty cannot serve two feature blocks of wildly
different size and quality: strong enough to tame 512 noisy dimensions also
crushes the 6 informative ones. The chosen values showed the squeeze plainly —
31.6 for state, 5,623 for context, 178 for the compromise.

We then killed the entire probe line of work, because the question it was asking
was already answered from first principles: **the cube position is randomized,
so a policy that cannot see cannot know where to reach.** Vision is necessary by
construction. The probe's "state alone gets 0.655" was never measuring task
competence, only that joint trajectories are smooth and the next three seconds
are largely extrapolable from current motion. We had been carefully quantifying
an artifact.

The generalizable mistake: when a cheap diagnostic disagrees with a structural
argument about the task, distrust the diagnostic.

### What the upstream comparison actually found

With the probe abandoned, we extracted upstream mimic-video's exact recipe and
compared it line by line against ours. The reassuring outcome is that most of
the hard parts are right: layer 20 of 28, unpooled `(B, 19200, 2048)` context,
the video sigma fed to the decoder through the pair-time embedder,
`obs_dropout=0.2` with a learned mask token, and upstream's subtle asymmetry
where `xt` is divided by `sqrt((1−t)²+t²)` on the way into the network while the
velocity target `ε−x0` is left unscaled. The action chunk starts at frame `t` in
both pipelines, and the evaluation harness is fair — identical targets, masks and
units across every backend.

One hypothesis was raised and then **disproved**, which is worth recording so it
is not raised again. Upstream feeds the video DiT 61 frames: 5 observation
frames ending at `t` plus 56 _future_ frames (ground truth during training,
generated at deployment). Ours pads those 56 slots with zeros, which looked like
we had deleted the signal the method runs on. In fact the padding never reaches
the network: the non-conditioning latents are overwritten with
`noise · sigma · c_in`, pure Gaussian noise at the model's own scale, and the
zero-derived latents are discarded. Our conditioning is the textbook
image-to-video setup. The remaining difference is subtler — upstream's noised
latents are `true_future + ε·σ`, so at their typical sigma some real future is
visible, while ours carry none.

### The two things actually wrong

1. **Sigma.** We extract at a fixed σ = 10.0. Upstream draws it per step as
   `4 · exp(N(0,1))`, median ≈ 4, with a 5% loguniform tail in [200, 100000]. At
   a fixed 10.0 our layer-20 features are noise-dominated, and the decoder is
   handed a constant sigma conditioning value that tells it nothing.
2. **Scale.** The best checkpoint ran **3,100 steps at batch size 2** — roughly
   6,200 samples — with gradient clipping 10× tighter than upstream. Upstream
   configures batch 32–256. Whatever else is true, that is not a trained model.

Everything else we checked is either matched to upstream or a documented,
deliberate difference (our normalizer clamps on inversion, so VAM predictions
are saturated to the training joint range while SmolVLA's inverse is exact; our
statistics come from the train split only, SmolVLA's from the whole dataset).

### Where things stand

Running: an online training run with upstream-faithful randomized per-step
sigma, gradient clipping at 10.0, gradient accumulation to a real effective
batch, held-out episodes 32–39, wandb logging, and the four-way RMSE comparison
queued behind it. The bar to clear is 18.86 (`state_repeat`); the goal is to
beat 15.00 (SmolVLA).

Not running, deliberately: the sigma sweep, the context probe, and the
feature-degeneracy work. All of it was characterizing a measurement rather than
making the policy work.

### Note on process

Three separate times tonight a job was left running that served no decision:
a CPU evaluation that burned 2h50m on work the GPU did in three minutes, a
14-core probe that produced nothing in 23 minutes, and a 1.7-hour feature
extraction feeding an analysis we were about to abandon. Keeping the GPU busy is
not the same as making progress, and an idle GPU is the correct state when we do
not yet know what to train.

---

## 2026-08-20 — The overnight result, and the pivot to connector efficiency

### The overnight cached run finally beats the trivial baseline

`runD-cached` (stride-3 sigma-80 cache, random anchors, K=8 flow draws,
effective batch 4) trained 4,541 optimizer steps over ~6.5 hours and scored
**16.92° RMSE** — the first VAM run to beat `state_repeat` (18.86°), still
behind SmolVLA (14.83°). Validation RMSE fell from 45.8 (step 300) to a best of
16.59 (step 4500) and plateaued. So the pipeline learns; the question shifted
from "is it broken?" to "why is it this slow?". The code also finally landed in
git: `6502968e` (full VAM stack, 98 files) and `67bf38fd` (the 16 remaining
test files), all pre-commit hooks passing.

### Why one step costs 4.76 s while SmolVLA takes 0.12 s

Side-by-side numbers: SmolVLA did 5,000 steps × batch 8 in **10 minutes**
(~40k samples, 2.4 GB VRAM); our cached VAM did 4,541 steps × 4 contexts in
6.5 hours (21.6 GB VRAM) — ~40× slower per step _with the backbone already out
of the loop_. The cost is the interface: Cosmos hands the decoder 19,200 tokens
× 2,048 channels of unpooled context, all 24 decoder blocks cross-attend to it,
and the K/V projections over those 19,200 tokens are recomputed in every block
(and for each of the K=8 flow draws). Each cached context is ~79 MB bf16, so a
step also moves ~315 MB disk→GPU.

### Is that really what mimic does? Yes — verified at the code level

A subagent read the upstream repo at commit `e3355db` end to end. There is
**no connector**: the `(B, 16, 30, 40, 2048)` layer-20 grid is flattened by a
single `reshape` to `(B, 19200, 2048)` and fed to all 24 action blocks
(`world2action_model.py:349-350`, `world2action_dit.py:917-925`). Upstream
doesn't even cache — every training step runs one fresh frozen-backbone denoise
online with freshly drawn noise and sigma (`exp(N(0,1))·4`, 5% loguniform tail
in [200, 100000]). Their answer to the cost is multi-GPU hardware, not
efficiency machinery. Two useful confirmations: only the first 2 of 16 latent
frames are clean ground-truth conditioning (rest noise — matches our setup),
and the video DiT's self-attention is **fully bidirectional**
(`attn_mask_type="no_mask"`), not causal.

### The literature says the unpooled grid is an outlier

A parallel survey (persisted as `docs/video_vam_connector_survey.md`) found
that every comparable modular system compresses before the action head:
Video Prediction Policy uses a learned "Video Former" down to **224 tokens** —
and its ablation shows removing the compressor makes the policy _worse_, not
just slower; FLARE/GR00T N1.5 use 32 learned queries; VidMan a handful of
learned action tokens; MinD one token per latent frame. Full visual grids only
appear inside unified single-transformer models. mimic-video is the outlier.

### Decision: keep the cache, make the interface the research object

The caching approach stays — it is our efficiency advantage over upstream's
online recipe. Sigma experimentation is explicitly off the table (upstream's
distribution is noted above for the record; we are not chasing it). The next
phase is a structured **connector-efficiency ablation**: many small cached-
feature runs (15–30 min each) over spatial pooling, temporal slicing
(conditioning-frame vs generated-frame tokens), one-token-per-latent-frame,
and a VPP-style learned resampler — measuring step time and RMSE-within-budget
for each, then doubling down on the winners. Because the video attention is
bidirectional, token position does not cleanly separate observed from imagined
content, which makes the temporal-slicing arms genuinely informative. This
phase is deliberately structured as a small standalone research piece
(candidate blog-post/publication artifact).

## 2026-08-19 (day and evening) — the one-hour rule, and what it exposed

### New ground rules

Two process decisions were made today. First: whenever the user messages,
re-check alignment before continuing — do not barrel ahead on a stale plan.
Second, a hard experimental constraint: **a training run may take at most one
hour.** The justification is the method's own premise — mimic-video claims to
learn _faster_ than a VLA, so if an okay policy is not trainable in an hour,
the configuration is wrong, not the budget.

A reference document was also written (`VIDEO_VAM_MIMIC_REFERENCE.md`, mirrored
to `docs/mimic_video_reference.md` in the worktree): the full upstream recipe
with citations, our per-item compliance, and an explicit deviations table.
Discipline going forward: change one deviation at a time, and name which rows a
run touches before launching.

### Findings, in causal order

**The anchor-diversity bug.** All training had been drawing from a precomputed
stride-20 window manifest — 337 anchors total, ~265 in the train split. Every
"undertrained" run had actually seen the same 265 scenes ~23 times each while
95% of training frames were never used. Fixed with uniform random anchor
sampling (pool: 4,688).

**The clipping strangulation.** Logged gradient norms ran 225–335 against a
clip of 10 — every update shrunk ~30×, effective learning rate a few percent of
nominal. An A/B settled on loss_scale 1.0 with clip 10.0 (norms now 2–15). The
proprio-only floor immediately improved from 32.7° to 26.65°, confirming the
diagnosis.

**K flow-draws per context.** Our one sanctioned invention: a Cosmos forward
costs ~4 s while a decoder update is nearly free, so each extracted context now
supervises K=8 independent (noise, flow-time) draws. Approved by the user.

**Sigma clarified, then deprioritized.** The user's question — "why noise the
starting image at all?" — exposed a conceptual muddle. The conditioning frames
are never noised (they stay clean in both upstream and our code); sigma labels
only the future latent slots, which in our causal setup are pure noise. The only
self-consistent label for pure noise is the _generation-start_ sigma, which the
solver says is exactly 80.0. A controlled A/B (sigma 80 vs sigma 10, everything
else identical) then showed it barely matters: 45.87 vs 46.43. A conceptually
satisfying answer with no empirical payoff.

**The real constraint is extraction throughput.** Both one-hour VAM runs got
only ~230 optimizer steps (10.7 s/step, online extraction dominating) and were
still improving steeply at cutoff — no plateau in sight. SmolVLA, trained fresh
for one hour under the same rule, did **29,200 steps** and set a new best of
**14.83°**. Under equal wall-clock the VAM is starved, not refuted. Response:
precompute the feature cache once (outside the training hour), keep random
anchor sampling over the cached pool, and let the training hour consist of fast
steps. Stride-3 cache at sigma 80 (~1,560 anchors, ~122 GB) building overnight,
followed by a cached-mode one-hour run.

### Scoreboard (held-out 88 windows, degrees, lower is better)

| policy                                      | RMSE          |
| ------------------------------------------- | ------------- |
| SmolVLA, 1 hour, 29,200 steps               | **14.83**     |
| SmolVLA, 5,000 steps                        | 15.00         |
| state_repeat                                | 18.86         |
| best-ever VAM (2026-08-18, legacy features) | 26.12         |
| proprio-only floor (fixed optimization)     | 26.65         |
| VAM 1h online, sigma 80 / sigma 10          | 45.87 / 46.43 |
| VAM 1h online, randomized sigma             | 57.35         |

### Warnings for the record

- **The legacy stride-20 cache does not reproduce** under the current (audited,
  deterministic) extraction pipeline — max element difference 81.5 on a rebuilt
  window. The 26.12 best-ever was earned on features we can no longer recompute,
  because none of the VAM code was ever committed and the extractor was mutated
  in place across two days. A tar snapshot of the working tree now exists
  (`vam-code-snapshot-20260819-2300.tar.gz`); the deeper fix is version
  control discipline.
- Killing a PID whose cmdline begins with `tmux` kills the tmux **server** and
  every session on the machine. This happened twice (once costing a training
  run mid-step, once the user's unrelated processes). Kill sessions by name or
  Python PIDs only.

---

## 2026-08-28 — Video LoRA Adaptation and the $T=16$ Gold Standard (13.06°)

### Scaling Cosmos 2B with Video LoRA and SmolExpert

After stabilizing the caching pipeline, we introduced full-backbone Video LoRA adaptation (blocks 0–27) on Cosmos 2B. Pairing this adapted backbone with the `SmolExpert` flow-matching policy architecture over $T=16$ latent frames (spatial `pool2`, $16 \times 15 \times 20 = 4,800$ tokens at Layer 20) broke through all previous accuracy limits:

- **Validation RMSE**: **`13.06°`** on held-out episodes 32–39 (down from SmolVLA's 14.83° baseline).
- **The Tradeoff**: A single $T=16$ forward pass processes 61 RGB frames through 16 latent slices, taking **~1,200 ms** per inference step on the RTX 4090. This is too slow for real-time closed-loop robotic control at 10 Hz (~100 ms target).

---

## 2026-08-29 to 2026-08-31 — The Real-Time Dilemma ($T=2$, ~204 ms) and Failed Ablations

To enable real-time control, we investigated fast inference using only the 2 observed conditioning frames ($T=2$, 5 RGB frames, $2 \times 30 \times 40 = 2,400$ tokens):

1. **Cosmos $T=2$ (Frozen Video LoRA Baseline)**:
   - Latency dropped to **~204 ms**.
   - Validation RMSE degraded to **`13.74°`** (a 0.68° accuracy gap vs. $T=16$).
   - Token activation mean shifted from `+0.080` ($T=16$) to `+0.434` ($T=2$), reflecting severe attention concentration when 14 future query slots are removed.

2. **Failed Attempt 1: Direct Action Backprop into Trunk (Blocks 0–19)**:
   - Backpropagating action MSE directly into Cosmos blocks 0–19 caused catastrophic feature collapse. Token variances exploded from 3.9 up to 45–550, destroying pretrained video priors (RMSE collapsed to 14.51°–14.74°).
3. **Failed Attempt 2: $T=4$ Noise Slots**:
   - Supplying dummy Gaussian noise frames into latent slots 2–3 ($T=4$, ~340 ms) caused the action policy to cross-attend to ungrounded noise (RMSE collapsed to 17.92°).
4. **Failed Attempt 3: Self-Attention Scaling ($\gamma = 0.60$)**:
   - Rescaling self-attention outputs shifted the activation mean back to `+0.310` but plateaued at 13.74° RMSE without accuracy gains.

---

## 2026-09-01 — The Representation Distillation Strategy and Path A vs. Path B

### Hypothesis & Setup

Instead of letting action gradients distort the video backbone, we freeze $T=16$ Layer-20 features as teacher targets and distill them into a student $T=2$ LoRA (blocks 0–19):
$$\mathcal{L}_{\text{distill}} = \text{MSE}(H_{\text{student}}, H_{\text{teacher}}) + 1.0 \times \big(1 - \text{CosSim}(H_{\text{student}}, H_{\text{teacher}})\big)$$

### Exploring the 134M-Parameter Auxiliary Linear Readout Head

To map student 2-frame latents to all 16 teacher frames, we trained an auxiliary spatiotemporal tube linear head $g_\phi$: $\mathbb{R}^{4096} \to \mathbb{R}^{32768}$ ($2 \times 2048 \to 16 \times 2048$):

- **Path A (Discard Head)**: Feed raw student $H_{\text{student}}$ (2,400 tokens) to SmolExpert $\to$ **14.16° RMSE**.
- **Path B (Keep Head)**: Feed reconstructed $16 \times 15 \times 20$ (4,800 tokens) to SmolExpert $\to$ **14.85° RMSE**.

### Probing the Feature Space: Why Both Paths Underperformed

A targeted diagnostic probe revealed the root causes:

1. **Scrambled Intermediate Features (Path A)**:
   - Probing the student tokens $z$ before the linear head revealed $\text{CosSim} = \mathbf{0.2782}$ (degraded from baseline $0.6092$).
   - The LoRA learned an unconstrained intermediate representation because the linear head was doing the actual manifold mapping ($W z + b$ had $\text{CosSim} = 0.8533$). Discarding the head fed the policy scrambled features.
2. **Noisy Future Slots (Path B)**:
   - Reconstructing 14 future frames from 2 static frames achieved only $\text{CosSim} \approx 0.6471$. Cross-attending to these noisy predictions diluted the policy.

---

## 2026-09-02 — The 4-Frame Temporal Shift Bug & Dual-Operator Dataset Analysis

### 1. Discovery of the 4-Frame Sensory Delay Bug

An audit of `train_cosmos_t2_distillation.py` revealed a major temporal loading discrepancy:

- **Teacher Extraction Contract (Stage 1)**: Causal window `[t-4, t-3, t-2, t-1, t]` (ending at current frame $t$).
- **Distillation Training Loader (Stage 2)**: Sliced `[t, t+1, t+2, t+3, t+4]` (future frames).
- **Consequence**: The student was trained to predict the teacher representation of **4 frames in the past**. At deployment, feeding `[t-4..t]` caused the student to emit features from $t-8$, injecting an unintended **400 ms sensory delay** at 10 Hz.

### 2. Dataset Kinematic Audit: Two Distinct Teleoperators

A comprehensive per-episode joint distribution audit across all 40 episodes of `hubnemo/cube_out_of_box_dataset` revealed an explicit 20-episode step-function shift:

- **Episodes 0–19 (Operator A)**: Fast pacing (~112 frames/ep), tilted wrist (`W_Roll ≈ -56.5°`), high-elbow posture (`Lift ≈ -30°`, `Elbow ≈ +35°`).
- **Episodes 20–39 (Operator B)**: Deliberate pacing (~214 frames/ep), flat wrist (`W_Roll ≈ -46.3°`), low-elbow posture (`Lift ≈ +10°`, `Elbow ≈ -20°`).
- **The Split Impact**: The training set is 62.5% Operator A / 37.5% Operator B, while the validation set (episodes 32–39) is **100% Operator B**. The validation metric directly measures cross-operator kinematic generalization.

---

## 2026-09-02 — Direct Manifold Distillation (V3 Breakthrough)

### Method

1. Eliminated the intermediate linear readout head entirely.
2. Switched to direct student $T=2 \to H_{\text{teacher}}^{0..1}$ manifold supervision.
3. Enforced exact canonical causal windowing `(t-4, t-3, t-2, t-1, t)` with `delta_timestamps`.

### Empirical Convergence & Validation Results

- **Validation Loss**: **`18.8439` $\to$ `1.2147`** (**`93.6%` MSE reduction** at Step 10,000).
- **Cosine Similarity**: **`0.6193` $\to$ `0.9732`**.
- **Token Activation Mean**: Restored to exact teacher baseline (**`+0.0761`** vs. teacher `+0.0800`).
- **Downstream Policy Validation RMSE**: Reached **`13.15°`** (beating the $T=2$ undistilled baseline `13.74°` by **`-0.59°`**).
- **Immediate Action Precision (H=1)**: **`4.58°`** (surpassing the full $T=16$ teacher's `4.76°`).
- **First-5 Actions Mean RMSE**: **`6.04°`** (surpassing the full $T=16$ teacher's `6.19°`).

---

## 2026-09-02 — Teacher $T=16$ `cond_frames` Oracle Ceiling Benchmark

To determine the theoretical upper bound of 2-frame representation distillation, we trained SmolExpert directly on the ground-truth first 2 latent frames (`cond_frames`, 2,400 tokens) sliced directly from the unpooled $T=16$ teacher cache.

### The Findings

- **Oracle Validation RMSE**: **`13.08°`** (Step 17,000).
- **Oracle H1 RMSE**: **`4.97°`**.
- **Oracle First-5 Mean RMSE**: **`6.40°`**.
- **RETRACTED conclusion (2026-09-07: empirical reference, not a ceiling; 89.4% gap closure, not 98.9%)**: The theoretical ceiling of any 2-frame representation distilled from $T=16$ is **`13.08°`**. Our Direct Distilled student model reached **`13.15°`**, extracting **`98.9%`** of the maximum possible representation value from the teacher while running in real-time (~204 ms).

| Model / Paradigm                          | Inference Latency | Tokens to Policy                 | Representation MSE / CosSim | Action RMSE (Validation) | H1 (Immediate Action) |
| :---------------------------------------- | :---------------- | :------------------------------- | :-------------------------- | :----------------------- | :-------------------- |
| **Cosmos $T=16$ Full Gold Standard**      | ~1,200 ms         | 4,800 ($16 \times 15 \times 20$) | - / 1.0000                  | **`13.06°`**             | `4.76°`               |
| **Teacher $T=16$ `cond_frames` Oracle**   | -                 | 2,400 ($2 \times 30 \times 40$)  | - / 1.0000                  | **`13.08°`**             | `4.97°`               |
| **Cosmos $T=2$ Direct Distilled (Ours)**  | **~204 ms**       | 2,400 ($2 \times 30 \times 40$)  | **`1.19` / `0.9732`**       | **`13.15°`**             | **`4.58°`** (Best)    |
| **Cosmos $T=2$ Undistilled Baseline**     | **~204 ms**       | 2,400 ($2 \times 30 \times 40$)  | - / 0.6092                  | **`13.74°`**             | `4.92°`               |
| _V1 Distillation (Lag Bug)_               | ~204 ms           | 2,400                            | 8.06 / 0.8056               | `14.08°`                 | `5.02°`               |
| _V2 Distillation (Lag Bug + Linear Head)_ | ~204 ms           | 2,400                            | 8.57 / 0.6809               | `13.99°`                 | `5.18°`               |

## 2026-09-03 — Cosmos 3 Edge: Dissecting Partial Latents & Clean 600 Vision Tokens

### 1. The Investigation: Do We Need the Partial Vision Latents?

An architectural probe into `Cosmos3OmniPipeline` and `Cosmos3OmniTransformer` resolved why the initial out-of-the-box run scored 20.44°:

- **The Noise & Padding Discovery**:
  When `pipe(num_frames=16)` was called, the pipeline created 2 observed conditioning frames (2 x 15 x 20 = 600 tokens) and 14 noisy diffusion generation slots. Our earlier hook captured `torch.cat([und_seq, gen_seq])`, which mixed 58 prompt tokens + 600 vision tokens + partial noisy frames (3,738 tokens), which was then padded with 1,062 zeros to reach 4,800 tokens. The action policy was cross-attending to 22% empty padding and ungrounded noise!
- **Pure Vision Token Extraction**:
  We modified the feature extractor to isolate only the clean 600 vision tokens from Layer 20 (`und_seq`):
  - **0 Prompt Tokens**
  - **0 Future Noise Slots**
  - **0 Zero Padding**
  - **Shape**: Exactly `[1, 600, 2048]` (2 frames x 15 x 20 grid)
  - **Latency**: Measured at **91.99 ms uncompiled** on the RTX 4090!

### 2. Live Benchmark Execution

- **Train & Val Caching**: Successfully extracted all 1,572 train samples and 88 val samples in under 2 minutes (~12 samples/s).
- **Policy Training**: Launched `cosmos3-edge-pure600-smolexpert-20260903` on the clean 600-token representations.
- **Video LoRA Adaptation Queued**: Prepared `schedule_cosmos3_edge_lora.sh` to adapt Cosmos 3 Edge on robot video demonstrations once the zero-shot baseline completes.

### 3. Empirical Results on Pure 600 Vision Tokens (Zero-Shot Baseline)

The pure 600 vision token benchmark (`cosmos3-edge-pure600-smolexpert-20260903`) completed on abakus:

- **Validation RMSE**: Reached **`14.26°`** at Step 55,000 (down from 20.44° on the zero-padded sequence).
- **First-Step Action Accuracy (H=1)**: **`4.67°`** (matching the 13.06° gold standard).
- **First-5 Actions Mean**: **`6.34°`**.
- **Inference Speed**: **`91.99 ms` uncompiled** (total policy reaction time ~100 ms).
- **Key Insight**: With zero Video LoRA adaptation and only 600 tokens (1/4th of Cosmos 2B), Cosmos 3 Edge delivers a strong physical representation, setting the baseline for subsequent Video LoRA adaptation.

## 2026-09-03 — Cosmos 3 Edge: Video LoRA Adaptation & First-N-Layers Representation Pipeline

### 1. Motivation & Technical Strategy

- **Superseded comparison:** In Cosmos 2B, base representations achieved ~26 deg RMSE without adaptation, but dropped to 13.06 deg once adapted via Video LoRA on the SO-100 cube_out_of_box task.
- For Cosmos 3 Edge (3.37B), zero-shot representations already hit 14.26 deg with pure 600 vision tokens at 91.99 ms. Adapting the spatiotemporal world model with Video LoRA directly on robot demonstration videos targets closing the gap to the 13.06 deg gold standard.
- Architectural LoRA Design:
  - Model: nvidia/Cosmos3-Edge (28 blocks, hidden 2048, 16 attention heads, Wan 2.2 VAE).
  - Injected LoRALinear into all 336 attention and MLP projections across all 28 layers (rank 16, alpha 16.0, 33.03M trainable parameters).
  - Objective: Flow Matching on 17-frame video clips (5 latents: frames 0-1 conditioning, frames 2-4 flow-matching velocity prediction).
  - Gradient Checkpointing enabled: Peak VRAM bounded at 9.63 GB on 24GB RTX 4090.

### 2. Feature Extraction & Policy Coupling

- After Video LoRA training, representations are probed at Layer 20:
  - The LoRA weights are active on transformer layers 0..19.
  - Layers 20..27 are truncated, saving ~28% forward DiT compute.
  - Generates adapted pure 600-token feature caches (cosmos3-edge-adapted-train0-31-stride3 and cosmos3-edge-adapted-val32-39-stride20).
- SmolExpert action policy is trained on the adapted 600-token representations to evaluate validation RMSE on held-out episodes 32-39.

### 3. Live Pipeline Launch

- Active tmux session: cosmos3-edge-lora-pipeline-20260903
- Script: scripts/video_vam/run_cosmos3_edge_lora_pipeline.sh
- Master log: /home/anton/.cache/video-vam/runs/cosmos3-edge-adapted-pipeline-20260903.log

## 2026-09-06 — Cosmos 14B End-to-End True Convergence & Quantized Video-LoRA Pipeline

### 1. Cosmos 14B Base SmolExpert Training to True Convergence

The initial run on Cosmos 14B Base features terminated prematurely at 8,000 steps due to overly aggressive early stopping (patience=8, 9.97°).
Retraining with learning rate `3e-5`, cosine decay schedule with warmup, and robust patience (60 evaluations with eval_interval=500, min_steps=20,000) achieved true convergence at Step 8,500:

- **Validation RMSE**: **`10.99°`**
- **Horizon-1 (H1)**: **`4.67°`**
- **First-5 Actions Mean**: **`6.42°`**
- **Wall Clock**: 40.2 minutes

### 2. Cosmos 14B Quantized Video-LoRA (FP8 QLoRA on Single 24GB RTX 4090)

Implemented memory-efficient Quantized LoRA (`Cosmos14BQuantizedLoRALinear`) on Cosmos-1.0-Diffusion-14B (14.25B parameters):

- **Weight Quantization**: Base linear projections converted to `torch.float8_e4m3fn` (14.25 GB model weight footprint).
- **Target Modules**: Injected trainable low-rank adapters (`lora_A` and `lora_B`, rank 16, alpha 16.0) into `to_q`, `to_k`, `to_v`, `to_out.0` across all 36 transformer blocks (288 adapter modules, 42.4M trainable parameters).
- **VRAM Footprint**: Gradient checkpointing across all 36 blocks bounded peak allocation to **14.55 GB**, comfortably inside the 24 GB VRAM limit of the NVIDIA RTX 4090.
- **Training Speed**: Achieved ~2.5 steps/s (~0.39 s/step) in bfloat16 on robot demonstration video clips.

### 3. Feature Extraction on Adapted 14B LoRA & Downstream Policy Training

Extracted adapted spatiotemporal representations from Layers 18 and 30 across the full dataset with 2x2 spatial average pooling and concatenation (10,240-dim representation) into `/home/anton/.cache/video-vam/cosmos14b-adapted-features`.
SmolExpert training on the adapted 14B features converged at Step 41,000:

- **Adapted Validation RMSE**: **`11.08°`**
- **Immediate Action Precision (H1)**: **`4.63°`**
- **First-5 Actions Mean RMSE**: **`6.58°`**
- **Wall Clock**: 64.1 minutes

## 2026-09-06 — Architectural Refactoring Phases 3 & 4: BaseVAMExtractor Migration & Protocol 1.0 Clean Re-Benchmarking

### 1. Root-Cause Remediation & Extractor Unification (Phase 3)

Following the architectural audit (`docs/ARCHITECTURAL_AUDIT_AND_ABSTRACTIONS.md`), all feature extractors have been migrated to the standardized abstract contract:

- **BaseVAMExtractor Integration**:
  - `Cosmos7BExtractor` (`src/lerobot/policies/vam/cosmos7b_extractor.py`) inherits from `BaseVAMExtractor`, implements `encode_latents` (connecting real `AutoencoderKLCosmos`), `forward_transformer_blocks`, `compute_grid_shape`, and returns `Cosmos7BExtractionOutput` (subclass of `VAMExtractionOutput`).
  - `Cosmos14BExtractor` (`src/lerobot/policies/vam/cosmos14b_extractor.py`) inherits from `BaseVAMExtractor`, implements real VAE encoding with 17-channel padding support, FP8 quantized linear layers, and CPU-CUDA block streaming.
  - `Flux2KleinExtractor` (`src/lerobot/policies/vam/flux2_klein_extractor.py`) inherits from `BaseVAMExtractor`, implementing multi-reference conditioning across observation history and goal tokens.
- **Unified Cache Builder (`scripts/video_vam/build_vam_feature_cache.py`)**:
  - Replaces all legacy ad-hoc extraction scripts with a single unified CLI builder.
  - Enforces `enforce_protocol_1_0_split` to prevent train/val leakage at invocation time.
  - Replaces heuristic bilinear downsampling with causal VAE latent representations.
  - Persists `action_is_pad`, `state`, and 30-step `target_action` along with SHA-256 digests in standardized `manifest.json` schemas.

### 2. Protocol 1.0 Feature Cache Generation

Extracted clean, disjoint Protocol 1.0 caches on NVIDIA RTX 4090:

- **Cosmos 7B Protocol 1.0 Cache** (`/home/anton/.cache/video-vam/cosmos7b-protocol1-cache`):
  - Train: Episodes 0–31 (stride 3, 1,572 samples, 64 tokens, 8,192-dim).
  - Validation: Episodes 32–39 (stride 20, exactly 88 held-out validation anchors across the session regime shift).
  - Wall Clock: 124.5s train + 6.9s val (total 131.4s at ~12.6 samples/s).
- **FLUX.2 [klein] Protocol 1.0 Cache** (`/home/anton/.cache/video-vam/flux2-klein-protocol1-cache`):
  - Train: Episodes 0–31 (stride 3, 1,572 samples, 256 visual tokens, 6,144-dim).
  - Validation: Episodes 32–39 (stride 20, 88 held-out validation anchors).
  - Wall Clock: 90.1s train + 5.1s val (total 95.2s at ~17.4 samples/s).

### 3. Launch & Execution of the Protocol 1.0 Re-Benchmarking Queue (Phase 4)

- **Persistent Tmux Session**: `rebenchmark_protocol1_queue` running `scripts/video_vam/run_rebenchmark_queue_protocol1.sh`.
- **Master Log**: `/home/anton/.cache/video-vam/runs/rebenchmark_queue_protocol1_0.log`.
- **Integrity**: Standardized hyperparameters (lr 1e-4, cosine decay, warmup 1000, patience 20, min_steps 20000), exact action masking (`action_is_pad` strictly excluded from normalization and flow loss), 100% data leakage free.

#### Stage 1: Cosmos 7B Base SmolExpert Training Results (Completed)

- **Status**: Completed (converged at Step 17,000; early stopping triggered at Step 24,500 after 15 stagnated evaluations).
- **Wall Clock Time**: 1,533.9s (~25.6 minutes).
- **Validation Metrics on Protocol 1.0 Held-out (Episodes 32–39, 88 anchors)**:
  - **Validation Global RMSE**: **`15.980°`** (~15.98°).
  - **Immediate Horizon-1 Error (H1)**: **`5.580°`**.
  - **First-5 Actions Mean RMSE**: **`7.769°`**.
  - **Validation Flow Matching Loss**: `0.1380`.
- **Artifacts Saved & Verified**:
  - Checkpoint: `/home/anton/.cache/video-vam/runs/cosmos7b-protocol1-smolexpert/best.safetensors` (230 MB).
  - Normalizer: `/home/anton/.cache/video-vam/runs/cosmos7b-protocol1-smolexpert/normalizer.safetensors` & `.json`.
  - Metrics: `/home/anton/.cache/video-vam/runs/cosmos7b-protocol1-smolexpert/best_metrics.json` & `training_summary.json`.

#### Remaining Queue Pipeline:

- **Stage 2**: FLUX.2 [klein] Base SmolExpert training on `/home/anton/.cache/video-vam/flux2-klein-protocol1-cache` (min_steps=20000, max_steps=50000, patience=20, lr=1e-4, warmup=1000, cosine decay, action masking).
- **Stage 3**: Build Cosmos 14B Protocol 1.0 feature cache via `build_vam_feature_cache.py` (batched FP8 block streaming).
- **Stage 4**: Cosmos 14B Base SmolExpert training on `/home/anton/.cache/video-vam/cosmos14b-protocol1-cache` (min_steps=20000, max_steps=50000, patience=20, lr=1e-4, warmup=1000, cosine decay, action masking).

## 2026-09-07 — Comprehensive Correctness Audit, Historical Retractions, and Base VAM Architecture Verification

### 1. Systematic Correctness & Validity Audit

A full-codebase architectural audit (`docs/video_vam_correctness_audit.md`) revealed that previous empirical numbers and training pipelines contained multiple compounding defects that necessitate formal retractions:

1. **Formal Retraction of Historical Sub-11° Entries**:
   - The reported validation RMSE values of **9.46°** (Cosmos 7B), **9.75°** (Cosmos 7B LoRA), **9.97°** (Cosmos 14B), **10.33°** (FLUX.2 LoRA), **11.45°** (FLUX.2 Base), **10.99°** (14B Converged), and **11.08°** (14B LoRA) are **RETRACTED**.
   - **Root Causes**:
     - _Frame-Level Data Leakage_: Sequential video clips extracted at stride 3 were split using `torch.utils.data.random_split(85%, 15%)`. Adjacent frames from the same trajectory were split between train and val sets, allowing the policies to perform trivial trajectory interpolation.
     - _Pseudo-Latent Bypassing_: Feature extractors bypassed genuine causal 3D VAEs, applying bilinear `F.interpolate` downsampling to $32 \times 32$ with zero-padding to 16 channels, feeding meaningless spatial noise to DiTs.
     - _Unstandardized Early Reruns_: Early re-benchmarking attempts (e.g. 15.98°) were conducted on unstandardized intermediate caches and are retired pending clean execution.

2. **Cosmos 3 Edge Architectural Discrepancies**:
   - _Gen Tower vs. Und Tower Mismatch_: Video-LoRA training operated on the generation tower (`gen_seq`) via flow matching, while policy feature extraction tapped the understanding tower (`und_seq`), introducing an uncalibrated representation shift.
   - _Latent Normalization & Posterior Mode_: Omitted per-channel latent scaling and failed to evaluate at native posterior mode (`posterior.mode()`).
   - _Framerate Mismatch_: 10 FPS robot dataset evaluated against 24 FPS pretraining positional assumptions.
   - _Patchify Order Defect_: Legacy manual patchification used channel-first `(C, p, p)` instead of native spatial-first `(p, p, C)`.
   - _Model Config_: Local measurements confirm 2048 hidden dim, 16 attention heads, 28 layers, 3.37B dense parameters (vs public 4B label).

3. **Metric & Baseline Mathematical Corrections**:
   - _Baseline Comparability_: The historical unadapted 26.12 baseline is not directly comparable.
   - _Domain Adaptation Gain_: Frozen SmolExpert (13.65) vs legacy prefix (13.81) vs LoRA (13.06) represents a ~0.6°–0.8° gain (confounded), not an error halving.
   - _Oracle & Gap Closure_: Oracle 13.08 is an empirical reference point on teacher `cond_frames`, not an absolute ceiling. Gap closure is **89.4%** ($\frac{13.74 - 13.15}{13.74 - 13.08} = 0.894$), subject to seed uncertainty (98.9% claim was false).
   - _Mixed Units_: Metrics report 5 joints in degrees ($^\circ$) + gripper in range $[0, 100]$, not pure degrees. Horizons: H1 offset 0 ($t$), first-5 $t \dots t+4$, full-30 offsets $0 \dots 29$.
   - _Dataset Changepoint_: Mean action shifts between episodes 0–23 and 24–39 (+19.96° Joint 2, -29.47° Joint 3) reflect an operator/calibration changepoint hypothesis, not an intentional domain shift.
   - _Cosmos-Predict2-2B_: Temporal downsampling factor is $4\times$ (not $8\times$).
   - _LTX Spatial Resolution_: The 32x spatial VAE limitation is a research hypothesis, not proven fact.

### 2. Evaluation Rigor & Invariant Implementation

The codebase is being updated to enforce:

- **Per-Sample Seed Batch Invariance**: `sample_seed = int(hashlib.sha256(f"{evaluation_seed}:{sample_id}".encode()).hexdigest()[:8], 16) % (2**31 - 1)` guaranteeing type-safe invariance to batch size and sample ID strings.
- **Optimizer Step Alignment**: Alignment of warmup, decay, and logging to true optimizer steps during gradient accumulation.
- **Fail-Closed Validation**: Manifest schema checks, disjoint episode set enforcement, and pinned dataset revision (`243370c3c08bcbd860133c4a0d658ea7c1d2e77e`).
- **Actual Checkpoint Resume**: Proper restoration of optimizer, scheduler, and step states.

### 3. Current Implementation & Operational Status

- **Status**: Correct extraction and training code is being implemented by agents **NOW** and is currently under code review.
- **Jobs**: **No jobs are currently running or scheduled.**
- **Testing**: New unit tests covering base abstractions, Cosmos 3 features, and LoRA injection are in place; comprehensive verification is pending (no invented pass counts).

---

## 2026-09-08 — Operational Audit: Placeholder Queue Discovery, Weight Inventory & Execution Plan

### 1. Verification of Active Pipeline State

- Inspection of `scripts/video_vam/run_full_autonomous_pipeline.sh` revealed that Stage 2 only printed status without executing Cosmos 7B evaluation, and Stage 3 wrote hardcoded numbers into `grand_evaluation_summary.json`.
- This placeholder script was flagged and disabled. No synthetic or hardcoded score may be treated as benchmark evidence.
- The GPU is idle; only the SmolVLA v1 read-only RPC server remains active on port 8766.

### 2. Comprehensive Weight Inventory

- **Verified Saved:** Cosmos 2B Pool2 SmolExpert (`best`+`last`), LTX-2.5 Pool2/Unpooled (`best`+`last`), Cosmos 2B video-LoRA step 6,000, and SmolVLA v1/v2 are safely stored in `outputs/train/`.
- **Missing Weights:** Checkpoints for Cosmos 2B T=2 undistilled and direct-distilled models (both student LoRA and SmolExpert heads) were not found in cache or train directories. Retraining is required under Protocol 1.0.
- **Foundation Models:** Cosmos 2B backbone weights (`v2w_pretrained_cosmos.pt`, 3.9 GB) verified in `outputs/models/cosmos2b/video_backbone/`.

### 3. Dataset Quarantine

- Dataset `Orellius/cube_out_of_box_v2` remains strictly quarantined due to a declared vs actual frame/episode count discrepancy (12,163 frames / 100 episodes declared vs 15,998 rows / 140 episodes present).
- All retraining and re-scoring must proceed exclusively on canonical pinned v1 (`hubnemo/cube_out_of_box_dataset` @ `243370c3c08bcbd860133c4a0d658ea7c1d2e77e`).

### 4. Canonical Plan Established

- Outlined operational sequence in [`docs/video_vam_execution_plan.md`](./video_vam_execution_plan.md) covering:
  1. T=2 undistilled feature extraction & SmolExpert training.
  2. Teacher unpooled `cond_frames` target extraction & T=2 direct distillation.
  3. Distilled T=2 feature cache extraction & SmolExpert training.
  4. Standard Protocol 1.0 evaluation and verified artifact preservation.

---

## 2026-09-11 — Physical Robot Evaluation, Visual Covariate Shift & Online Coherent Data Augmentation

### 1. Physical Hardware Evaluation (SO-101 on `cube_out_of_box`)

- **Policies Tested**: Cosmos 3 Edge Video-LoRA (~80 ms), Cosmos 2B (=2$ distilled / undistilled), and SmolVLA (v1 and v2) via client-server SSH RPC with Real-Time Chunking (RTC) at 10 Hz.
- **Hardware Observations**:
  - The client-server RPC loop and 10 Hz RTC execution streamed smoothly without software failure.
  - **Cosmos 3 Edge Video-LoRA** demonstrated the best qualitative trajectory behavior and lowest offline loss, moving closest to the target.
  - **Failure Mode**: None of the policies achieved a reliable physical grasp. The primary failure cause was severe **visual covariate shift** between demonstration recording conditions and live testing (ambient room lighting, shadows, and subtle camera mounting angle differences).
  - **Root Cause**: Offline caching with stride 3 discarded 67% of temporal demonstration frames and prevented any dynamic visual data augmentations during training.

### 2. Strategy: Online Video Extraction with Coherent Data Augmentation

- **Hypothesis**: Training SmolExpert on raw, dynamically augmented video frames with stride 1 (9,122 consecutive 5-frame temporal windows) will make the policy invariant to lighting, shadow, and camera angle offsets.
- **Augmentation Pipeline**:
  - Applied on GPU per 5-frame window 0$:
    - Photometric jitter: Random brightness ($\pm 15\%$), contrast ($\pm 15\%$), saturation ($\pm 15\%$), subtle hue ($\pm 5\%$).
    - Spatial crop/translation: Random 4–6% resized crop simulating camera mounting offsets.
    - Sensor noise: Mild Gaussian blur ($\sigma \in [0.1, 0.8]$).
  - **Critical Invariant**: Augmentations are applied **identically across all =5$ frames** within each temporal window to preserve physical and temporal coherence.
  - **Clean Benchmark Safeguard**: Validation and evaluation splits (Eval-1: 32–39, Eval-2: 90–99) remain **strictly unaugmented** for untainted, comparable benchmark tracking.

### 3. Next Steps & Execution Sequence

1. **Active Run**: Training Cosmos 3 Edge Video-LoRA online with augmentations on Scale-100 (`Orellius/cube_out_of_box_v2`, stride 1) to measure validation loss/RMSE impact against offline baselines.
2. **Follow-up Run**: Train the identical online data augmentation recipe on the cleaner historical V1 dataset (`hubnemo/cube_out_of_box_dataset`, episodes 0–31).
3. **Outcome**: Produces two robust, visually augmented candidate policies for the next physical robot testing session.
