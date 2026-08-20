# Native World2Action decoder

This document records the approved native mimic-video action decoder path for
SO-101. The pinned upstream source is mimic-video commit
`e3355dbc93132b576c02f920a59b4fc18a4f5906`. The LeRobot wrapper is
`lerobot.policies.vam.world2action.World2ActionDecoder`.

## Contract and capacity

| Tensor                | Shape          | Dtype/runtime rule                                     |
| --------------------- | -------------- | ------------------------------------------------------ |
| state                 | `[B, 1, 6]`    | float16/bfloat16/float32; one observed SO-101 state    |
| action target/sample  | `[B, 30, 6]`   | same action dtype as state                             |
| frozen Cosmos context | `[B, N, 2048]` | bfloat16, detached at the wrapper boundary             |
| context timestep      | `[B, 1]`       | finite floating tensor, default video sigma `10`       |
| decoder time          | `[B, 31]`      | action time expanded over the state + 30 action tokens |

The native decoder uses `max_horizon=31`, `in_channels=6`, and
`out_channels=6`. Its approved full configuration is:

- `model_channels=1024`, `num_blocks=24`, `num_heads=8`, `mlp_ratio=4`;
- `use_adaln_lora=True`, `adaln_lora_dim=128`,
  `pair_timestep_feature_rank=1024`;
- Cosmos cross-attention layer `20`, context width `2048`, and video sigma `10`;
- PyTorch 2.11 `scaled_dot_product_attention` through the upstream
  `torch_attention_op`, with no projection adapter;
- no selective activation checkpoint (`SACConfig(mode="none")`).

The normalizer is a frozen artifact, not a trainable projection or adapter.
`parameter_count_report()` reports decoder parameters separately and reports
zero normalizer parameters.

## Objective and sampling

For train-set-normalized action `x0`, sample `epsilon ~ N(0, I)` and a uniform
training time from the registered `Beta(alpha=1.0, beta=1.0)` scheduler. The
official bounded implementation returns `t` in `[0.001, 1.0]`. Training uses:

```text
xt       = (1 - t) * x0 + t * epsilon
u_target = epsilon - x0
model_in = xt / sqrt((1 - t)^2 + t^2)
loss     = MSE(float32(model(model_in)), float32(u_target))
```

Observation dropout is `0.2` in training and `0.0` in inference. The wrapper
accepts `t` and `epsilon` for deterministic unit tests. The context tensor is
explicitly detached before the denoiser call; gradients may train the decoder
and its context LayerNorm, but never the frozen video context.

Inference starts from a seeded Gaussian at `t=1`. It applies the same
pre-scaling before every denoiser call and invokes the official Euler
`BetaScheduler.step` exactly ten times with `dt=-0.1`, ending at `t=0`.

## Normalization artifact

`ActionStateNormalizer.from_training_tensors` computes per-joint min/max from
state/action tensors from the `train` split only. Validation/test provenance is
rejected. Each range must be non-degenerate. Statistics are stored as four
float32 tensors in a safetensors file, with a strict JSON sidecar containing
shape, schema, source split, and clamp provenance; no pickle is used.

Normalization maps each train range to `[-1, 1]` and clamps out-of-range values.
Denormalization clamps normalized values first, so policy outputs saturate at
train-set joint limits rather than extrapolating beyond them.

## Vendor closure and adaptations

Only the upstream `world2action_dit.py` and `beta_scheduler.py` are added for
this decoder. The existing vendored Cosmos `SACConfig` and `_compat.py` seam
supply the minimal compatibility surface; Hydra, Megatron, imaginaire, and
pipeline framework dependencies are not pulled in.

Upstream imports Transformer Engine eagerly. The vendored decoder loads TE only
at model construction and raises a precise dependency error if TE RMSNorm and
its fused rotary implementation are unavailable. The loader imports
`DotProductAttention` from `transformer_engine.pytorch.attention`, and first
tries the current `transformer_engine.pytorch.attention.rope` location for
`apply_rotary_pos_emb`, falling back to the legacy attention location. An API
mismatch is reported separately from TE being absent. This is import-only
compatibility: rotary numerics and arguments are unchanged.

The approved torch SDPA helper returns `[B, S, H, D]`, matching the common
Attention wrapper contract; the wrapper alone performs the final head flatten
to `[B, S, query_dim]`. This is an interface correction only: SDPA scaling,
architecture, and weights are unchanged.

The native denoiser internally processes one state token plus 30 action tokens
and returns `[B, 31, 6]`. The wrapper slices the first `HO=1` state token exactly
and exposes `[B, 30, 6]`; that state-token prediction participates internally
but is excluded from action loss and policy output. Explicitly injected denoisers
may return the already-sliced `[B, 30, 6]` shape, while every other token length
is rejected. This is required
for official numerics even when attention uses torch SDPA. The torch backend is
an approved
runtime implementation adaptation: capacity, architecture, and weights remain
unchanged, but latency differs from the upstream `flash_attn_no_cp` option and
must be reported separately in fairness comparisons.

Other explicit scientific adaptations are: SO-101 six-dimensional state/action
instead of upstream padded 10D actions; horizon 31 instead of the upstream
16/61 action/state horizon choices; generic frozen Cosmos first instead of a
Bridge-first backbone; and explicit hidden context/context-timestep inputs.
These are interface/runtime adaptations, not decoder-capacity reductions.

## Scratch initialization and parameter count

No compatible pretrained 6D SO-101 action checkpoint exists. When
`action_checkpoint_path` is omitted, the wrapper intentionally constructs the
full 24-block native DiT with the official constructor and `init_weights`
path. It does not reset global RNG state: the LeRobot trainer must establish
its seed before constructing the policy, and that seed must be recorded in
later training provenance. Cosmos remains frozen while the full action decoder
is trained from scratch for the 6D overfit.

With or without a checkpoint, report the split count without running a training
job:

```bash
cd /home/anton/lerobot-video-vam
PYTHONPATH=src .venv/bin/python - <<PY
from lerobot.policies.vam.world2action import World2ActionConfig, World2ActionDecoder
model = World2ActionDecoder(World2ActionConfig())
print(model.parameter_count_report())
PY
```

If `action_checkpoint_path` is supplied, strict key/shape loading is retained;
upstream 10D checkpoints are never silently accepted. CPU tests inject a tiny
fake denoiser and never instantiate the 24-block model, load TE, access the
network, or execute GPU-heavy code.

## Diagnostic reopen and cache evaluation

The 1,000-step artifact is diagnostic-only and must not be treated as rollout
or generalization evidence. It reconstructs the training cache, using every
manifest entry, fixed `t=0.5` flow inputs, and a seeded ten-step action sample;
physical MSE is masked by `action_is_pad` and reported in the original action
units. Reopen validates the safetensors/JSON pair, exact state keys/shapes/
dtypes, full parameter count, manifest hash/path, normalizer path, step, and
non-rollout metadata before moving the decoder to its configured runtime.

Parent command for the exact checkpoint:

```bash
cd /home/anton/lerobot-video-vam
source scripts/video_vam/cosmos_cuda_env.sh
scripts/video_vam/run_evaluate_cosmos_world2action_cache.sh \
  --checkpoint /home/anton/.cache/video-vam/runs/cosmos-overfit-even8-s0/checkpoint-step-001000.safetensors \
  --manifest /home/anton/.cache/video-vam/cosmos-overfit-ep0-64/manifest-even8.json \
  --normalizer /home/anton/.cache/video-vam/runs/cosmos-overfit-even8-s0/normalizer.safetensors \
  --output-json /home/anton/.cache/video-vam/runs/cosmos-overfit-even8-s0/evaluation-step-001000.json \
  --expected-step 1000 --seed 424242
```

The evaluator requires `--overwrite` to replace an existing JSON result and
records runtime, VRAM, software, GPU, checkpoint, manifest, normalizer, and
evaluation-seed provenance. It never marks the decoder rollout-ready.

## Real train/validation protocol

The diagnostic overfit launcher is intentionally unchanged. For experiment
runs, first create one cache manifest containing the desired episodes, then
persist the episode-level split and fixed validation probe:

```bash
source scripts/video_vam/cosmos_cuda_env.sh
$VAM_VENV/bin/python scripts/video_vam/create_vam_split.py \
  --manifest /home/anton/.cache/video-vam/cosmos-features/manifest.json \
  --output /home/anton/.cache/video-vam/cosmos-features/split.json \
  --probe-seed 424242
scripts/video_vam/run_train_cosmos_world2action.sh \
  --manifest /home/anton/.cache/video-vam/cosmos-features/manifest.json \
  --split /home/anton/.cache/video-vam/cosmos-features/split.json \
  --output-dir /home/anton/.cache/video-vam/runs/cosmos-real \
  --max-steps 500000 --val-every 1000 --patience 10
```

The default split is train episodes `0..31` and validation episodes `32..39`,
but both lists and the split name are configurable in the split-generation
CLI. A manifest episode not covered by the split is an error. The split JSON
also contains one persisted `tau_a`, 30x6 action-noise draw, and action-sampler
seed for every validation window. Validation never samples new flow inputs;
this fixed probe is what makes validation flow loss literally comparable across
Cosmos 2B, Bridge, quantized, Cosmos 14B, and LTX backbones. Padded action
positions are excluded from both validation metrics.

The trainer computes normalization only from train entries, supports a
microbatch plus `--grad-accum-steps`, validates with fixed-probe flow loss and
physical per-joint/aggregate RMSE, and stops on the primary validation loss.
`last.safetensors` includes optimizer, scheduler, and RNG resume state;
`best.safetensors` is the best fixed-probe checkpoint. Both have strict JSON
metadata and explicitly report `rollout_readiness: false`.

The upstream defaults adopted by the real trainer are lr `1e-4`, fused AdamW
with weight decay `0.1`, betas `(0.9, 0.99)`, eps `1e-8`, cross-attention layer
20, `loss_reduce="mean"`, `loss_scale=10`, and LambdaLinear scheduler
`warm_up_steps=1000`, `f_max=1`, `f_min=0.2`. Its cycle length is scaled to
`--max-steps` instead of copying upstream's 500000-step cycle. The native
PyTorch fused AdamW path has no separate `master_weights` option; trainable
parameters remain FP32, and any fused fallback is recorded in metadata.

### Cache integrity verification and data loading

`CosmosFeatureCacheDataset` verifies both the manifest file hashes and the
safetensors tensor hashes on the first read of each artifact in a dataset
process. The verified sample ID is then remembered, so later epochs still
validate safetensors structure and provenance but do not rescan the 78 MB
context with SHA-256 or repeat finite-value scans. This is safe for the
immutable cache artifacts produced by the builder: any artifact corrupted
before its first read fails before entering training. For paranoid operation,
use `CosmosFeatureCacheDataset(..., verify_every_access=True)` to perform the
full hash checks on every access.

The trainer can overlap cache reads with compute using worker prefetching:

```bash
scripts/video_vam/run_train_cosmos_world2action.sh \
  --manifest /path/to/manifest.json \
  --split /path/to/split.json \
  --output-dir /path/to/run \
  --num-workers 4 --prefetch-factor 2
```

`--pin-memory` is optional and should be benchmarked on the target machine;
it is not enabled by default.

### Weights & Biases study logging

The trainer logs to the `video-vam-world2action` project by default while
retaining JSONL as the authoritative local record. Disable remote logging with
`--no-wandb`; missing credentials, unavailable W&B, offline mode, and failed
uploads are reported once and never stop training. Resumed runs reuse the
`wandb_run_id.txt` saved in the output directory.

Each run config records backbone identity/checkpoint/hash and context shape,
adapter usage, manifest and cache stride, split identity/hash/episodes/probe
seed, dataset revision, optimizer/scheduler settings, batch size, and gradient
accumulation. Metric names are shared across arms: `train_loss`,
`val_fixed_flow_loss`, `val_aggregate_rmse`, `val_rmse_joint_0` through
`val_rmse_joint_5`, `learning_rate`, `samples_per_sec`, `peak_vram_bytes`, and
`wall_clock_seconds`. Use `--wandb-log-horizon-rmse` for 30 additional
per-horizon RMSE scalars; this adds only a small reduction over already sampled
validation actions and is disabled by default.

### Online extraction mode

Cached training uses the persisted high-noise context and passes its fixed
`video_sigma=10` explicitly to the decoder. For an upstream-faithful
randomization study, use `--context-mode online` with a local LeRobot dataset,
Cosmos tokenizer, prompt embedding, and backbone checkpoint:

```bash
scripts/video_vam/run_train_cosmos_world2action.sh \
  --manifest /path/to/manifest.json --split /path/to/split.json \
  --output-dir /path/to/online-run --context-mode online \
  --dataset-root /path/to/lerobot-dataset \
  --tokenizer /path/to/tokenizer --prompt /path/to/prompt.safetensors \
  --backbone-checkpoint /path/to/video-backbone.pt \
  --batch-size 1 --grad-accum-steps 2
```

Online training runs the frozen extractor under `no_grad` and eval mode for
each microbatch. It samples `sigma=exp(N(0,1))*sqrt(16)` (the upstream
`adjust_video_noise=True`, `state_t=16` contract), replaces 5% of video
samples with log-uniform sigma in `[200, 100000]`, and redraws latent Gaussian
noise. The drawn sigma is passed as decoder context conditioning. Validation
continues to use the persisted fixed probe and `video_sigma=10`, so its flow
loss and action RMSE remain comparable with cached and control runs.
