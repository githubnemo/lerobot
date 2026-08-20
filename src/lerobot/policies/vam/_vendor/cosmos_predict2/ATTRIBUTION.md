# Vendored Cosmos-Predict2 source

This directory contains a minimal source closure copied from the Apache-2.0
licensed [mimic-video](https://github.com/mimic-video/mimic-video) repository at
commit `e3355dbc93132b576c02f920a59b4fc18a4f5906` (25 June 2026). The upstream
repository builds on NVIDIA Cosmos-Predict2. Copied files retain their upstream
SPDX headers. `LICENSE` and `ATTRIBUTIONS.md` are copied from upstream
`model/LICENSE` and `model/ATTRIBUTIONS.md`.

Copied paths, upstream hashes, vendored hashes, and modifications are recorded
in `source_manifest.json`.

Intentional narrow modifications:

- Imports are rewritten to package-local relative imports and `_compat.py`, so
  global `cosmos_predict2`/`imaginaire` packages are not installed or imported.
- The official `CosmosTextEncoderConfig` shape constants and lightweight logging,
  checkpoint I/O, and non-initializing distributed helpers are provided by
  `_compat.py`. T5 is never loaded by this extractor.
- `module/a2a_cp.py` uses its local `module/attention.py` import.
- `models/text2image_dit.py` keeps the upstream `minimal_a2a` and
  `transformer_engine` branches. Transformer Engine is imported lazily during official model construction for
  normalization and fused rotary kernels; minimal_a2a remains the selected
  attention backend and uses the copied upstream attention implementation.
- The tokenizer constructor accepts explicit device, dtype, and temporal-window
  settings so the LeRobot wrapper can own runtime placement.

No model weights, tokenizer checkpoint, T5 files, guardrails, Hydra configs,
training code, or distributed initialization are vendored.

World2Action additions:

- `models/world2action_dit.py` and `schedulers/beta_scheduler.py` are copied from
  the same pinned mimic-video commit above. The decoder keeps official
  Transformer Engine RMSNorm numerics, but loads TE lazily; its approved
  LeRobot runtime selects the upstream `torch_attention_op` (PyTorch SDPA).
- `networks/selective_activation_checkpoint.py` and `_compat.py` are reused as
  the existing minimal closure. The approved decoder passes `SACConfig(mode="none")`;
  no selective activation checkpoint or Megatron/Hydra/pipeline framework is
  included.

- Both vendored DiT lazy TE loaders use the current `attention.rope.apply_rotary_pos_emb` location with a legacy `attention` fallback; `DotProductAttention` remains imported from `attention`. This is import-only compatibility and does not alter rotary numerics or arguments.

Checkpoint loading compatibility:

- The pinned official generic checkpoint was observed to contain 28
  `blocks.<0..27>.cross_attn.attn_op._extra_state` Transformer Engine metadata
  entries that are not registered by this vendored/runtime module.
- The pinned upstream `model/cosmos_predict2/pipelines/video2world.py` loads
  the DiT state dict with `strict=False` after removing the `net.` prefix. The
  LeRobot wrapper is intentionally stricter for this kind of inert TE metadata:
  it allows only that exact key pattern, only with no missing keys, and only for
  the explicitly opted-in official generic checkpoint path. Values must be
  one-dimensional `torch.uint8` tensors; all other unexpected or malformed keys
  fail loading. Ignored keys and their count are recorded in provenance.
- `_compat.py` exposes the upstream logger's `success` method through a local
  adapter that delegates to stdlib `Logger.info`; no global logging monkeypatch
  is applied, and the standard debug/info/warning/error methods remain unchanged.

- The approved World2Action torch SDPA helper preserves the common backend output shape `[B, S, H, D]`; only the wrapper flattens heads. This fixes an upstream interface mismatch without changing attention math.

- The LeRobot wrapper handles the native state-token output contract outside the copied model: native `[B,31,6]` is sliced to action-only `[B,30,6]`, while explicitly injected already-sliced outputs remain supported.
