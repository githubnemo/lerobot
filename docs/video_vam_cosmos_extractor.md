# Frozen Cosmos-Predict2 extractor foundation

This is the first LeRobot-only foundation for the frozen Video2World
representation. It uses the official mimic-video source closure at commit
`e3355dbc93132b576c02f920a59b4fc18a4f5906` and does not download weights,
load T5, train, or initialize distributed systems during package import.

## Frozen contract

- The pinned dataset's native RGB history is TCHW `[5, 3, 480, 640]`.
- The harness reorders that history to the extractor's BCTHW input
  `[B, 3, 5, 480, 640]`; `resize_online=False` means no online resize occurs.
- Prompt input: the strict loader validates the cached official T5 embedding
  `[B, 512, 1024]`; prompt strings and T5 loading remain outside the extractor.
- Tokenizer output: two conditional latent frames `[B, 16, 2, 60, 80]`, padded
  with zeros to the 16-frame backbone state.
- First representation: one high-noise forward at `sigma=10.0`,
  `stop_after_step=0`, hidden layer `20`, full token grid (no pooling).
- Expected hidden grid: `[B, 16, 30, 40, 2048]`.
- Flattened output: `[B, 19200, 2048]`; returned `sigma` is `[B]`.
- Backbone parameters and tokenizer parameters are frozen; extraction runs
  under `torch.no_grad()`.
- The official CUDA BF16/FP16 forward uses `torch.autocast` around the Cosmos
  backbone only. Timestep sinusoidal embeddings remain FP32 and are converted
  by the backbone's autocast-aware linear layers; CPU and float32 runs disable
  autocast. Extractor provenance records the selected device and dtype.
- The selected official 2B backend is `minimal_a2a`, which dispatches through
  upstream `module/attention.py` (PyTorch SDPA on the RTX 4090 target). This is
  not a fallback; Transformer Engine is required lazily for the official normalization and fused rotary kernels; it is not the selected attention backend.

## Source and checkpoint provenance

The vendored source attribution and exact copied upstream paths are in
`src/lerobot/policies/vam/_vendor/cosmos_predict2/ATTRIBUTION.md` and the
machine-readable `source_manifest.json`. The checkpoint helper pins
`jonpai/mimic-video` to revision
`f28339034831e3c2374be075e622e1ff38ebe0f8` and only constructs these default
patterns:

- `video_backbone/v2w_pretrained_cosmos.pt`
- `video_backbone/tokenizer/*`

The released Bridge LoRA is a separately named optional artifact
`video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt`.
The downloader defaults exclude the large T5 model because prompt embeddings
are cached separately using the official T5 preprocessing path. The current
setup explicitly retains its downloaded T5 files alongside the generated
prompt artifact; the smoke harness does not load or remove the large model.

## Checkpoint commands (dry-run by default)

From the repository root, after the Python 3.12 environment has `huggingface_hub`:

```bash
python src/lerobot/policies/vam/download_cosmos_checkpoints.py --include-bridge-lora --include-t5
python src/lerobot/policies/vam/download_cosmos_checkpoints.py --include-bridge-lora --include-t5 --execute
```

The first command only prints the pinned plan and disk estimate. The second is
the explicit network mutation. Keep the downloaded T5 files when generating
or reviewing the prompt artifact; no deletion step is part of this setup.

## Placeholder smoke command

After the CUDA runtime and checkpoints are provisioned, replace the two paths
and run this exact shape smoke command:

```bash
uv run python - <<'PY'
import torch
from lerobot.policies.vam.cosmos_predict2_extractor import CosmosPredict2Extractor, CosmosPredict2ExtractorConfig

extractor = CosmosPredict2Extractor(CosmosPredict2ExtractorConfig(
    checkpoint_path="/checkpoints/video_backbone/v2w_pretrained_cosmos.pt",
    tokenizer_path="/checkpoints/video_backbone/tokenizer/tokenizer.pth",
    device="cuda",
    dtype="bfloat16",
))
out = extractor.extract(torch.zeros(1, 3, 5, 480, 640, device="cuda", dtype=torch.uint8), torch.zeros(1, 512, 1024, device="cuda", dtype=torch.bfloat16))
print(out.hidden_grid.shape, out.tokens.shape, out.sigma.shape, out.provenance)
PY
```

Expected shapes are `torch.Size([1, 16, 30, 40, 2048])`,
`torch.Size([1, 19200, 2048])`, and `torch.Size([1])`. After the reviewed CUDA
runtime dependencies and pinned local weights are provisioned, this is the exact
shape check. Do not generate T5 embeddings with this wrapper; use the official
T5 encoder preprocessing and pass the retained cached tensor.

## First-real-forward smoke harness

The LeRobot-owned smoke CLI loads one local sample from the pinned cube dataset,
validates metadata and the expanded sample contract, calls the frozen extractor
with RGB history plus the already validated prompt embedding, verifies the
hidden output, and round-trips one detached cache artifact:

```bash
cd /home/anton/lerobot-video-vam
scripts/video_vam/run_smoke_test_cosmos_extractor.sh
```

The launcher sources `scripts/video_vam/cosmos_cuda_env.sh`, which derives
`VAM_REPO_ROOT` and `VAM_VENV`, validates the NVIDIA wheel library directories,
preserves any existing `LD_LIBRARY_PATH`, and exports the CUDA/Transformer
Engine include and home variables required by the installed CUDA 12 wheels.
Pass smoke CLI options directly to the launcher, for example
`run_smoke_test_cosmos_extractor.sh --frame-index 14`.

The default sample is episode `0` at absolute frame `4`, so its causal window is
`[0, 1, 2, 3, 4]` and does not contain padded RGB history. Use
`--frame-index N` or `--window-start N` to select another five-frame window;
`--overwrite` is required to replace an existing cache. The CLI never downloads
videos and never passes state or target actions to the extractor. State is stored
as action-decoder conditioning, while `target_action` is stored only as a
label.

The expected report includes separate prompt/model load and extraction latency,
load/extraction peak allocated and reserved VRAM, raw hidden shape, flattened
context shape/dtype, and cache/provenance paths. With the pinned mimic-video
480p contract, the expected shapes are raw `[1, 16, 30, 40, 2048]` and flattened
`[1, 19200, 2048]`; the extractor uses BF16 CUDA, generic checkpoint only,
`high_noise_sigma=10`, seed `0`, hidden layer `20`, and `stop_after_step=0`.

The pinned mimic-video config inspected at commit
`e3355dbc93132b576c02f920a59b4fc18a4f5906` is resolution `480`,
`resize_online=False`, with official `frame_replace` conditioning. The source
camera is therefore natively 480x640, and the harness uses BCTHW
`[B, 3, 5, 480, 640]` without online resizing. `704x1280` was an erroneous
generic-Cosmos assumption and is deliberately not used here; it is not a
runtime blocker for this approved camera/checkpoint contract.

In the vendored DiT, `max_img_h=240` and `max_img_w=240` feed the positional
embedding lengths after division by the spatial patch size. Cache provenance
records these as `official_positional_latent_max_h/w`: latent spatial maxima
before patching, not pixel dimensions.

No cube training entrypoint in this checkout currently constructs the VAM
policy or Transformer Engine modules, so no training script is changed. When a
VAM policy training CLI is wired, source the helper immediately before that
entrypoint, for example:

```bash
source /home/anton/lerobot-video-vam/scripts/video_vam/cosmos_cuda_env.sh
exec "${VAM_VENV}/bin/python" path/to/vam_training_cli.py "$@"
```

The cache seed is fixed for this first real sample, so rerunning the same
episode/window with seed `0` has fixed noise semantics across action epochs.
The later cache builder should derive a deterministic seed from episode/window
identity when it materializes a larger cache.

The pinned upstream `model/cosmos_predict2/pipelines/video2world.py` uses
`load_state_dict(..., strict=False)` after removing the `net.` prefix. The
LeRobot wrapper keeps strict parameter/buffer matching and narrows that upstream
exception to the known inert metadata below.

The official generic checkpoint contains 28 inert Transformer Engine
`_extra_state` entries under `blocks.0` through `blocks.27` for
`cross_attn.attn_op`. The strict loader ignores only this exact metadata
allowlist after confirming there are no missing keys and that every payload is
a 1-D `torch.uint8` tensor; all parameter/buffer mismatches and other
unexpected keys still fail. The ignored key names and count are included in
extractor and cache provenance.

## Diagnostic manifest subset

To select the approved eight evenly spaced samples from the existing 64-entry
cache, without copying or re-extracting any tensors:

```bash
cd /home/anton/lerobot-video-vam
scripts/video_vam/subset_cosmos_cache_manifest.py \
  /home/anton/.cache/video-vam/cosmos-overfit-ep0-64/manifest.json \
  /home/anton/.cache/video-vam/cosmos-overfit-ep0-64/manifest-even8.json \
  --count 8
```

The subsetter validates the source manifest, preserves its entry dictionaries,
relative artifact paths, hashes, dataset, and global seed, then atomically writes
the same-root output. It records the parent manifest hash, selection strategy,
source/selected counts, and selected sample IDs as diagnostic-only provenance.
Use `--overwrite` explicitly to replace an existing output manifest; cache
artifacts are never copied or modified.

## Strided cache windows

`build_cosmos_feature_cache.py` enumerates current frames independently inside
each requested episode. With `--stride S`, the candidates are
`episode_start + 4`, `episode_start + 4 + S`, and so on, stopping before the
episode end. The causal five-frame window therefore never crosses an episode
boundary. `--frame-start` and `--frame-end` are absolute current-frame filters
applied after this episode-local enumeration (`frame-end` is exclusive), and
`--max-samples` truncates the combined episode-major ordered list. `--resume`
requires the same episode list, frame filters, stride, seed, dataset revision,
and exact ordered selection; `--overwrite` remains mutually exclusive with
`--resume`. The default `--stride 1` preserves the previous selection.

New manifests record `subset.stride` and
`provenance.selection.ordered_pairs`, the exact ordered `(episode_index,
current_frame)` list. Legacy schema-v1 manifests without those fields remain
readable as stride-1 manifests.
