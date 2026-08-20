# LTX-2.5 VAM extractor

Status: additive extractor and CPU contract tests are implemented. The real
checkpoint smoke is intentionally separate because the single RTX 4090 is
shared and the official weights were not cached when this work began.

## Stage 1: established facts

### Selected checkpoint and loader

The selected checkpoint is the official distilled LTX-2.5 22B transformer:

- Hugging Face repository: [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5)
- Pinned revision: `6c7e5e573ac1667efc83407806fe9b0b93730e60`
- Transformer: `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors`
- Video VAE: `vae/ltx-2.5-video-vae-conv-bf16.safetensors`
- Prompt encoder (only needed when embeddings are not precomputed):
  `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`; this packed
  artifact includes the tokenizer/config sidecars.
- Official loader: [`Lightricks/LTX-2`](https://github.com/Lightricks/LTX-2),
  source commit `400fd31054597515f47125691032c04b1c3ee24e`
- Python route: `ltx_pipelines.utils.blocks.DiffusionStage.from_checkpoint`,
  with `offload_mode=cpu` and the official `fp8-cast` policy for a BF16
  checkpoint. The extractor hooks the selected `transformer_blocks` entry and
  captures its video stream before the final output projection.

`/home/anton/AuViMi` was checked first, but it is a DeepDaze/SIREN project and
contains no LTX loader or LTX checkpoint. The older `/home/anton/mus2vid`
reference mentioned in the research notes has been removed; its shell history
records an LTX-2 (not 2.5) Diffusers distilled + BNB4 setup. That older result
is useful evidence that the 4090 needs quantization/offload, but it is not a
2.5 architecture or VRAM measurement.

### Architecture and geometry

The official LTX-2 source and the pinned LTX-2.5 repository establish:

- 48 dual-stream transformer blocks shared by video and audio.
- Video stream: 32 attention heads × 128 head dimension = **4096 hidden
  width**, 128 input/output latent channels.
- Audio stream: 32 × 64 = 2048 width. It is not run by this video-only
  extraction call; it is retained in the checkpoint's AV model contract.
- Gemma 4 12B text encoder with LTX projection. The video prompt context is
  **4096 wide** (audio context is 2048 wide); the 22B checkpoint carries the
  text-side projection, so the extractor accepts already-encoded video prompt
  embeddings and never receives state or action tensors.
- Causal video VAE: 128 channels, spatial compression 32×32 and temporal
  compression 8. The causal temporal grid is `1 + 8*k` input frames; the
  encoder output is `(B, 128, 1 + (F-1)/8, H/32, W/32)`.
- Transformer patch size is 1×1×1. Thus at 480×640, one latent frame is
  15×20 tokens. An 8-latent-frame state is
  `8 × 15 × 20 = 2,400` tokens, each width 4096: **`[B, 2400, 4096]`**.
  The raw grid form used by the extractor is `[B, 8, 15, 20, 4096]`.
- The matched policy state uses **8 latent frames**, representing `8×8−7 = 57`
  frame positions (`5.7 s` at 10 Hz), the closest legal LTX horizon to Cosmos'
  61-frame (`6.1 s`) state. This gives `8 × 15 × 20 = 2,400` tokens.
- The policy still observes exactly five causal RGB frames. They are padded by
  repeating the last observed frame four times, producing nine VAE input frames
  and two clean conditional latent frames. The remaining six latent frames are
  future/noise positions. Thus LTX retains the observation contract but has a
  different legal temporal grid and a slightly shorter total horizon (57 vs.
  61 frames); this is explicit, not a silent change in policy observations.
- Cosmos therefore has `19,200 × 2,048` context while LTX has `2,400 × 4,096`.
  After projection to 2,048, LTX cross-attention is roughly 8× cheaper per
  decoder step. Training wall-clock is consequently not a fair compute proxy
  between the two arms.

Sources:

- [LTX-2 core README](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/README.md)
- [official transformer configurator](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py)
- [official VAE implementation](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py)
- [LTX-Video compression paper, Table 1](https://arxiv.org/html/2501.00103v1)
- [LTX-2.5 model card](https://huggingface.co/Lightricks/LTX-2.5)

### Flow-time parameterization

LTX uses rectified-flow interpolation with normalized `sigma`/flow time:

- `sigma=1` is the fully noisy endpoint and `sigma=0` is the clean endpoint.
- The official distilled schedule starts at `1.0` and ends at `0.0`:
  `[1.0, .99375, .9875, .98125, .975, .909375, .725, .421875, 0.0]`.
- `GaussianNoiser` implements `lerp(clean, noise, sigma)` on denoised
  positions. Therefore the extractor's one-forward high-noise point is
  exactly **`sigma=1.0`**, not a near-clean value.

The Cosmos reference's `sigma=10` is in its unnormalized sigma parameterization;
it must not be copied numerically into LTX. The matched point is the endpoint
with the same relative noise level, LTX `sigma=1.0`.

### 4090 memory requirement

The selected loading strategy is the official `ltx-core` `fp8-cast` policy
against the BF16 transformer, with CPU block streaming (`offload_mode=cpu`).
This is preferred over implementing BNB4 inside the official loader: the source
repository supports FP8 cast and block streaming directly, while the 21.5 GiB
ComfyUI INT8-convrot file requires replicating a ComfyUI-specific loader and the
18.7 GiB NVFP4 file targets Blackwell, not Ada sm_89. The benchmark's BNB4
result remains a useful fallback if FP8-cast cannot fit or is unstable.

The 42.0 GiB BF16 file cannot be resident in 24 GiB VRAM. FP8-cast is a
precision confound relative to the BF16 Cosmos arm: weights/linear arithmetic
use FP8 casting policies for the video backbone, while the VAE and returned
hidden states remain BF16. This must be reported alongside any comparison.
Only blocks through the matched depth are semantically needed, but the first
measurement uses the official full-stage loader; truncation is a follow-up
optimization after correctness is established.

The real-weight measurement command is:

```bash
source scripts/video_vam/cosmos_cuda_env.sh
scripts/video_vam/run_smoke_test_ltx_extractor.sh \
  --checkpoint /home/anton/.cache/video-vam/ltx-2.5-models/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --video-vae /home/anton/.cache/video-vam/ltx-2.5-models/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  --prompt-embedding /path/to/gemma4-video-embedding.safetensors
```

It reports wall-clock, peak allocated/reserved VRAM, shape, dtype, and full
checkpoint/VAE hashes. The real-weight run remains unverified because the
checkpoint files were not downloaded from the gated repository. No large
generation or training job is started by the launcher.

## Stage 2: matched extraction point

The extractor selects **LTX block index 34 of 48**:

```text
round(20 / 28 * 48) = 34
```

This is the nearest block index to Cosmos layer 20/28 (71.4% depth), giving an
index-relative depth of 34/48 (70.8%). It captures the output of that block
before the final output normalization/projection, just as the Cosmos seam
captures an intermediate hidden state rather than the predicted velocity.

The noise point is **LTX sigma=1.0**. LTX's schedule is normalized to `[0, 1]`,
so this is the maximum-noise endpoint and is directly analogous in relative
terms to Cosmos' intentionally high `sigma=10` forward. Choosing e.g. `.4`
or `.1` would instead move toward the clean endpoint, where the intermediate
representation is less useful for the mimic-video comparison.

Both choices are explicit in `LTXExtractorConfig` and recorded in
`LTXProvenance`.

## Stage 3: implementation contract

New files:

- `src/lerobot/policies/vam/ltx_extractor.py`: frozen extractor, canonical
  seed derivation, causal VAE padding, official optional loader, hidden hook,
  and provenance.
- `src/lerobot/policies/vam/context_adapter.py`: reusable trainable
  `LayerNorm(4096) -> Linear(4096, 2048)` adapter.
- `src/lerobot/policies/vam/adapted_world2action.py`: additive composition that
  puts the adapter inside the existing decoder's denoiser boundary, after the
  decoder's intentional frozen-context detach.
- `src/lerobot/policies/vam/ltx_feature_provenance.py`: strict producer identity
  object for the next cache schema.
- `scripts/video_vam/smoke_test_ltx_extractor.py` and
  `scripts/video_vam/run_smoke_test_ltx_extractor.sh`.

The extractor accepts only `rgb_history` and a 4096-wide video prompt
embedding (or a caller-supplied prompt encoder). State, action, and padding
labels are not accepted by the backbone call. Both the latent encoder and
transformer are put in eval mode with `requires_grad=False`; extraction runs
under `no_grad`.

### Deterministic noise

`derive_window_seed` is byte-for-byte the Cosmos cache formula:

```text
SHA256(f"{revision}\0{episode}\0{frame}\0{global_seed}")[:8] mod (2**32 - 1)
```

The resulting seed initializes NumPy `RandomState`, matching Cosmos'
architecture-invariant noise. The seed and all geometry are recorded in
`LTXProvenance`.

### Adapter capacity

For LTX width 4096 and decoder width 2048:

- LayerNorm affine parameters: `2 × 4096 = 8,192`.
- Linear parameters: `4096 × 2048 + 2048 = 8,390,656`.
- Total delta: **8,398,848 trainable parameters**.

For matching widths, `ContextAdapter(2048)` is a true `nn.Identity` with zero
parameters, so the existing Cosmos-2B path remains unchanged. The same module
supports Cosmos-14B's 5120-wide context.

### Cache format proposal (not applied here)

The current `cosmos_feature_cache.py` schema is version 2 and has strict
nested key sets plus a hard-coded `[B, N, 2048]` context validator. It cannot
record an LTX producer without either rejecting the new metadata or silently
losing it. That owned file was therefore not edited.

The next cache schema should increment to version 3 and add a strict
`backbone` object, at minimum:

```json
{
  "name": "LTX-2.5-22B-distilled",
  "checkpoint_identity": "...",
  "checkpoint_sha256": "...",
  "model_revision": "6c7e5e573ac1667efc83407806fe9b0b93730e60",
  "hidden_width": 4096,
  "adapter_output_width": 2048,
  "hidden_layer": 34,
  "num_blocks": 48,
  "noise_parameterization": "normalized_rectified_flow_sigma",
  "noise_level": 1.0,
  "token_geometry": [8, 15, 20],
  "token_count": 2400,
  "target_frame_count": 57,
  "dtype": "bfloat16",
  "quantization": "fp8-cast",
  "offload_mode": "cpu"
}
```

The save/load path should reject mixed `backbone` identity, checkpoint hash,
layer, noise, geometry, or output-width values in one training manifest. The
additive `ltx_feature_provenance.py` module provides this object and equality
check now; integration into the owned cache schema remains a follow-up.

## Stage 4: verification

CPU checks currently cover:

- canonical deterministic seed and NumPy noise equivalence;
- layer/noise selection;
- five-to-nine-frame causal padding and output `[1, 2400, 4096]` contract;
- full provenance round trip and mismatch rejection;
- adapter shape, exact parameter count, trainability, and 2048-wide identity;
- existing World2Action flow-matching loss with adapter gradients and decreasing
  loss over six optimizer steps.

The adapter training plumbing was exercised on two synthetic LTX-width windows through the existing `World2ActionDecoder` loss wrapper and the new `AdaptedWorld2ActionDecoder` composition (CPU, six SGD steps, LR=0.01, fixed flow time and noise):

```text
step 0  1.04927957
step 1  0.30347410
step 2  0.08554403
step 3  0.02266963
step 4  0.00576827
step 5  0.00143524
```

The adapter projection received non-zero gradients and the loss decreased
monotonically. This validates trainability of the adapter/decoder plumbing, but
not the unavailable real LTX weights.

The focused suite (`test_context_adapter.py`, `test_ltx_feature_provenance.py`,
`test_ltx_extractor.py`, and `test_adapted_world2action.py`) passes: **10 passed**.
`ruff check`, `ruff format --check`, and `bash -n` for the launcher all pass.
The full repository suite was attempted CPU-only. It reached unrelated camera
and CUDA-dependent failures; the first reproducible failure was the existing
`tests/cameras/test_opencv.py::test_read[128x128]` fixture (91 passed, 17
skipped before `--maxfail=1` stopped). The long unconstrained run was stopped
at 44% after several minutes without progress; no LTX-focused test failed.

The real GPU smoke is intentionally gated by `nvidia-smi` and requires a free
window. On the attempted run, the guard observed 24,009 MiB used and 100% GPU
utilization, so it correctly refused to start. The official checkpoint download
also returned Hugging Face `401 GatedRepoError` because this machine is not
authenticated for `Lightricks/LTX-2.5`; no authorization workaround was used.
Therefore real LTX wall-clock/peak-VRAM numbers remain **unverified** and must
be collected after Hugging Face access is granted and the 4090 is free.

A prior injected CUDA contract smoke completed in 0.090 s with the former
16-latent-frame geometry; that measurement is intentionally stale after the
matched-horizon decision. The final 8-latent-frame CUDA smoke has not been run
because the 4090 is occupied by another training job. It should report hidden
grid `(1, 8, 15, 20, 4096)`, context `(1, 2400, 4096)`, and peak VRAM once the
GPU is free. Injected-backbone numbers would validate plumbing only and would
not measure LTX-2.5 model memory or throughput.
