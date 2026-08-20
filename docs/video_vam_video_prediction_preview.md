# Cosmos Video2World prediction preview

This preview is a diagnostic harness for the pinned `hubnemo/cube_out_of_box_dataset`
scene and the frozen Cosmos-Predict2 2B video2world backbone. It is deliberately
separate from feature extraction and does not modify the extractor contract.

## Contract

`run_preview_cosmos_video_prediction.sh` sources `cosmos_cuda_env.sh` and invokes
the preview CLI. The CLI loads one validated LeRobot sample using the same
`load_real_sample`/`prepare_sample` path as the extractor, so the RGB input is the
causal five-frame window `[current-4, ..., current]`, in native `480x640` RGB,
with no online resize. The cached official T5 embedding is loaded and passed as
`[1, 512, 1024]` BF16. The backend then calls the extractor's existing
`_preprocess_images` and `_encode_conditional_latents` helpers; those helpers are
not duplicated or changed here.

The current generic seam is `VideoBackboneSpec` plus
`CosmosVideo2WorldBackend`. It carries the checkpoint, model size, device/dtype,
latent geometry, prompt geometry, and sampler settings. Only the `2B` implementation
is enabled today; future Bridge-LoRA, quantized, 14B, and LTX backends can implement
the same seam without changing dataset, alignment, metrics, or report code.

## Temporal alignment

The pinned upstream 480p/10fps config has `state_t=16`,
`latent_conditional_frames=2`, `temporal_compression_factor=4`,
`temporal_window=16`, and `chunk_duration=81`. The tokenizer's public mapping is:

```text
pixel_frames = 1 + (latent_frames - 1) * 4
16 latent frames -> 61 RGB frames (6.1 seconds at 10 fps)
81-pixel VAE chunk -> 21 latent frames (the chunk limit, not this rollout horizon)
5 RGB conditioning frames -> 2 latent prefix frames
```

Therefore decoded output frame `0` is dataset frame `current-4`, output frame `4`
is `current`, and output frame `5` is the first genuinely predicted future frame.
The full decoded video covers absolute dataset indices
`current-4 ... current+56`. Metrics are computed only on `current+1 ... current+56`
(the 56 future frames); the first five decoded frames are retained as conditioning
reconstructions for visual inspection and are never included in the score.

Every real run records an alignment probe. It encodes the five RGB conditioning
frames to the two clean prefix latents, zero-pads the remaining latent state,
decodes it, and compares decoded frames against the input history for candidate
offsets 0 through 8. The report must show 61 decoded frames and the expected
frame-zero offset. This is the empirical check for the VAE's causal convention;
it is independent of the learned denoising quality.

## Sampler and determinism

The rollout follows the pinned upstream `RectifiedFlowAB2Scheduler` at mimic-video
commit `e3355dbc93132b576c02f920a59b4fc18a4f5906`: Karras-like sigma schedule
`80.0 -> 0.002`, order `7`, 35 denoising steps, two-step AB after the initial
Euler step, and a final clean pass. It uses the official frame-replacement mask,
with the first two latent frames kept clean at `sigma_conditional=0.0001`.
The official empty-negative-prompt path makes the conditional and unconditional
text embeddings identical, so the configured guidance value is recorded but does
not trigger a second distinct text branch.

A fixed seed controls the architecture-invariant initial noise and is recorded in
the strict JSON report along with the sampler, checkpoint SHA256, resolution,
fps, and every conditioning/output/metric frame index. The report uses
`allow_nan=False`; an exact zero-MSE PSNR is represented as `null` rather than
non-standard JSON `Infinity`.

## Running

From `/home/anton/lerobot-video-vam` on `abakus`:

```bash
scripts/video_vam/run_preview_cosmos_video_prediction.sh \
  --episode 0 --frame-index 60 --seed 0 \
  --output-dir /home/anton/.cache/video-vam/cosmos-prediction-previews
```

The selected current frame must leave 56 frames in the same episode. Add
`--overwrite` to replace an existing window. Each run writes:

- `*-predicted-full.mp4`: all 61 decoded frames, including conditioning reconstructions;
- `*-predicted-future.mp4`: the 56 genuinely predicted frames;
- `*-ground-truth-future.mp4`: the aligned dataset frames at 10fps;
- `*-side-by-side.mp4`: predicted future on the left, ground truth on the right;
- `*.json`: strict schema version 1 report with per-frame/mean PSNR and SSIM and
  LPIPS status. LPIPS is computed only when an already-installed implementation
  initializes without installing or downloading anything; otherwise the report
  says why it was skipped.

Unit tests are CPU-only:

```bash
source scripts/video_vam/cosmos_cuda_env.sh
"${VAM_VENV}/bin/pytest" -q tests/policies/test_cosmos_video_prediction_preview.py
```

## Measured 2B windows

The authoritative values are the corresponding JSON reports; this section records
the same values for quick review. Both runs used seed `0`, 35 steps, and the same
RTX 4090 process configuration.

- Early window, episode `0`, current frame `4`:
  - mean PSNR `14.747557580647225 dB`, mean SSIM `0.563569112015622`;
  - first rollout `44.0309270239959 s`, total clip `58.662134070007596 s`; same-seed verification rerun `61.60313704801956 s` / `77.83545838299324 s`;
  - peak allocated VRAM `7.083672046661377 GiB`, reserved `10.74609375 GiB`;
  - report `/home/anton/.cache/video-vam/cosmos-prediction-previews/episode-0000-frame-000004.json`.
- Mid-manipulation window, episode `19`, current frame `2203`:
  - mean PSNR `10.901444021990203 dB`, mean SSIM `0.49007481602685793`;
  - rollout `43.986209274997236 s`, total clip `58.760937601007754 s`;
  - peak allocated VRAM `7.083672046661377 GiB`, reserved `10.74609375 GiB`;
  - report `/home/anton/.cache/video-vam/cosmos-prediction-previews/episode-0019-frame-002203.json`.

The predicted-full MP4 SHA256 was reproduced identically on a same-seed rerun for
the early window (`4cbc33d735299cbf0b0e9ef790ccbd44208ab0c5cd5ac0d2c165a39a114c0c31`).
The cache's metadata lists episode 20 as a valid episode, but its local video shard
`videos/observation.images.front/chunk-000/file-001.mp4` is absent; episode 19 was
used for the complete mid-manipulation measurement instead.

The report's `runtime` object contains rollout wall-clock seconds, total clip
seconds, peak allocated/reserved VRAM, GPU name, and software versions.

## Diagnostic quality findings

The rerun reports now include separate conditioning, baseline, VAE-only, and
pixel-path diagnostics. Values below use the same PSNR/SSIM implementation and
the same 56 genuinely predicted frame indices as the Cosmos score.

- Episode 0, frame 4: conditioning reconstruction `47.2747 dB / 0.9894`;
  static baseline `25.6867 dB / 0.8721`; conditioning mean
  `24.2760 dB / 0.8812`; ground-truth copy `null / 1.0000`; VAE-only
  round-trip `43.7419 dB / 0.9810`; Cosmos `14.7476 dB / 0.5636`.
- Episode 19, frame 2203: conditioning reconstruction `43.8535 dB / 0.9876`;
  static baseline `13.7211 dB / 0.7063`; conditioning mean
  `14.5943 dB / 0.7328`; ground-truth copy `null / 1.0000`; VAE-only
  round-trip `42.2212 dB / 0.9843`; Cosmos `10.9014 dB / 0.4901`.

These results rule out a broken pixel path or temporal alignment. Conditioning
reconstruction and VAE round-trip fidelity are high, the copy baseline is perfect,
and the MP4 write/read probe is `44.61 dB / 0.9870` (episode 0) and
`42.19 dB / 0.9848` (episode 19), with no resize or vertical flip. The pixel
boundaries are recorded in each JSON report: dataset TCHW RGB uint8, extractor
BCTHW BF16 `[-1,1]`, VAE output `[-1,1]` converted to THWC RGB uint8, and PyAV
`rgb24` throughout the writer/reader path.

The headline is therefore genuine model quality, not a scoring bug: Cosmos is
worse than the static baseline on the early window and better than static on the
mid window, but still poor in absolute terms. The contact sheets are the visual
inspection artifacts; they are ordered as eight evenly sampled ground-truth
frames on the top row and the corresponding predicted frames on the bottom row,
with absolute dataset frame indices burned into each tile:

- `/home/anton/.cache/video-vam/cosmos-prediction-previews/episode-0000-frame-000004-contact-sheet.png`
- `/home/anton/.cache/video-vam/cosmos-prediction-previews/episode-0019-frame-002203-contact-sheet.png`

The local dataset cache is incomplete: episodes 0–19 are readable from
`file-000.mp4`, while episodes 20–39 fail because
`videos/observation.images.front/chunk-000/file-001.mp4` is absent. No re-download
was attempted.
