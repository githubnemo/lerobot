# Video-VAM index

Status: current entry point, 2026-08-29. This document maps the current Video-VAM workflows, artifacts, scripts, and contracts; experiment-specific documents remain authoritative for details.

## Orientation

Video-VAM evaluates video foundation models as robot-policy backbones for the SO-101 cube-out-of-box task. Cosmos and LTX extract causal video features, which are consumed by SmolExpert or World2Action-style action heads, with caches and provenance preserved for repeatable comparisons. The final rollout path keeps heavy Cosmos/LTX inference on `abakus` and can forward action chunks to a Mac over SSH RPC.

- `docs/video_vam_roadmap.md` — current decisions, priorities, and open ablations.
- `docs/video_vam_cube_out_of_box_leaderboard.md` — append-only results ledger.
- `docs/video_vam_research_diary.md` — historical journey; its latest substantive entry is Aug 26 and it is stale for newer T=2, WorldExpert, and INT4 work.
- `docs/video_vam_action_rmse_protocol.md` — frozen physical-action evaluation protocol.
- `docs/video_vam_rollout_path.md` and `docs/video_vam_rpc.md` — offline rollout and hardware/RPC contracts.

## Canonical workflows

### A. Frozen cache -> SmolExpert -> RMSE

1. Create or select an episode split with `scripts/video_vam/create_vam_split.py` (no dedicated wrapper); contract: `video_vam_world2action.md`.
2. Build a backend cache with `build_cosmos_feature_cache.py` via `run_build_cosmos_feature_cache.sh`, or `build_ltx_feature_cache.py` directly; contracts: `video_vam_cosmos_extractor.md` and `video_vam_ltx_extractor.md`.
3. Train `train_smolexpert_on_cosmos.py` (the implementation is backend-generic despite its name); current queues include `run_ltx_smolexpert_overnight.sh` and Cosmos SmolExpert queues.
4. Run `post_train_rmse.sh` or `evaluate_action_rmse.py`; contract: `video_vam_action_rmse_protocol.md`.
5. Append the measured result to `video_vam_cube_out_of_box_leaderboard.md`; do not rewrite prior rows.

### B. Video LoRA -> adapted cache -> expert

1. Train `train_cosmos_video_lora.py`, normally through `run_cosmos_video_lora_then_smolexpert.sh`.
2. Fuse/verify with `merge_cosmos_lora.py`, or use the adapter-aware extraction path.
3. Rebuild Cosmos caches with `build_cosmos_feature_cache.py` and preserve LoRA/checkpoint provenance.
4. Train `train_smolexpert_on_cosmos.py` using the adapted manifests.
5. Evaluate with the frozen RMSE protocol and record the result in the leaderboard.

The historical joint alternative is `train_cosmos_lora_cotrain.py` via `run_cosmos_lora_cotrain.sh`; it is a distinct experiment because video and action gradients are coupled.

### C. T=2 WorldExpert chain

1. Use `run_cosmos_t2_we.sh` to train `train_cosmos_t2_world_expert.py` on observed-only Cosmos `state_t=2` features.
2. For the cheap comparison, `run_cosmos_t2_linear_then_we.sh` first trains the `LinearWorldReadout` head and then invokes the WorldExpert chain.
3. Generate anchor videos with `preview_cosmos_t2_we_video_prediction.py`; no standalone wrapper is required because the queue invokes it.
4. Compare the resulting video/world-model behavior and then rebuild a T=2 cache plus SmolExpert if the arm is retained.

Contract: `docs/video_vam_world_expert.md`, with current status summarized in `video_vam_roadmap.md`.

### D. Preview and video generation

- Generic Cosmos prediction: `preview_cosmos_video_prediction.py`, wrapper `run_preview_cosmos_video_prediction.sh`; contract: `video_vam_video_prediction_preview.md`.
- T=2 WorldExpert prediction: `preview_cosmos_t2_we_video_prediction.py`; contract: `video_vam_world_expert.md`.
- Prompt artifacts: `generate_cosmos_prompt_embedding.py` and `precompute_ltx_prompt.py`; contracts: `video_vam_prompt_embedding.md` and `video_vam_ltx_extractor.md`.

### E. Rollout and RPC

1. Rehearse without hardware using `dry_run_rollout.py`; contract: `video_vam_rollout_path.md`.
2. Validate the wrapper with `validate_video_vam_rollout.py` and `run_wrapper_validation_queue.sh`.
3. Start `rpc_server.py` with `run_rpc_server.sh` on `abakus`; contract: `video_vam_rpc.md`.
4. Use `rpc_client.py` or `run_mac_vam_rpc.py` from the Mac. Hardware motion requires calibrated per-motor safety limits; the RPC server must not be treated as a simulator.
5. Analyze chunk timing and RTC seams with `temporal_consistency.py`; contract: `video_vam_temporal_consistency_report.md`.

## Cache compatibility

| Producer/cache              | Schema and context                                                                                                               | Compatible consumers                                                               | Notes                                                                                   |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Cosmos frozen feature cache | `cosmos_feature_cache.py`, schema v2; context width 2048; layer-20 contract; typically `state_t=16` or observed-only `state_t=2` | `train_smolexpert_on_cosmos.py`, `train_cosmos_world2action.py`, Cosmos evaluators | Strict provenance, tensor, sidecar, and manifest validation.                            |
| LTX frozen feature cache    | `ltx_feature_cache.py`, cache/manifest schema v1; context width 4096; hidden layer 34; `state_t=2` or `state_t=8`                | LTX action trainers/evaluators and generic SmolExpert dispatch                     | LTX producer identity, context grid, noise seed, and both artifact hashes are checked.  |
| Cosmos layer-mix cache      | `cosmos_layer_mix_cache.py`, schema v1; multi-depth Cosmos contexts with backend-specific grids                                  | `train_smolexpert_on_cosmos.py` and Cosmos layer-mix consumers                     | Layer names, temporal frames, and producer provenance are not interchangeable with LTX. |
| LTX layer-mix cache         | `ltx_layer_mix_cache.py`, schema v1; six-depth LTX contexts, commonly layers 8/14/20/26/34/40                                    | `train_ltx_layer_mix.py`, LTX layer-mix evaluators                                 | Uses LTX-specific token counts and provenance.                                          |

Do not rename artifact identities, alter sidecar key sets, or regenerate existing artifacts after changing serializers. A cache with the same tensor shape but a different backend, `state_t`, context transform, noise seed, or checkpoint is not compatible.

## Script inventory

### Training

| File                                   | Role                                                                      |
| -------------------------------------- | ------------------------------------------------------------------------- |
| `train_cosmos_world2action.py`         | Main frozen-Cosmos action trainer with random anchors and resume support. |
| `train_cosmos_world2action_overfit.py` | Tiny Cosmos cached-context overfit diagnostic.                            |
| `train_smolexpert_on_cosmos.py`        | Backend-generic cached-context SmolExpert trainer.                        |
| `train_cosmos_video_lora.py`           | Video-only Cosmos LoRA fine-tuning.                                       |
| `train_cosmos_lora_cotrain.py`         | Joint Cosmos video/action LoRA co-training.                               |
| `train_cosmos_t2_world_expert.py`      | T=2 Cosmos LoRA plus latent WorldExpert training.                         |
| `train_ltx_world2action_tiny.py`       | Bounded online LTX tiny-overfit gate.                                     |
| `train_ltx_world2action.py`            | Full cached LTX World2Action training.                                    |
| `train_ltx_layer_mix.py`               | LTX multi-depth layer-mix action training.                                |

### Cache and data artifacts

| File                                  | Role                                                  |
| ------------------------------------- | ----------------------------------------------------- |
| `build_cosmos_feature_cache.py`       | Builds strict Cosmos feature artifacts and manifests. |
| `build_ltx_feature_cache.py`          | Builds resumable LTX feature artifacts and manifests. |
| `build_ltx_layer_mix_cache.py`        | Builds six-depth LTX layer-probe caches.              |
| `create_vam_split.py`                 | Creates the fixed episode split and validation probe. |
| `subset_cosmos_cache_manifest.py`     | Selects a deterministic diagnostic cache subset.      |
| `generate_cosmos_prompt_embedding.py` | Generates the official Cosmos T5 prompt artifact.     |
| `precompute_ltx_prompt.py`            | Generates the LTX Gemma prompt artifact.              |
| `precompute_fastwam_text.py`          | Generates standalone FastWAM text rows.               |
| `merge_cosmos_lora.py`                | Fuses and verifies a Cosmos LoRA adapter.             |
| `prepare_smolvla_train_only_stats.py` | Creates a train-only statistics overlay.              |
| `inspect_cube_dataset.py`             | Audits pinned dataset metadata.                       |

### Benchmarks and evaluation

| File                                    | Role                                                     |
| --------------------------------------- | -------------------------------------------------------- |
| `benchmark_cosmos_extraction.py`        | Cosmos extraction and optional decoder benchmark.        |
| `benchmark_ltx_extraction.py`           | Correctness-gated LTX extraction benchmark.              |
| `benchmark_ltx_layer_mix.py`            | Measures LTX multi-tap extraction cost.                  |
| `bench_ltx_state_t.py`                  | Compares LTX `state_t=8` with observed-only `state_t=2`. |
| `bench_ltx_int4_resident.py`            | Measures GPU-resident INT4 LTX extraction.               |
| `evaluate_action_rmse.py`               | Frozen physical-action comparison across policies.       |
| `evaluate_cosmos_world2action_cache.py` | Evaluates cached Cosmos-compatible samples.              |
| `evaluate_ltx_action_rmse.py`           | Evaluates LTX action checkpoints under protocol 1.0.     |
| `report_ltx_expert_result.py`           | Appends completed LTX expert results.                    |
| `report_ltx_layer_probe.py`             | Appends completed LTX layer-probe results.               |

### Preview and smoke tests

| File                                       | Role                                                            |
| ------------------------------------------ | --------------------------------------------------------------- |
| `preview_cosmos_video_prediction.py`       | Generic Cosmos prediction and alignment preview.                |
| `preview_cosmos_t2_we_video_prediction.py` | T=2 WorldExpert future-video preview.                           |
| `smoke_test_cosmos_extractor.py`           | Cosmos extraction smoke test and shared sample/runtime helpers. |
| `smoke_test_ltx_extractor.py`              | LTX extraction smoke test and resource report.                  |

### Rollout, RPC, and diagnostics

| File                                 | Role                                                       |
| ------------------------------------ | ---------------------------------------------------------- |
| `dry_run_rollout.py`                 | No-motor open-loop replay for Cosmos, LTX, and SmolVLA.    |
| `prepare_video_vam_rollout.py`       | Writes policy configs beside expert runs.                  |
| `validate_video_vam_rollout.py`      | Compares wrapper output with offline inference.            |
| `temporal_consistency.py`            | Measures flow uncertainty, context changes, and RTC seams. |
| `rpc_client.py`                      | Mac-side SSH RPC client.                                   |
| `rpc_server.py`                      | Inference HTTP server on `abakus`.                         |
| `run_mac_vam_rpc.py`                 | Real Mac-side SO-101 rollout over RPC.                     |
| `check_context_plumbing.py`          | Runtime context/state/action pairing assertion.            |
| `isolated_sigma427_probe.py`         | Probe runner for a sigma-4.27 artifact.                    |
| `resumable_sigma427_probe.py`        | Resumable grouped Cosmos context probe.                    |
| `sigma_discriminability_probe.py`    | Noise-level context discriminability probe.                |
| `state_context_incremental_probe.py` | State/context incremental-information probe.               |
| `verify_random_anchor.py`            | Random-anchor and normalizer verification.                 |

### Shell helpers and wrappers

- `cosmos_cuda_env.sh`, `gpu_lock.sh`, `post_train_rmse.sh`, and `test_gpu_lock.sh` provide environment, locking, evaluation, and lock-test primitives.
- `run_build_cosmos_feature_cache.sh`, `run_evaluate_cosmos_world2action_cache.sh`, `run_preview_cosmos_video_prediction.sh`, `run_smoke_test_cosmos_extractor.sh`, `run_smoke_test_ltx_extractor.sh`, `run_train_cosmos_world2action_overfit.sh`, and `run_train_cosmos_world2action.sh` are thin Python wrappers.
- Queue/orchestration files include `run_cosmos_bridge_retry_queue.sh`, `run_cosmos_expert_bench_retry.sh`, `run_cosmos_lora_cotrain.sh`, `run_cosmos_statet2_lora_attention_smolexpert.sh`, `run_cosmos_statet2_unpooled_smolexpert.sh`, `run_cosmos_t2_linear_then_we.sh`, `run_cosmos_t2_we.sh`, `run_cosmos_video_lora_then_smolexpert.sh`, `run_experiments.sh`, `run_fastwam_baseline.sh`, `run_gpu_chain.sh`, `run_hour.sh`, `run_ltx_attention_smolexpert_queue.sh`, `run_ltx_layer_probe_queue.sh`, `run_ltx_mixer_variants_queue.sh`, `run_ltx_plateau.sh`, `run_ltx_smolexpert_overnight.sh`, `run_ltx_statet2_unpooled_smolexpert.sh`, `run_rpc_server.sh`, `run_smolvla_train_only_stats_queue.sh`, `run_videolora_smolexpert_from_best.sh`, and `run_wrapper_validation_queue.sh`.
- `chain_smoke.sh`, `clip_ab.sh`, and `smoke_random_anchor.sh` are small historical smoke/ablation launchers.

### Historical / completed experiments

These files remain in place for reproducibility; they should not be assumed to be current entry points:

- Aug 18–20 probes: `isolated_sigma427_probe.py`, `resumable_sigma427_probe.py`, `sigma_discriminability_probe.py`, `state_context_incremental_probe.py`, `check_context_plumbing.py`, and `verify_random_anchor.py`.
- Historical one-hour/random-anchor launchers: `run_experiments.sh`, `run_hour.sh`, `clip_ab.sh`, `smoke_random_anchor.sh`, and `chain_smoke.sh`.
- Completed or superseded LTX queues: `run_ltx_smolexpert_overnight.sh`, `run_ltx_plateau.sh`, `run_ltx_layer_probe_queue.sh`, and `run_ltx_mixer_variants_queue.sh`.
- Failed/comparison-only joint training: `run_cosmos_lora_cotrain.sh` and `train_cosmos_lora_cotrain.py`.
- `run_fastwam_baseline.sh` is explicitly a guarded non-launching baseline.
- `run_cosmos_bridge_retry_queue.sh` is tied to a partial/queued Bridge experiment.

No files are moved or deleted by this classification.

## Conventions

- Work from `/home/anton/lerobot-video-vam`; use the repository `.venv/bin/python` or source `scripts/video_vam/cosmos_cuda_env.sh` so `VAM_VENV` is configured.
- Source `cosmos_cuda_env.sh` before Cosmos/Transformer Engine work. Source `gpu_lock.sh` and call `acquire_gpu_lock <name>` before GPU work.
- Primary artifact root: `/home/anton/.cache/video-vam/` (equivalent to `~/.cache/video-vam/` for user `anton`). Keep caches, runs, prompt artifacts, splits, and logs in separately named subdirectories.
- W&B project: `video-vam-world2action`; preserve run names and links in result records.
- Queue scripts log through `tee`, use stage names, and commonly write `DONE`, `FAILED`, or stage-specific completion markers. A `FAILED` marker is not evidence that a partial artifact is safe to reuse; validate its manifest and sidecars.
- Leaderboard updates are append-only. Use the frozen RMSE protocol and record manifest/checkpoint/provenance identifiers with every new row.

## Known wrinkles

- `train_smolexpert_on_cosmos.py` is backend-generic despite its historical filename; it dispatches Cosmos, LTX, and layer-mix manifests.
- `smoke_test_cosmos_extractor.py` doubles as a helper library. Several LTX builders, benchmarks, and trainers import its sample preparation and runtime helpers.
- `run_ltx_layer_probe_queue.sh` calls `rg` at lines 50 and 98. `rg` is not installed on `abakus`; those checks must use portable `grep`.
- Cosmos feature caches and LTX feature caches have different schemas, hidden widths, producer identities, and validation contracts. Equal tensor names do not make them interchangeable.
- The research diary is a historical record, not the current implementation index; use the roadmap and this document for current navigation.
