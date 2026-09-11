# Training artifacts

## Unified SmolExpert

Run from the repository with existing, matched feature caches:

```bash
uv run python scripts/video_vam/train_smolexpert.py \
  --backbone cosmos14b \
  --train-manifest /path/to/features/train/manifest.json \
  --val-manifest /path/to/features/val/manifest.json \
  --save-last-every 1000
```

Omitting `--output-dir` chooses a unique directory under repository-relative
`outputs/train/smolexpert-*`, independent of the working directory. `outputs/`
is already ignored. Explicit output paths retain their existing semantics.
Non-empty directories are refused unless the existing `--overwrite` option is
explicitly supplied. Overwrite removes only known trainer files (including the
new last weights and manifest), not unrelated files or checkpoint directories.
Never use overwrite to continue training: resume is not implemented.

The run contains `best.safetensors`, `last.safetensors`, `run_manifest.json`,
`normalizer.safetensors`, `normalizer.json`, and the existing metric/summary files.
Best minimizes validation `val_rmse`; Eval-2 does not select best. Last is saved
at optimizer-step intervals and on normal/early-stopping completion, **before**
best is loaded for final evaluation. `--save-last-every 0` disables periodic last
saves. No numbered snapshots are accumulated. A run interrupted before its first
save may have no weights. No shutdown/signal checkpoint handler is added.

Weights and manifest files are individually atomically replaced using temporary
files in the same directory. This is not a multi-file transaction or a guarantee
against power loss. Check recorded SHA-256 hashes after interruption: a crash
between weight and manifest publication can leave an old manifest. Temporary
files left by a hard kill are not swept automatically.

For evaluation, specify the existing run explicitly:

```bash
uv run python scripts/video_vam/train_smolexpert.py \
  --val-manifest /path/to/features/val/manifest.json \
  --output-dir /path/to/run --eval-only
```

Evaluation does not rewrite the training manifest or recompute its normalizer.
The current evaluator does not automatically validate run-manifest hashes.

### Provenance and portability gaps

The versioned manifest records arguments, code commit/dirty state, policy identity,
action dimensions/horizon, dataset episode splits and source-manifest hashes,
normalizer statistics/hash, checkpoint paths/hashes/steps, and selection metric.
All source cache metadata except per-sample entries is preserved, including
available dataset revisions, backbone/transformer/VAE identities, LoRA metadata,
frame/FPS/pooling and prompt configuration. Requested context transform and actual
resulting context shape are recorded. Missing source configuration is **unknown**,
not silently inferred; legacy caches may lack LoRA rank/alpha, frame offsets or
immutable model revisions. Pretrained policy revision is currently unresolved.

Bundled artifact paths are relative to the run; external source paths and arguments
are historical hints. Moving a run preserves its internal references, but does not
bundle datasets, feature caches, backbones or adapters. A dirty-tree flag is not a
snapshot of source edits. These records aid identification, not guaranteed replay.
SmolExpert checkpoints are **weights only**: no optimizer, scheduler, RNG or
sampler state and no exact training resume.

## Native SmolVLA Scale-100 wrapper

```bash
bash scripts/video_vam/run_train_smolvla_scale100_1hr.sh
# Optional fresh run parent:
RUN_DIR=/path/to/new-run bash scripts/video_vam/run_train_smolvla_scale100_1hr.sh
```

Defaults to a fresh repository `outputs/train/smolvla-scale100-*` parent containing
`train.log`, `COMPLETE` (only on success), and `training/` native output. `RUN_DIR`
and optional `OUTPUT_DIR` environment overrides must not already exist. With an
external OUTPUT_DIR override, logs remain in RUN_DIR; default layout is colocated.
No directory is deleted. Native training requires the output child not to exist
at launch, so logs live in its parent.

The wrapper uses the existing native `--save_freq=0`: only the final checkpoint,
with native processor/config/optimizer/RNG state and the native relative `last`
link. Validation is disabled, so **there is no best checkpoint**, and interruption
before completion loses training progress. No native run-manifest integration,
retention hook or distributed-trainer changes are included. Native train config
and processor artifacts remain its metadata source. A retention helper without
save-loop integration would not safely bound live checkpoints; none is added.

No existing artifacts are moved, migrated, pruned or deleted. Video-LoRA trainers,
rollout consumers and historical runs are unchanged.

## Focused verification

```bash
uv run pytest tests/utils/test_run_artifacts.py tests/scripts/test_train_smolexpert.py -q
bash -n scripts/video_vam/run_train_smolvla_scale100_1hr.sh
```

Tests use temporary directories and CPU synthetic data, not real training jobs.

## Audited Checkpoint Inventory (2026-09-08)

| Category          | Model / Name                         | Path                                                                           | Status / Disposition                         |
| :---------------- | :----------------------------------- | :----------------------------------------------------------------------------- | :------------------------------------------- |
| **Foundation**    | Cosmos 2B Video Backbone             | `outputs/models/cosmos2b/video_backbone/v2w_pretrained_cosmos.pt`              | ✅ Verified (3.9 GB)                         |
| **Foundation**    | Cosmos 14B Video Backbone            | `outputs/models/cosmos14b`                                                     | ⚠️ Symlink to cache; unverified durability   |
| **LoRA**          | Cosmos 2B Video-LoRA (step 6,000)    | `outputs/train/cosmos-video-lora-step6000/best_lora.safetensors`               | ✅ Verified; on Hugging Face                 |
| **LoRA**          | Cosmos 2B T=2 Distilled Student      | _Targeted cache/output paths_                                                  | ❌ Missing weights; retrain required         |
| **SmolExpert**    | Cosmos 2B Pool2 (T=16)               | `outputs/train/cube-out-of-box-cosmos-pool2-smolexpert/`                       | ✅ Verified (`best`+`last`); on Hugging Face |
| **SmolExpert**    | Cosmos 2B T=2 Undistilled            | _Targeted cache/output paths_                                                  | ❌ Missing weights; retrain required         |
| **SmolExpert**    | Cosmos 2B T=2 Distilled              | _Targeted cache/output paths_                                                  | ❌ Missing weights; retrain required         |
| **SmolExpert**    | Cosmos 7B Protocol 1.0               | `outputs/train/cosmos7b-protocol1-smolexpert/`                                 | ✅ Verified (`best`, step 17k)               |
| **SmolExpert**    | LTX-2.5 Pool2 & Unpooled             | `outputs/train/cube-out-of-box-ltx-{pool2,unpooled}-smolexpert/`               | ✅ Verified (`best`+`last`); on Hugging Face |
| **Native Policy** | SmolVLA v1 (train-only stats, 29.2k) | `outputs/train/cube_out_of_box_il_smolvla_train_only_stats_0_31_20260826_1hr/` | ✅ Verified; on Hugging Face                 |
| **Native Policy** | SmolVLA v2 (scale-100, 25k)          | `outputs/train/cube_out_of_box_scale100_smolvla_1hr/`                          | ⚠️ Saved, but dataset quarantined            |

_Rules:_

1. Always write checkpoints to `outputs/train/<run-name>/` with `best.safetensors`, `last.safetensors`, and `run_manifest.json`.
2. Never store persistent weights in `/tmp/` or volatile cache roots.
3. Do not run training or evaluation on `Orellius/cube_out_of_box_v2` until the dataset metadata/row discrepancy is resolved.
