# Video VAM action-RMSE comparison protocol

**Protocol version:** 1.0
**Status:** frozen; concrete frame IDs are supplied by the cache manifest at evaluation time
**Purpose:** native-systems physical-action comparison, not flow-loss comparison

## Dataset and split

- Dataset repository: `hubnemo/cube_out_of_box_dataset`
- Dataset revision: the immutable `dataset.revision` recorded in the VAM cache manifest.
- Training episodes: `0..31`.
- Validation episodes: `32..39`.
- Exact validation frame identifiers: the ordered `(episode_index, frame_index, sample_id)` records in `CacheManifest.entries` for the supplied manifest. The evaluator records the complete list in its JSON output and rejects missing or duplicate identifiers. No random re-splitting or frame resampling is permitted.
- This checkout does not contain a materialized evaluation cache manifest yet. Until one is supplied, the concrete frame-ID list is intentionally unresolved; the evaluator-generated JSON is the versioned run artifact that freezes it.

## Action target and mask

- Raw action feature: the six stored SO-101 `.pos` dimensions, in degrees.
- Horizon: exactly 30 future actions at 10 Hz.
- SmolVLA: use only indices `0:30` from its native 50-action chunk; no retraining.
- FastWAM: use only indices `0:30` from its native 32-action chunk; no retraining.
- VAM: use its native `[30, 6]` action output.
- Padding: use the same boolean `action_is_pad` mask for both policies. A padded action token contributes zero error and zero denominator weight.
- Predictions must be denormalized to stored physical units before scoring.

## Sampling and aggregation

- Each sample receives `evaluation_seed + batch_position` as its deterministic seed.
- SmolVLA receives fixed Gaussian noise through `predict_action_chunk(..., noise=noise)` with shape `[1, 50, 32]`; only `[0:30, 0:6]` is scored.
- VAM receives the same per-sample seed through its deterministic seeded sampler.
- Aggregate RMSE is exactly `sqrt(sum(masked_squared_error) / n_valid_scalars)`, where `n_valid_scalars = valid_action_tokens * 6`.
- Report aggregate RMSE and six per-joint RMSE values in degrees.
- Aggregate squared-error sums globally; never average batch RMSEs.

## Native policy configurations

### FastWAM

- Native FastWAM policy and checkpoint-native pre/post processors.
- Native horizon: 32 actions; score only the first 30 actions.
- Text conditioning: cached UMT5 context from the checkpoint training config, with the text encoder disabled.

### VAM

- Native World2Action decoder and Cosmos context cache.
- Horizon: 30 actions, six dimensions.
- State/action normalization: train-window-only per-dimension min/max to `[-1, 1]`, excluding padded action tokens.
- Context source: the `context` tensor in the supplied cache artifact.
- This normalization deliberately differs from SmolVLA and is disclosed, not harmonized.

### SmolVLA

- Standard LeRobot pretrained checkpoint containing `config.json`, model weights, and processor artifacts.
- Native checkpoint chunk size is expected to be 50; score only its first 30 actions.
- Native normalization: checkpoint-native dataset-wide mean/std for state and action.
- This normalization deliberately differs from VAM and is disclosed, not harmonized.

### Baseline

- `state_repeat`: repeat the current six-dimensional proprioceptive position for all 30 actions. This is the “do not move” baseline.

## Reproducibility record

The evaluator output records protocol version, manifest path, dataset revision, ordered sample IDs and frame IDs, seed, backend/checkpoint paths, normalization provenance, horizon, action dimension, mask rule, and aggregation formula.
