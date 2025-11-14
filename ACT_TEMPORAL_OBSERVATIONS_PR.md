# Add Temporal Observation Support to ACT Policy

## Summary

This PR adds support for `n_obs_steps > 1` to the ACT (Action Chunking Transformer) policy, enabling it to process multiple consecutive observation frames to understand temporal behaviors and remember information from previous frames.

## Motivation

The original ACT implementation only supported `n_obs_steps=1`, meaning it could only see the current observation frame. This limitation prevents the model from:
- Understanding temporal dynamics and motion patterns
- Remembering objects or information that move out of the current frame
- Learning time-dependent behaviors

## Changes

### 1. Configuration (`src/lerobot/policies/act/configuration_act.py`)

**Removed validation restriction:**
- Removed the check that prevented `n_obs_steps != 1`

**Added observation delta indices:**
```python
@property
def observation_delta_indices(self) -> list | None:
    if self.n_obs_steps == 1:
        return None
    return list(range(1 - self.n_obs_steps, 1))
```

**Design Decision:** Following the pattern used by diffusion and vqbet policies, `observation_delta_indices` tells the dataset loader which temporal frames to load. For `n_obs_steps=32`, this returns `[-31, -30, ..., -1, 0]`, loading the last 32 observations.

### 2. Model Architecture (`src/lerobot/policies/act/modeling_act.py`)

**Temporal observation handling:**

- **Automatic shape detection:** The model detects whether observations have temporal dimension by checking tensor shapes
- **Backward compatibility:** Maintains full support for `n_obs_steps=1` (original behavior)
- **Temporal image processing:**
  - Images with shape `(B, n_obs_steps, C, H, W)` are flattened to `(B*n_obs_steps, C, H, W)`
  - Processed through the backbone in one batch for efficiency
  - **Spatial pooling applied** to avoid memory explosion: Features are globally averaged over spatial dimensions
  - Result: `n_obs_steps` tokens per camera (one per timestep) instead of `n_obs_steps * H * W`
  
- **State handling:** Robot state and environment state use the most recent observation (last timestep)
  - **Design Decision:** Using only the latest state is simpler and the temporal information is primarily useful for vision. Future work could concatenate all state observations.

- **Positional embeddings:** Fixed expansion to match batch size for both temporal and non-temporal cases

**Architecture comparison:**

| Configuration | Tokens per camera | Spatial detail | Temporal context |
|--------------|------------------|----------------|------------------|
| `n_obs_steps=1` (original) | `H' × W'` (~64) | ✅ Full spatial | ❌ No temporal |
| `n_obs_steps>1` (this PR) | `n_obs_steps` (~16) | ❌ Pooled spatial | ✅ Full temporal |

**Key implementation details:**
```python
# Images: Process all timesteps through backbone
img_flat = einops.rearrange(img, "b t c h w -> (b t) c h w")
cam_features = self.backbone(img_flat)["feature_map"]

# Add positional embeddings and apply global average pooling
cam_features_with_pos = cam_features + cam_pos_embed
cam_features_pooled = einops.reduce(cam_features_with_pos, "bt c h w -> bt c", "mean")

# Reshape to separate batch and temporal dimensions
cam_features_seq = einops.rearrange(cam_features_pooled, "(b t) c -> t b c", b=b, t=t)
```

### 3. Inference Support (`src/lerobot/policies/act/modeling_act.py`)

**Observation queue mechanism:**

- **`reset()` method:** Initializes observation queues (deques) for each input modality when `n_obs_steps > 1`
  - Separate queues for each camera (e.g., `observation.images.front`)
  - Queues for robot state and environment state if present
  - Queue size is automatically set to `n_obs_steps`

- **`select_action()` method:** Manages observation history during inference
  - Uses `populate_queues()` utility to add new observations to queues
  - Automatically duplicates first observation to fill queue during initial steps
  - Stacks queued observations along temporal dimension before feeding to model
  - Maintains backward compatibility with `n_obs_steps=1` (no queues needed)

**Inference flow:**
```python
# Step 1: Reset initializes empty queues
policy.reset()  # Creates deques with maxlen=n_obs_steps

# Step 2-17: Queue fills up (for n_obs_steps=16)
# First observation is duplicated until queue is full
action = policy.select_action(observation)  

# Step 18+: Full temporal window
# Queue contains last 16 observations, oldest is dropped when new one arrives
action = policy.select_action(observation)
```

### 4. Documentation Update (`robot/train_sort.sh`)

- Updated comments to list ACT alongside diffusion and vqbet as supporting `n_obs_steps > 1`
- Added automatic `_contextN` suffix to model repo ID when `n_obs_steps > 1` to prevent overwriting models

## Design Decisions

### 1. **Feature Aggregation Strategy**
Chose to use spatial pooling with temporal sequence tokens rather than:
- Keeping all spatial tokens per timestep (memory explosion: `n_obs_steps * H * W` tokens)
- Temporal pooling across all frames (loses temporal information)
- Separate temporal encoder (adds complexity)
- Channel concatenation (increases memory, limits flexibility)

**Rationale:** 
- Spatial pooling prevents memory explosion when using many temporal frames (e.g., `n_obs_steps=16`)
- Creates `n_obs_steps` tokens per camera, making sequence length manageable
- Transformer can learn temporal relationships through self-attention across pooled features
- **Trade-off:** Sacrifices spatial detail for temporal context (all frames are pooled, including current)

### 2. **State Handling**
Use only the most recent robot/environment state rather than all temporal states.

**Rationale:** 
- State is typically low-dimensional and slowly changing
- Temporal information is most critical for vision
- Simplifies implementation and reduces sequence length
- Can be extended in future if needed

### 3. **Backward Compatibility**
Maintained full compatibility with `n_obs_steps=1` by checking tensor dimensions.

**Rationale:** Ensures existing models and workflows continue working without modification.

## Testing

Created comprehensive test (`test_act_temporal.py`) covering:
- ✅ Forward pass with `n_obs_steps=32`
- ✅ Backward pass (gradient computation)
- ✅ Backward compatibility with `n_obs_steps=1`
- ✅ VAE encoder with temporal observations
- ✅ Multiple camera views

All tests passed successfully.

## Usage

```bash
# Train with 32 temporal observations
./robot/train_sort.sh --policy-type=act --n-obs-steps=32 --batch-size=4

# Will upload to: orellius/so101_sort_act_context32
# (automatically adds _contextN suffix to avoid overwriting)
```

## Performance Considerations

- **Memory usage:** Increases linearly with `n_obs_steps` (32x more image data for `n_obs_steps=32`)
- **Computation:** Backbone processes all timesteps in one batch (efficient)
- **Sequence length:** Increases by factor of `n_obs_steps` (e.g., 32x longer for images)

**Recommendation:** Start with smaller batch sizes when using high `n_obs_steps` values.

## Breaking Changes

None. All changes are backward compatible with `n_obs_steps=1`.

## Future Work

- [x] ~~Add observation history queues for inference~~ **COMPLETED**
- [ ] **Hybrid architecture**: Keep full spatial tokens for current frame + pooled tokens for history
  - Current frame: `H'*W'` tokens (full spatial detail, like original ACT)
  - Historical frames: `(n_obs_steps-1)` pooled tokens (temporal context)
  - Total per camera: `H'*W' + (n_obs_steps-1)` tokens
  - **Benefits**: Best of both worlds - spatial detail AND temporal context
  - **Note**: Would require retraining, but likely to improve performance significantly
- [ ] Experiment with different state aggregation strategies (concatenate vs. latest)
- [ ] Add learnable temporal positional encoding to distinguish temporal positions
- [ ] Benchmark performance vs. single-frame baseline
- [ ] Investigate attention patterns: Does the model actually use temporal information?

## Related Issues

Addresses the limitation that ACT could not process temporal sequences, which is critical for tasks requiring:
- Motion understanding
- Occlusion handling  
- Memory of out-of-view information

---

**Status: Complete.** The implementation includes both training and inference support, is well-tested, and maintains backward compatibility while enabling temporal observation processing for ACT.

**Test the inference:**
```bash
python test_act_temporal_inference.py
```

This loads the trained model `Orellius/so101_sort_act_context16` and verifies that observation queues work correctly during inference.


