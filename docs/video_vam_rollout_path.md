# Video-VAM rollout path and policy contract

This is the code-grounded contract for the current LeRobot checkout. It describes the path up to `Robot.send_action`, including the production `VideoVAMPolicy` adapter for the two frozen-feature experts.

## Reference hardware path

`lerobot-record` is teleoperation-only and explicitly directs policy deployment to `lerobot-rollout` (`src/lerobot/scripts/lerobot_record.py:192-197`). The policy rollout path is:

1. Context setup connects the configured robot, captures its initial `.pos` values, derives dataset/policy features, loads checkpoint-native policy pre/post-processors, and creates the inference engine (`src/lerobot/rollout/context.py:330-341`, `src/lerobot/rollout/context.py:366-370`, `src/lerobot/rollout/context.py:493-542`).
2. The episodic loop calls `robot.get_observation()`, applies the robot observation processor, asks `send_next_action` for an action, and waits on `CycleTimer` (`src/lerobot/rollout/strategies/episodic.py:239-272`).
3. `send_next_action` builds policy observation arrays from dataset feature metadata, calls the inference engine, maps the returned tensor to ordered action names, applies the robot action processor, and only then calls `robot.send_action` (`src/lerobot/rollout/strategies/core.py:321-369`).
4. The synchronous engine converts arrays to tensors, applies the checkpoint preprocessor, calls `policy.select_action`, applies the checkpoint postprocessor, squeezes the batch dimension, moves to CPU, and restores dataset action ordering (`src/lerobot/rollout/inference/sync.py:101-134`).
5. `CycleTimer.wait()` enforces cadence against monotonic deadlines with `precise_sleep`; it reports work groups that exceed `1/fps` (`src/lerobot/utils/cycle_timer.py:383-430`). `precise_sleep` is the low-level wait primitive (`src/lerobot/utils/robot_utils.py:19-55`).

The older helper `predict_action` has the same logical preprocessor → `select_action` → postprocessor sequence (`src/lerobot/common/control_utils.py:43-91`).

## Observation contract

For this dataset, policy-facing keys and shapes are:

- `observation.images.front`: one RGB image. Live hardware supplies HWC `uint8` `[480,640,3]`; `prepare_observation_for_inference` moves it to the selected device, divides by 255, permutes to CHW, and adds a batch dimension, producing float32 `[1,3,480,640]` in `[0,1]` (`src/lerobot/policies/utils.py:102-146`).
- `observation.state`: six values ordered `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos`; it becomes float32 `[1,6]`. `build_dataset_frame` constructs vectors strictly from feature `names` order (`src/lerobot/utils/feature_utils.py:110-136`; dataset declaration in `src/lerobot/datasets/vam/config.py:14-39`).
- `task`: exactly `take cube out of box` for the frozen experiment contract (`src/lerobot/datasets/vam/config.py:37`). The SmolVLA preprocessor appends a newline and tokenizes to length 48, as serialized in the checkpoint's `policy_preprocessor.json`.
- `robot_type`: the inference helper also adds this string (`src/lerobot/policies/utils.py:143-144`).

Cosmos and LTX do **not** consume the one-image SmolVLA contract directly. Their trained feature contract is a rolling causal history of five RGB frames `[1,3,5,480,640]`, then:

- Cosmos Predict2 layer 20, sigma 80, observed-prefix VAE mode, pool2 → bfloat16 `[1,4800,2048]`.
- LTX-2.5 layer 34, sigma 1, nine-frame VAE padding for five observed frames, pool2 → bfloat16 `[1,640,4096]`.

Those exact shapes and transforms are recorded in each expert checkpoint's `best.json`. The RTC engine exposes this history behind `--inference.observation_history_size=5`; `VideoVAMPolicy` losslessly reconstructs the original uint8 pixels, invokes the persistent extractor, applies pool2, and calls the expert checkpoint.

## Action and chunk contract

A robot-ready single action is a CPU tensor `[1,6]` or `[6]` in dataset feature order. `make_robot_action` maps each index to the metadata name without any semantic validation (`src/lerobot/policies/utils.py:183-209`). `SOFollower.send_action` strips `.pos` and writes absolute `Goal_Position` targets; only `max_relative_target`, when configured, clips the command (`src/lerobot/robots/so_follower/so_follower.py:204-230`).

Units are mixed despite the existing RMSE label “degrees”:

- The first five joints use `MotorNormMode.DEGREES` when the default `use_degrees=True` is retained (`src/lerobot/robots/so_follower/config_so_follower.py:41-42`; `src/lerobot/robots/so_follower/so_follower.py:49-59`).
- The gripper always uses `MotorNormMode.RANGE_0_100`, not degrees (`src/lerobot/robots/so_follower/so_follower.py:59`).
- `meta/info.json` verifies names, order, dtype and shape, but does not store units. A rollout configuration with `use_degrees=False` would therefore accept numerically plausible but wrong arm commands.

The two API levels differ:

- `predict_action_chunk(batch)` returns a batch of model-space actions before the external postprocessor. `PreTrainedPolicy` defines this as the full chunk API (`src/lerobot/policies/pretrained.py:282-289`).
- `select_action(batch)` returns one action and owns queue/history semantics (`src/lerobot/policies/pretrained.py:291-298`). For SmolVLA, a fresh model chunk is generated only when its queue is empty, only the first `config.n_action_steps` are queued, and one is popped per call (`src/lerobot/policies/smolvla/modeling_smolvla.py:341-370`).

The selected SmolVLA checkpoint is **not** a 30-action/3-second execution policy: it predicts `chunk_size=50` and queues `n_action_steps=10`, i.e. one second at 10 Hz. The custom SmolExpert decoders return exactly `[B,30,6]` raw actions and apply train-split state normalization plus action denormalization internally (`src/lerobot/policies/vam/smol_expert.py:878-937`). `VideoVAMPolicy` wraps that decoder without external normalization, implements `predict_action_chunk`/`select_action`, and accepts the existing RTC prefix contract.

## Normalization path and finding

Checkpoint processors are loaded by filename and run outside the model (`src/lerobot/policies/factory.py:180-220`). The default pipeline order is rename → batch → device → normalize and unnormalize → CPU (`src/lerobot/processor/factory.py:158-175`). Therefore calling `predict_action_chunk` directly without the checkpoint preprocessor/postprocessor returns normalized actions that can look numerically reasonable but are unsafe to send.

Audit result:

- The selected SmolVLA processor files are internally paired, but their serialized `action.count` is 6,536 and their episode statistics reach episode 39. The training config selects episodes 0–31, so these are dataset-global statistics that include held-out episodes 32–39, not train-only statistics. Training and inference are consistent with each other, but the evaluation has normalization leakage.
- Both custom SmolExpert run directories serialize identical normalizer tensors and JSON provenance for episodes 0–31 only, 1,572 training anchors, with padded actions excluded. `SmolExpertActionDecoder` calls `normalize_state` before the state projection and `denormalize_action` before returning (`src/lerobot/policies/vam/smol_expert.py:595-603`, `src/lerobot/policies/vam/smol_expert.py:933-937`).

## Wrong-but-plausible mismatch risks

1. Sending normalized model output instead of postprocessed physical targets.
2. Treating gripper range units as degrees, or starting SOFollower with `use_degrees=False`.
3. Changing action-name order: tensor-to-name mapping is positional.
4. Passing HWC/uint8 directly to a model, or pre-normalizing an image and then applying `/255` again.
5. Giving Cosmos/LTX fewer than five causal frames, wrong temporal order, wrong sigma, wrong hidden layer, wrong prompt artifact, or missing pool2.
6. Reusing a feature/noise seed that differs from cache-training provenance.
7. Executing all 30 SmolVLA outputs even though its saved native queue executes 10 of 50.
8. Assuming the synchronous production engine hides 1.1–1.2 s inference. It calls `select_action` inline; the custom experts have no deployed asynchronous queue.
9. Assuming physical joint limits are enforced. No saved SO-101 calibration file or explicit joint limits are present on `abakus`, and `max_relative_target` defaults to `None`.
10. A chunk replacement seam can jump even when every individual target is within the observed range; seam size must be checked separately.

## What the dry run establishes

`scripts/video_vam/dry_run_rollout.py` deliberately never constructs a robot. It streams one held-out episode frame at a time, maintains the five-frame history, runs inference in one background thread, drains the active 30-step chunk at 10 Hz until a replacement is ready, and emits raw predictions, anchor-aligned horizon errors, time-aligned executed-prefix errors, latency margin, seams, observed-range excursions, normalization provenance, and fixed-noise repeatability.

This scheduler remains the no-motor rehearsal. The production adapter now shares its extractor/expert computation, while collection-day work is limited to hardware calibration, real joint-limit values, relative-target caps, and a supervised first rollout.

## Temporal-consistency and RTC analysis

`scripts/video_vam/temporal_consistency.py` reuses the rollout backends and cached pool2 contexts to separate three effects: independent action-flow draws at fixed context/state, adjacent-context changes with fixed flow noise, and RTC-guided chunk replacement. It emits per-anchor `[30,6]` standard deviations, pairwise draw RMSE, horizon growth, context cosine/relative-RMS changes, feature-only and state-only counterfactual action changes, and RTC seam/accuracy metrics. Exact episode-32 adjacent contexts can be measured with `--live-features`; this performs only the requested in-memory backbone forwards and writes no feature cache.

The original dry-run `aggregate_rmse` seam is a deliberately naive replacement metric: it compares the next chunk's action 0 against the action executing when inference completes. LeRobot's `ActionQueue` instead drops the new chunk actions whose timestamps elapsed during inference. Schema version 2 therefore also reports `delay_aligned_*` seams using action index `installed_frame - anchor_frame`; this is the comparable unguided baseline for RTC.

`SmolExpertActionDecoder.sample_actions` now accepts the existing `RTCProcessor` guidance contract. Previous actions enter in stored physical units, are normalized with the checkpoint's train-split action statistics, padded to the expert's 32-dimensional latent action width, guided inside the shared flow integrator, and denormalized once on return. RTC disables the serialized unguided CUDA graph so guided and baseline semantics remain explicit.

Both experts are now hardware-loadable as `VideoVAMPolicy`; the observation-history boundary remains the source of their five policy-rate frames. With `--inference.observation_history_size=5`, `RTCInferenceEngine.notify_observation` snapshots each policy-rate observation, waits for a full history before first inference, and adds `<image_key>.history` tensors shaped `[B,C,T,H,W]` to the policy batch. The rollout loop calls this hook at the configured policy/dataset cadence (10 Hz here), independently of the slower asynchronous inference loop; therefore the five frames are consecutive 100 ms observations rather than inference-spaced frames. The default remains one and adds no key, so existing policies are unchanged. Reset clears the history to prevent frames leaking between episodes.

For one 480x640 RGB camera, five raw `uint8` snapshots occupy about 4.39 MiB. Preparing the history as float32 for the policy briefly occupies another 17.58 MiB (plus the ordinary current-frame tensor); each additional camera scales those figures linearly. `VideoVAMPolicy` consumes `observation.images.<camera>.history`, restores the extractor's uint8 `[B,3,5,H,W]` contract exactly, keeps the selected backbone resident, loads the SmolExpert checkpoint/normalizer, and returns physical 30x6 chunks to RTC. Hardware rollout remains intentionally blocked until real safety limits are supplied and tested with the arm present.

## Production adapter and collection-day gates

The implementation is split across:

- `src/lerobot/policies/vam/configuration_video_vam.py`: checkpoint paths, exact extractor settings, mixed-unit action order, feature-seed provenance, and mandatory hardware safety values.
- `src/lerobot/policies/vam/modeling_video_vam.py`: persistent Cosmos/LTX extractor, lossless float-history-to-uint8 restoration, pool2, SmolExpert/RTC call, physical action validation, and queue API.
- `src/lerobot/policies/vam/processor_video_vam.py`: rename/device-only processor pair. It deliberately contains no normalizer or unnormalizer.
- `scripts/video_vam/validate_video_vam_rollout.py`: same-anchor, same-noise numerical comparison against the offline dry-run backend.

The two expert run directories contain generated `config.json` files, so they can be passed directly as `--policy.path`. Rollout refuses to load the large backbone, and does not connect the robot, until all of these are provided:

1. `--policy.joint_limits_min='[...six values...]'` and `--policy.joint_limits_max='[...six values...]'` in trained action order. The first five values are degrees; the sixth is gripper `range_0_100`. These must come from the collection-day arm calibration/manual, not observed dataset extrema.
2. `--robot.use_degrees=true`.
3. `--robot.max_relative_target='{...}'` as a complete six-motor dictionary, in each motor's native configured units. The adapter rejects a scalar because the five arm targets are degrees while the gripper is `range_0_100`.
4. `--policy.feature_seed_episode_index=<non-negative collection episode id>`. If matching a recorded dataset anchor, also set `feature_seed_frame_offset` to that episode's absolute starting frame. For a new online episode, choose and record a stable index/offset so feature noise is deterministic and auditable.
5. `--inference.type=rtc --inference.observation_history_size=5 --fps=10`.

Every returned chunk is checked for shape and finiteness. Values beyond a configured hard limit raise `VideoVAMSafetyError` and never enter either RTC queue. `joint_limit_tolerance` defaults to zero; if a small positive tolerance is explicitly configured, only tolerance-sized floating-point excursions are clamped, while material excursions still fail loudly. `SOFollower.send_action` then independently enforces `max_relative_target` against the current pose.
