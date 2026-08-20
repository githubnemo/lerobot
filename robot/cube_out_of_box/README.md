# Cube out-of-box Video VAM foundation

This folder contains the first safe, model-independent foundation for the Video VAM experiment. It does not contain Cosmos code or a VAM policy.

## Exact data contract

- Hub dataset: `hubnemo/cube_out_of_box_dataset`
- Required revision: `243370c3c08bcbd860133c4a0d658ea7c1d2e77e`
- Format: LeRobotDataset v3 (`v3.0`), 40 episodes, 6536 frames, 10 Hz
- Camera feature: `observation.images.front`, metadata shape `480x640x3` RGB. The physical camera is the wrist camera despite the `front` key.
- State key: `observation.state`; action key: `action`. Both are absolute 6D joint positions in this exact order: `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos`
- Task: `take cube out of box`
- Image observation history: causal offsets `[-4, -3, -2, -1, 0]`; proprioception is one current `observation.state` token at offset `[0]` with sample shape `[1,6]`
- Action chunk: 30 steps at 10 Hz (3 seconds); `action_is_pad` marks only the trailing actions outside the current episode
- Deterministic episode subsets: 1 = `[0]`, 3 = `[0,20,39]`, 5 = `[0,9,20,33,39]`

The validator checks metadata, decoded TCHW camera samples, finite state/action/image tensors, ordered names, causal history masks, trailing `action_is_pad`, task strings, and available timing or episode boundaries. It does not use dataset-level quantile statistics for normalization.

## Metadata audit

From the repository root, using the repository environment:

```bash
uv run python scripts/video_vam/inspect_cube_dataset.py --subset 1
uv run python scripts/video_vam/inspect_cube_dataset.py --subset 3 --root /path/to/local/cube_out_of_box_dataset
uv run python scripts/video_vam/inspect_cube_dataset.py --subset 5
uv run python scripts/video_vam/inspect_cube_dataset.py --subset all
```

The audit is read-only with respect to dataset contents: it uses `LeRobotDataset` to load metadata and boundary samples, and never calls a writer or Hub push operation. A local root can still be populated by LeRobot normal read/download path when files are missing.

## Tiny SmolVLA baseline

The training wrapper pins the dataset revision, sets `chunk_size=30`, supports the deterministic subsets, keeps the HF cache under `/home/anton` by default, and refuses to overwrite an existing output directory:

```bash
cd /home/anton/lerobot-video-vam
./robot/cube_out_of_box/1.1_train_il.sh \
  --policy-type smolvla --subset 1 --steps 100 --batch-size 1 \
  --device cpu --no-amp
```

Use `--policy-type act` to train ACT from scratch, or `--from-scratch` to avoid the SmolVLA pretrained checkpoint. `--resume` is explicit and requires the existing `last` checkpoint. The `vam` policy name is reserved as a future extension point and currently exits without training.

## Physical rollout safety gate

`1.2_eval_il.sh` uses the current `lerobot-rollout --strategy.type=episodic` interface because `lerobot-record` is teleoperation-only in this checkout. It refuses to connect to or move a robot unless `--allow-robot` is explicitly supplied. It also defaults to `--dataset.push_to_hub=false` and refuses to delete an existing evaluation output unless `--overwrite` is explicit.

Before any rollout, pass the metadata audit, inspect the selected checkpoint and camera semantics, verify the follower and camera ports, and run a short supervised check with an emergency stop ready. Unit tests and audits do not send robot actions.
