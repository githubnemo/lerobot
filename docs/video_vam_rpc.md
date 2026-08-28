# Video-VAM SSH RPC v1

The RPC keeps the LTX/Cosmos backbone and decoder on `abakus`; the Mac only encodes frames and forwards JSON over SSH. The server binds to `127.0.0.1:8765`, and handles one request at a time.

## Start on abakus

Wait until the GPU is free. `run_rpc_server.sh` acquires the repository GPU lock. Direct `rpc_server.py` does not; hold the lock yourself if you start it that way:

```bash
ssh abakus
cd /home/anton/lerobot-video-vam
source scripts/video_vam/cosmos_cuda_env.sh
# Hold the existing lock using the normal operator procedure, then:
/home/anton/lerobot-video-vam/.venv/bin/python scripts/video_vam/rpc_server.py \
  --policy video_vam \
  --checkpoint /home/anton/.cache/video-vam/runs/ltx25-smolexpert-20260826/run-a-pool2 \
  --joint-limits-min -180 -180 -180 -180 -180 0 \
  --joint-limits-max 180 180 180 180 180 100
```

For SmolVLA, use its checkpoint directory and omit joint limits:

```bash
/home/anton/lerobot-video-vam/.venv/bin/python scripts/video_vam/rpc_server.py \
  --policy smolvla \
  --checkpoint /home/anton/lerobot-video-vam/outputs/train/cube_out_of_box_il_smolvla_heldout_0_31_20260819_1hr/checkpoints/029200/pretrained_model
```

`--host` and `--port` are configurable, but the defaults are `127.0.0.1` and `8765`.

## Call from the Mac

`run_mac_vam_rpc.py` is a real SO-101 rollout. The Mac owns USB/cameras and
executes each returned `[30][6]` chunk at `--fps=10`. Flags match
`lerobot-rollout` for robot, cameras, fps, duration, and task. Cosmos/LTX
inference stays on abakus over SSH RPC.

Use the same Python env as `lerobot-rollout` (needs `import lerobot`).

```bash
python scripts/video_vam/run_mac_vam_rpc.py \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodemXXXX \
    --robot.id=so101 \
    --robot.use_degrees=true \
    --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 10}}' \
    --robot.max_relative_target='{"shoulder_pan": 10, "shoulder_lift": 10, "elbow_flex": 10, "wrist_flex": 10, "wrist_roll": 10, "gripper": 20}' \
    --fps=10 --duration=30 --task="take cube out of box" \
    --keep-server
```

`--robot.max_relative_target` is required and must be a complete per-motor
dict (degrees vs gripper 0-100 are different units). `--inference.type=rtc`
executes `--inference.rtc.execution_horizon` steps (default 10) then replans;
it is not overlapping RTC guidance. Default `--inference.type=sync` executes
the full 30-step chunk, then replans. Pause between chunks is expected.

`--dry-run` tests the JSON path without connecting the arm.

The launcher starts the `video-vam-rpc` tmux session on abakus if `/healthz`
is not already 200. The joint limits in `run_rpc_server.sh` are placeholders,
not this arm's calibrated limits.

Low-level JSON client (`rpc_client.py`) is still available for debugging:

```bash
python scripts/video_vam/rpc_client.py --policy video_vam --dry-json
```

The request is JSON with `state`, `images_front_u8_png_b64` (one PNG string for
SmolVLA or five strings for Video-VAM), and optional `feature_seed`. The
successful response contains `action_chunk` shaped `[30][6]` in physical
units, `policy`, and `latency_s`.
