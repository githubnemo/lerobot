# Mac RPC rollout quickstart

## Install on the Mac

From the Mac `lerobot-video-vam` checkout (not another project's environment):

```sh
uv sync --locked --extra feetech
```

Keep the Mac launcher and remote RPC scripts at matching versions. Preserve local edits when syncing.
The SSH host alias `abakus` must work non-interactively, including its configured SOCKS/proxy connection:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=no -o ControlPath=none abakus true
```

A SOCKS/SSH failure during POST is a transport failure, not evidence that inference succeeded.
Predictions are not automatically retried.

## RPC-only test: no camera or motor connection

Original SmolVLA checkpoint (the launcher's default 029200 checkpoint):

```sh
uv run python scripts/video_vam/run_mac_vam_rpc.py \
  --policy smolvla --dry-run --inference.type rtc --no-compile \
  --ready-timeout 180 --rpc-timeout 120 --stop-server
```

Explicit checkpoint **on abakus**, not a Mac filesystem path:

```sh
uv run python scripts/video_vam/run_mac_vam_rpc.py \
  --policy smolvla \
  --checkpoint /home/anton/lerobot-video-vam/outputs/train/cube_out_of_box_scale100_smolvla_1hr/checkpoints/025000/pretrained_model \
  --dry-run --inference.type rtc --no-compile \
  --ready-timeout 180 --rpc-timeout 120 --stop-server
```

RTC dry-run makes two sequential predictions using synthetic grey images and zero state:
1. An initial no-prefix request.
2. A prefix-guided request using the first response's original chunk, minus one simulated consumed
   action, with `inference_delay=1`. It sends the real remaining prefix without zero-padding.

SmolVLA originals are normalized model actions; executed actions are physical units.
VideoVAM originals/leftovers are physical actions; its decoder normalizes internally.
Each response is checked for finite chunks of the advertised shape. Timings are printed separately
and include transport plus inference (the first may include warmup). History is refreshed between calls.
This does **not** validate cameras, calibration, hardware safety, sustained control rate, or task success.

For a single no-prefix request with no RTC conditioning fields, use `--inference.type sync`.
Native RTC uses the repository's denoising-guidance approximation: this is **not a paper-equivalence guarantee**.

## Server lifecycle

- Reuse requires matching resolved checkpoint, checkpoint-file fingerprint, protocol, and configuration.
- Switching policy/checkpoint/config replaces only the launcher-owned tmux instance. Missing identity
  metadata fails closed. An unmanaged listener is not killed to free the port.
- Ownership is recorded before starting the model child. Startup failures report the remote log under
  `/tmp/video-vam-rpc-PORT-INSTANCE.log`.
- Normally, a server started by this invocation is stopped on exit; a reused server is left alone.
  `--keep-server` retains a successfully used new instance. `--stop-server` requests cleanup even for
  a reused instance, but still requires matching managed ownership.
- No exclusive GPU lock is acquired; concurrent inference processes are permitted.

## Before any hardware rollout

Do not remove `--dry-run` until the real robot setup is verified separately. Hardware mode requires:
- An SO-100/101 follower serial port and robot ID, existing matching calibration, and an RGB 640x480
  OpenCV front camera delivering at least 10 FPS.
- `--robot.use_degrees=true` and **six measured bounds each** in `--joint-limits-min` and
  `--joint-limits-max`: five arm joints in degrees, then gripper position in [0,100]. Do not invent limits.
- Positive relative-target limits and a supported arm: disconnect releases torque and may let it fall.

The loop uses 10 Hz observations, one in-flight request, and aborts on invalid actions, stale timing,
RTC underruns, or clipping. A passing RPC-only test is not authorization to move hardware.

## Current holds

V2 data remains **on hold**; RPC validation does not authorize new collection or promotion.
The default Cosmos/VideoVAM deployment checkpoint is missing. Do not substitute an unrelated checkpoint
or claim VideoVAM rollout readiness until compatible artifacts and calibrated limits are available.
