# Physical Robot Testing Plan: Video Action Models (VAM) vs Baselines

This document outlines the step-by-step physical test protocol for the SO-100 / SO-101 robot arm running the `cube_out_of_box` task using `scripts/video_vam/run_mac_vam_rpc.py`.

---

## 1. Architecture & Execution Overview

- **MacBook (`antons-macbook-pro`)**: Reads front camera (OpenCV, 640x480 @ 10 FPS), communicates with Feetech bus servo follower arm via serial (`/dev/tty.usbmodem*`), and runs Real-Time Chunking (RTC) action queuing.
- **Remote GPU (`abakus` RTX 4090)**: Hosts the vision backbone + SmolExpert action policy over SSH HTTP RPC.
- **Automatic Server Management**: The Mac script automatically checks whether the correct model server is running on `abakus`. If a different model is requested, it gracefully stops the old server and starts the new one in tmux (`video-vam-rpc-8766`).
- **Safety First**: Test every policy with `--dry-run` first to verify RPC connectivity and action generation before enabling motor torque. Always support the arm when disconnecting or shutting down.

---

## 2. Hardware Pre-Flight Checklist

1. Check follower USB serial device:
   ```bash
   ls /dev/tty.usbmodem*
   # Boxed follower typically: /dev/tty.usbmodem5A460820701
   ```
2. Verify camera index (box camera is usually index 1, built-in Mac webcam is 0):
   ```bash
   # In robot/hw.sh:
   # boxed setup uses follower /dev/tty.usbmodem5A460820701 and camera index 1
   ```

---

## 3. Prioritized Model Test Sequence

Run each model from `/Users/antonwiehe/lerobot` on the MacBook:

### 🥇 1. Cosmos 3 Edge Video-LoRA (~80 ms) — Benchmark Leader

The Scale-100 benchmark champion. Highest accuracy (13.62° V1 / 17.12° V2), lowest latency (~80 ms), and uses 600 unpooled spatiotemporal tokens.

```bash
# Dry run verification:
.venv/bin/python scripts/video_vam/run_mac_vam_rpc.py --checkpoint cosmos3_lora --dry-run

# Live physical rollout:
.venv/bin/python scripts/video_vam/run_mac_vam_rpc.py \
    --checkpoint cosmos3_lora \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A460820701 \
    --robot.id=so101 \
    --duration=30 --task="take cube out of box"
```

---

### 🥈 2. Cosmos 2B T=2 Distilled (~204 ms) — Representation Distillation Winner

The core breakthrough of the project: closes the performance gap to the heavy $T=16$ teacher (CosSim 0.81, loss 8.06) while running at full real-time 2-frame speed (~204 ms).

```bash
# Dry run verification:
.venv/bin/python scripts/video_vam/run_mac_vam_rpc.py --checkpoint cosmos2b_t2_distilled --dry-run

# Live physical rollout:
.venv/bin/python scripts/video_vam/run_mac_vam_rpc.py \
    --checkpoint cosmos2b_t2_distilled \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A460820701 \
    --robot.id=so101 \
    --duration=30 --task="take cube out of box"
```

---

### 🥉 3. Cosmos 2B T=2 Undistilled (~204 ms) — Fast Baseline

Direct ablation comparison against the distilled model to visually confirm how representation distillation reduces jitter and fixes trajectory drift.

```bash
# Dry run verification:
.venv/bin/python scripts/video_vam/run_mac_vam_rpc.py --checkpoint cosmos2b_t2_undistilled --dry-run

# Live physical rollout:
.venv/bin/python scripts/video_vam/run_mac_vam_rpc.py \
    --checkpoint cosmos2b_t2_undistilled \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A460820701 \
    --robot.id=so101 \
    --duration=30 --task="take cube out of box"
```

---

### 4. Cosmos 2B T=16 Reference (~1,200 ms) — Full-Context Reference

The gold standard teacher representation model. High accuracy but slow latency (~1.2 s per chunk); tests how RTC handles large inference delays.

```bash
# Dry run verification:
.venv/bin/python scripts/video_vam/run_mac_vam_rpc.py --checkpoint cosmos2b_t16 --dry-run

# Live physical rollout:
.venv/bin/python scripts/video_vam/run_mac_vam_rpc.py \
    --checkpoint cosmos2b_t16 \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A460820701 \
    --robot.id=so101 \
    --duration=30 --task="take cube out of box"
```

---

### 5. Cosmos 3 Edge Base (~80 ms) — Pretrained World Model Representation

Cosmos 3 Edge backbone without robot video LoRA fine-tuning. Shows how well zero-shot foundation features transfer compared to task-adapted representations.

```bash
# Dry run verification:
.venv/bin/python scripts/video_vam/run_mac_vam_rpc.py --checkpoint cosmos3_base --dry-run

# Live physical rollout:
.venv/bin/python scripts/video_vam/run_mac_vam_rpc.py \
    --checkpoint cosmos3_base \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A460820701 \
    --robot.id=so101 \
    --duration=30 --task="take cube out of box"
```

---

### 6. SmolVLA v2 (Scale-100) — Full 100-Episode Imitation Baseline

Trained on the 100-episode dataset. Pure end-to-end vision-language-action policy baseline.

```bash
# Dry run verification:
.venv/bin/python scripts/video_vam/run_mac_vam_rpc.py --checkpoint smolvla_v2 --dry-run

# Live physical rollout:
.venv/bin/python scripts/video_vam/run_mac_vam_rpc.py \
    --checkpoint smolvla_v2 \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A460820701 \
    --robot.id=so101 \
    --duration=30 --task="take cube out of box"
```

---

### 7. SmolVLA v1 — 40-Episode Historical Baseline

Trained strictly on historical Episodes 0–31 of the V1 dataset.

```bash
# Dry run verification:
.venv/bin/python scripts/video_vam/run_mac_vam_rpc.py --checkpoint smolvla_v1 --dry-run

# Live physical rollout:
.venv/bin/python scripts/video_vam/run_mac_vam_rpc.py \
    --checkpoint smolvla_v1 \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A460820701 \
    --robot.id=so101 \
    --duration=30 --task="take cube out of box"
```

---

## 4. Useful Flags & Tuning

- `--dry-run`: Runs full RPC inference with synthetic inputs to test server health without opening camera or motor ports.
- `--robot.cameras={front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}`: Override camera index if box camera is on index 0 instead of 1.
- `--keep-server`: Keeps the remote model server running in tmux after the rollout finishes (faster subsequent runs).
- `--stop-server`: Forces the remote server to shut down upon rollout completion.
- `--duration <seconds>`: Rollout length in seconds (default: 30).
