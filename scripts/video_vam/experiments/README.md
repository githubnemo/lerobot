# Numbered Experiment Launch Scripts (`expNNN_<slug>.sh`)

Every training, evaluation, or RL post-training run is launched from an **immutable, sequentially numbered script** in this directory so every result in `docs/video_vam_leaderboard.md` maps 1:1 to the exact command that produced it.

## Rules

1. **Sequential numbering:** Pick the next unused 3-digit prefix (`exp001_...`, `exp002_...`, ...).
2. **Immutability:** Once a script has been executed on any GPU host (`abakus`, RunPod, `sahni`), **never edit it in place** — copy to the next number for any hyperparameter or dataset change.
3. **Portable paths:** Use `${VAM_CACHE_DIR:-$HOME/.cache/video-vam}` and `${REPO_ROOT}` rather than hardcoded `/home/anton/...` paths so scripts run identically on RunPod (`root`), `abakus` (`anton`), and `sahni` (`nemo`).
4. **Post-run logging (mandatory):** After the run completes (or stops/fails), record the result in:
   - `docs/video_vam_leaderboard.md`
   - `docs/video_vam_status.md`
   - `docs/video_vam_research_diary.md`

## Template (`expNNN_<slug>.sh`)

```bash
#!/usr/bin/env bash
# expNNN: <short hypothesis / goal>
# Target host: RunPod RTX 5090 (32 GB) | Dataset: <repo>@<rev>
set -Eeuo pipefail

export RUN_TAG="expNNN_<slug>"
export VAM_CACHE_DIR="${VAM_CACHE_DIR:-$HOME/.cache/video-vam}"

bash scripts/video_vam/run_experiment.sh <preset> \
  --backbone-checkpoint "${VAM_CACHE_DIR}/cosmos3-edge" \
  --seed 0
```

## Index

| ID       | Script                                             | Backbone / Method                                         | Dataset Split                            | Auto HF Repo                                                   |
| :------- | :------------------------------------------------- | :-------------------------------------------------------- | :--------------------------------------- | :------------------------------------------------------------- |
| `exp001` | `exp001_c3_edge_joint_lora_bs64.sh`                | Cosmos 3 Edge (L20) + Joint LoRA (`gen_only`), BS 64      | `sort_cubes@a7c156de` (`0:68` / `68:77`) | `Orellius/so101_exp001_c3_edge_joint_lora_bs64`                |
| `exp002` | `exp002_c3_nano_l24_joint_lora_bs32.sh`            | Cosmos 3 Nano (L24/36) + Joint LoRA (`gen_only`), BS 32   | `sort_cubes@a7c156de` (`0:68` / `68:77`) | `Orellius/so101_exp002_c3_nano_l24_joint_lora_bs32`            |
| `exp003` | `exp003_c3_nano_l24_joint_dora_bs32.sh`            | Cosmos 3 Nano (L24/36) + Joint DoRA (`gen_only`), BS 32   | `sort_cubes@a7c156de` (`0:68` / `68:77`) | `Orellius/so101_exp003_c3_nano_l24_joint_dora_bs32`            |
| `exp004` | `exp004_flux3_sort_cubes_lora_w1.sh`               | FLUX 3 Action SO-101 LoRA (W1 duplicated wrist) + P1 eval | `sort_cubes@a7c156de` (`0:68` / `68:77`) | `Orellius/so101_exp004_flux3_sort_cubes_lora_w1`               |
| `exp005` | `exp005_flux3_sort_cubes_dora_w1.sh`               | FLUX 3 Action SO-101 DoRA (W1 duplicated wrist) + P1 eval | `sort_cubes@a7c156de` (`0:68` / `68:77`) | `Orellius/so101_exp005_flux3_sort_cubes_dora_w1`               |
| `exp006` | `exp006_smolvla_sort_cubes_clean_heldout.sh`       | SmolVLA 450M clean held-out baseline (50k steps, BS 32)   | `sort_cubes@a7c156de` (`0:68` / `68:77`) | `Orellius/so101_exp006_smolvla_sort_cubes_clean_heldout`       |
| `exp007` | `exp007_c3_edge_direct_action_lora_3500.sh`        | Cosmos 3 Edge (L20, 8GB-fit) Direct 1-Stage Action-LoRA   | `sort_cubes@a7c156de` (`0:68` / `68:77`) | `Orellius/so101_exp007_c3_edge_direct_action_lora_3500`        |
| `exp008` | `exp008_c3_edge_video_pred_lora.sh`                | Cosmos 3 Edge Stage 1 Video-Prediction LoRA (`gen_only`)  | `sort_cubes@a7c156de` (`0:68` / `68:77`) | _(Stage 1 adapter for `exp009` / `exp010`)_                    |
| `exp009` | `exp009_c3_edge_vidlora_then_frozen_action.sh`     | Cosmos 3 Edge `exp008` Video-LoRA (frozen) -> SmolExpert  | `sort_cubes@a7c156de` (`0:68` / `68:77`) | `Orellius/so101_exp009_c3_edge_vidlora_then_frozen_action`     |
| `exp010` | `exp010_c3_edge_vidlora_then_joint_action_lora.sh` | Cosmos 3 Edge `exp008` Video-LoRA -> Joint Action-LoRA    | `sort_cubes@a7c156de` (`0:68` / `68:77`) | `Orellius/so101_exp010_c3_edge_vidlora_then_joint_action_lora` |
| `exp011` | `exp011_c3_edge_policy_droid_joint_lora.sh`        | Cosmos 3 Edge Policy-DROID (L20, 8 GB-fit) + Joint LoRA   | `sort_cubes@a7c156de` (`0:68` / `68:77`) | `Orellius/so101_exp011_c3_edge_policy_droid_joint_lora`        |
| `exp012` | `exp012_c3_nano_policy_droid_joint_lora.sh`        | Cosmos 3 Nano Policy-DROID (L24/36, RPC) + Joint LoRA     | `sort_cubes@a7c156de` (`0:68` / `68:77`) | `Orellius/so101_exp012_c3_nano_policy_droid_joint_lora`        |
| `exp013` | `exp013_c3_super_l42_nf4_qlora.sh`                 | Cosmos 3 Super 60B (L42/64, 4-bit NF4 QLoRA, 4090 RPC)    | `sort_cubes@a7c156de` (`0:68` / `68:77`) | `Orellius/so101_exp013_c3_super_l42_nf4_qlora`                 |
