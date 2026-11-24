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

_(Experiments start at `exp001` for the 2026-11-24 RunPod 5090 sprint; historical pre-numbering runs are documented in `docs/video_vam_leaderboard.md` and `scripts/video_vam/presets/`.)_
