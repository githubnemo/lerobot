#!/usr/bin/env bash
# Unattended night pipeline: sigma-4 validation cache, online randomized-sigma
# training with crash resume, then the four-way physical action RMSE comparison.
# Every stage is guarded: a failing stage is logged and the next independent
# stage still runs. Only one GPU job is active at any time.
set -uo pipefail

cd /home/anton/lerobot-video-vam || exit 1
# shellcheck disable=SC1091
source scripts/video_vam/cosmos_cuda_env.sh
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1

PY=/home/anton/lerobot-video-vam/.venv/bin/python
CACHE=/home/anton/.cache/video-vam
BACKBONE=$CACHE/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt
TOKENIZER=$CACHE/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth
PROMPT=$CACHE/prompt-embeddings/cube-out-of-box-t5-11b.safetensors
DATASET=$CACHE/cube-out-of-box-dataset
TRAIN_MANIFEST=$CACHE/cosmos-rehearsal-stride20/manifest.json
SPLIT=$CACHE/splits/rehearsal-stride20.json
SIGMA4_DIR=$CACHE/cosmos-rehearsal-stride20-sigma4
SIGMA4_MANIFEST=$SIGMA4_DIR/manifest.json
RUN=$CACHE/runs/cosmos2b-online-randsigma-eb8
LOGDIR=$CACHE/runs/night-20260819
SMOLVLA=/home/anton/lerobot-video-vam/outputs/train/cube_out_of_box_il_smolvla_heldout_0_31_20260818_retry4/checkpoints/005000/pretrained_model

EVAL_SIGMA=4.0
MICRO_BATCH=1
ACCUM=8
MAX_STEPS=950
WARMUP=250
VAL_EVERY=50
SAVE_EVERY=50
PATIENCE=40
GRAD_CLIP=10.0

# Hard wall-clock deadline for the whole night, so the comparison still runs.
DEADLINE_EPOCH=$(date -d "today 08:40" +%s)
if [ "$DEADLINE_EPOCH" -le "$(date +%s)" ]; then
  DEADLINE_EPOCH=$(date -d "tomorrow 08:40" +%s)
fi

mkdir -p "$LOGDIR"
log() { echo "[$(date -Is)] $*" | tee -a "$LOGDIR/driver.log"; }

log "night pipeline start; deadline $(date -d "@$DEADLINE_EPOCH" -Is)"

# ---------------------------------------------------------------- stage 1
CACHE_OK=0
if [ -f "$SIGMA4_MANIFEST" ] && [ -f "$SIGMA4_DIR/.complete" ]; then
  log "stage 1 skipped: sigma-$EVAL_SIGMA cache already complete"
  CACHE_OK=1
else
  for attempt in 1 2 3; do
    log "stage 1 attempt $attempt: building sigma-$EVAL_SIGMA context cache"
    # shellcheck disable=SC2046
    $PY scripts/video_vam/build_cosmos_feature_cache.py \
      --root "$DATASET" \
      --episodes $(seq 0 39) \
      --stride 20 \
      --sigma "$EVAL_SIGMA" \
      --prompt "$PROMPT" \
      --checkpoint "$BACKBONE" \
      --tokenizer "$TOKENIZER" \
      --output-dir "$SIGMA4_DIR" \
      --seed 0 \
      --resume >>"$LOGDIR/cache_sigma4.log" 2>&1
    if [ $? -eq 0 ]; then
      touch "$SIGMA4_DIR/.complete"
      CACHE_OK=1
      log "stage 1 done"
      break
    fi
    log "stage 1 attempt $attempt failed; see cache_sigma4.log"
    sleep 30
  done
fi

VAL_ARGS=()
if [ "$CACHE_OK" -eq 1 ]; then
  VAL_ARGS=(--val-manifest "$SIGMA4_MANIFEST")
else
  log "WARNING: sigma-$EVAL_SIGMA cache unavailable; validating on sigma-10 contexts is inconsistent, so training will validate on the training manifest and its numbers are not comparable"
fi

# ---------------------------------------------------------------- stage 2
mkdir -p "$RUN"
TRAIN_OK=0
for attempt in $(seq 1 60); do
  NOW=$(date +%s)
  REMAIN=$(( DEADLINE_EPOCH - NOW ))
  if [ "$REMAIN" -le 600 ]; then
    log "stage 2 stopping: less than 10 minutes to deadline"
    break
  fi
  MAX_HOURS=$(awk -v r="$REMAIN" 'BEGIN { printf "%.4f", (r - 300) / 3600 }')
  if [ -f "$RUN/last.safetensors" ]; then
    MODE=(--resume)
  else
    MODE=(--overwrite)
  fi
  log "stage 2 attempt $attempt (${MODE[0]}), max_hours=$MAX_HOURS"
  $PY scripts/video_vam/train_cosmos_world2action.py \
    --manifest "$TRAIN_MANIFEST" \
    --split "$SPLIT" \
    --output-dir "$RUN" \
    --context-mode online \
    --dataset-root "$DATASET" \
    --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" \
    --backbone-checkpoint "$BACKBONE" \
    --backbone-identity cosmos-predict2-2b-online-randsigma \
    "${VAL_ARGS[@]}" \
    --eval-sigma "$EVAL_SIGMA" \
    --select-metric rmse \
    --batch-size "$MICRO_BATCH" \
    --grad-accum-steps "$ACCUM" \
    --max-steps "$MAX_STEPS" \
    --max-hours "$MAX_HOURS" \
    --warmup-steps "$WARMUP" \
    --val-every "$VAL_EVERY" \
    --save-every "$SAVE_EVERY" \
    --patience "$PATIENCE" \
    --lr 1e-4 \
    --weight-decay 0.1 \
    --grad-clip "$GRAD_CLIP" \
    --seed 0 \
    --wandb-project video-vam-world2action \
    "${MODE[@]}" >>"$LOGDIR/train.log" 2>&1
  RC=$?
  if [ "$RC" -eq 0 ]; then
    TRAIN_OK=1
    log "stage 2 finished cleanly"
    break
  fi
  log "stage 2 attempt $attempt exited with $RC; retrying with --resume after 30s"
  sleep 30
done

# ---------------------------------------------------------------- stage 3
if [ ! -f "$RUN/best.safetensors" ]; then
  log "stage 3 skipped: no best checkpoint was produced"
  exit 1
fi
if [ "$CACHE_OK" -ne 1 ]; then
  log "stage 3 skipped: the comparison requires sigma-$EVAL_SIGMA contexts"
  exit 1
fi
for attempt in 1 2 3; do
  log "stage 3 attempt $attempt: four-way RMSE comparison at sigma $EVAL_SIGMA"
  $PY scripts/video_vam/evaluate_action_rmse.py \
    --manifest "$SIGMA4_MANIFEST" \
    --split "$SPLIT" \
    --dataset-root "$DATASET" \
    --output "$LOGDIR/action_rmse_comparison.json" \
    --vam-checkpoint "$RUN/best.safetensors" \
    --vam-normalizer "$RUN/normalizer.safetensors" \
    --vam-sigma "$EVAL_SIGMA" \
    --smolvla-checkpoint "$SMOLVLA" \
    --device cuda \
    --batch-size 1 \
    --seed 0 \
    --overwrite >>"$LOGDIR/eval.log" 2>&1
  if [ $? -eq 0 ]; then
    log "stage 3 done: $LOGDIR/action_rmse_comparison.json"
    tail -3 "$LOGDIR/eval.log" | tee -a "$LOGDIR/driver.log"
    exit 0
  fi
  log "stage 3 attempt $attempt failed; see eval.log"
  sleep 30
done
log "stage 3 failed after 3 attempts"
exit 1
