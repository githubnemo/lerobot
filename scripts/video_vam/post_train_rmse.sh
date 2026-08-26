#!/usr/bin/env bash
# Convention: every native-policy training runner calls this helper immediately after
# successful training so the frozen-protocol RMSE and append-only evaluation log stay paired.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  printf 'usage: %s <backend flag> <checkpoint path> <output json path>\n' "$0" >&2
  exit 2
fi

backend_flag="$1"
checkpoint="$2"
output="$3"
case "$backend_flag" in
  --fastwam-checkpoint|--smolvla-checkpoint|--vla-jepa-checkpoint) ;;
  *)
    printf 'unsupported backend flag: %s\n' "$backend_flag" >&2
    exit 2
    ;;
esac

repo_root=/home/anton/lerobot-video-vam
cd "$repo_root"
python_bin="$repo_root/.venv/bin/python"
eval_log=/home/anton/.cache/video-vam/runs/EVAL_LOG.md
mkdir -p "$(dirname "$output")" "$(dirname "$eval_log")"

"$python_bin" scripts/video_vam/evaluate_action_rmse.py \
  --manifest /home/anton/.cache/video-vam/cosmos-rehearsal-stride20-sigma80/manifest.json \
  --split /home/anton/.cache/video-vam/splits/rehearsal-stride20.json \
  --dataset-root /home/anton/.cache/video-vam/cube-out-of-box-dataset \
  --device cuda \
  --batch-size 1 \
  --seed 0 \
  --overwrite \
  "$backend_flag" "$checkpoint" \
  --output "$output"

backend="${backend_flag#--}"
backend="${backend%-checkpoint}"
backend="${backend//-/_}"
aggregate_rmse="$($python_bin - "$output" "$backend" <<'PYTHON'
import json
import sys

output_path, backend = sys.argv[1:]
payload = json.loads(open(output_path).read())
print(payload["backends"][backend]["aggregate_rmse_deg"])
PYTHON
)"
printf '%s\tbackend=%s\tcheckpoint=%s\taggregate_rmse_deg=%s\toutput=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$backend" "$checkpoint" "$aggregate_rmse" "$output" \
  >> "$eval_log"
printf 'RMSE backend=%s aggregate_rmse_deg=%s output=%s\n' "$backend" "$aggregate_rmse" "$output"
