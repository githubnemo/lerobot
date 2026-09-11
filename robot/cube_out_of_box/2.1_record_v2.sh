#!/usr/bin/env bash
# Append teleop episodes to cube_out_of_box v2 (new data only).
# After both setups, merge with v1 via 2.2_merge_v1_v2.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../record_teleop.sh" --task cube_v2 "$@"
