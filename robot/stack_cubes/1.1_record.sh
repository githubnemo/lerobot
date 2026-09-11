#!/usr/bin/env bash
# Record stack_cubes on one setup. Run once per arm; the second run appends.
#
#   ./robot/stack_cubes/1.1_record.sh --setup boxed --num-episodes 20
#   ./robot/stack_cubes/1.1_record.sh --setup free --num-episodes 20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../record_teleop.sh" --task stack_cubes "$@"
