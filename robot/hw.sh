# Shared hardware presets for the two SO-101 followers.
# Source this file; then call `load_setup boxed` or `load_setup free`.
#
#   boxed  — arm in the cardboard box (shabby). Original cube-out-of-box rig.
#   free   — freely standing arm (viola).
#
# If those labels are swapped on the bench, pass --follower-port / --leader-port
# (or edit the defaults below). Camera index often changes when USB is replugged;
# override with --camera-index.

boxed_follower_port="${BOXED_FOLLOWER_PORT:-/dev/tty.usbmodem5A460820701}"
boxed_follower_id="${BOXED_FOLLOWER_ID:-shabby}"
# Leader defaults to the viola teleop arm; override if the boxed rig has its own.
boxed_leader_port="${BOXED_LEADER_PORT:-/dev/tty.usbmodem5A460845501}"
boxed_leader_id="${BOXED_LEADER_ID:-viola_leader}"
boxed_camera_index="${BOXED_CAMERA_INDEX:-1}"

free_follower_port="${FREE_FOLLOWER_PORT:-/dev/tty.usbmodem5A460846381}"
free_follower_id="${FREE_FOLLOWER_ID:-viola_real_follower}"
free_leader_port="${FREE_LEADER_PORT:-/dev/tty.usbmodem5A460845501}"
free_leader_id="${FREE_LEADER_ID:-viola_leader}"
free_camera_index="${FREE_CAMERA_INDEX:-0}"

load_setup() {
    local setup="${1:?setup name required (boxed|free)}"
    case "$setup" in
        boxed)
            FOLLOWER_PORT="$boxed_follower_port"
            FOLLOWER_ID="$boxed_follower_id"
            LEADER_PORT="$boxed_leader_port"
            LEADER_ID="$boxed_leader_id"
            CAMERA_INDEX="$boxed_camera_index"
            ;;
        free)
            FOLLOWER_PORT="$free_follower_port"
            FOLLOWER_ID="$free_follower_id"
            LEADER_PORT="$free_leader_port"
            LEADER_ID="$free_leader_id"
            CAMERA_INDEX="$free_camera_index"
            ;;
        *)
            echo "Unknown setup '$setup'. Use boxed or free." >&2
            return 2
            ;;
    esac
}
