#!/bin/bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
cd "$EXPLORE_ROOT"
exec ros2 launch home_explore bringup.launch.py results_dir:="$EXPLORE_ROOT/results" "$@"
