#!/bin/bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$EXPLORE_ROOT/results/saved_maps"
exec ros2 run nav2_map_server map_saver_cli -f "$EXPLORE_ROOT/results/saved_maps/online_home" --ros-args -p use_sim_time:=true -p save_map_timeout:=10.0
