#!/bin/bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
cd "$EXPLORE_ROOT"
colcon build --base-paths src --symlink-install --parallel-workers 2 --cmake-args -DBUILD_TESTING=OFF
