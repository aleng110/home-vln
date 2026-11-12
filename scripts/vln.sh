#!/bin/bash
source "$(dirname "$0")/env.sh"
exec ros2 run home_explore cli "$@"
