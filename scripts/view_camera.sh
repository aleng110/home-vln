#!/bin/bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
case "${1:-}" in
  inspection)
    shift
    exec ros2 run home_explore viewer --ros-args -r __node:=vln_inspection_viewer -p image_topic:=/vln/inspection/image -p transient_local:=true "$@"
    ;;
  detections)
    shift
    exec ros2 run home_explore viewer --ros-args -r __node:=vln_detection_viewer -p image_topic:=/vln/perception/image "$@"
    ;;
  *) exec ros2 run home_explore viewer "$@" ;;
esac
