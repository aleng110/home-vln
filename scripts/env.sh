#!/bin/bash
# ROS 使用系统 Python；Conda 仅由 scripts/yolo.sh 加载。
EXPLORE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export EXPLORE_ROOT
VLN_ENV_YOLO_OVERRIDE="${YOLO_PYTHON:-}"
VLN_ENV_CREDENTIALS_OVERRIDE="${VLN_CREDENTIALS_FILE:-}"
if [ -f "$EXPLORE_ROOT/config/local.env" ]; then
  source "$EXPLORE_ROOT/config/local.env"
fi
if [ -n "$VLN_ENV_YOLO_OVERRIDE" ]; then YOLO_PYTHON="$VLN_ENV_YOLO_OVERRIDE"; fi
if [ -n "$VLN_ENV_CREDENTIALS_OVERRIDE" ]; then VLN_CREDENTIALS_FILE="$VLN_ENV_CREDENTIALS_OVERRIDE"; fi
unset VLN_ENV_YOLO_OVERRIDE VLN_ENV_CREDENTIALS_OVERRIDE
if [ ! -f /opt/ros/humble/setup.bash ]; then
  echo '缺少 ROS 2 Humble：请先安装 Ubuntu 22.04 的 ROS 2 Humble/Nav2/SLAM/Gazebo 11 依赖。' >&2
  return 1 2>/dev/null || exit 1
fi
export PATH=/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONHOME AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
source /opt/ros/humble/setup.bash
if [ -f "$EXPLORE_ROOT/install/local_setup.bash" ]; then
  source "$EXPLORE_ROOT/install/local_setup.bash"
fi
# 仅加载明确指定的本机凭据，不搜索旧项目。
if [ -f "$EXPLORE_ROOT/.env.local" ]; then
  set -a
  source "$EXPLORE_ROOT/.env.local"
  set +a
elif [ -n "${VLN_CREDENTIALS_FILE:-}" ]; then
  if [ ! -f "$VLN_CREDENTIALS_FILE" ]; then
    echo 'VLN_CREDENTIALS_FILE 指向的凭据文件不存在；请重新运行 scripts/setup_local.sh。' >&2
    return 1 2>/dev/null || exit 1
  fi
  set -a
  source "$VLN_CREDENTIALS_FILE"
  set +a
fi
if [ -f "$EXPLORE_ROOT/config/model.env" ]; then
  source "$EXPLORE_ROOT/config/model.env"
fi
export VLN_PROVIDER="${EXPLORE_PROVIDER:-${VLN_PROVIDER:-dashscope}}"
export VLN_MODEL="${EXPLORE_MODEL:-${VLN_MODEL:-qwen3-vl-plus}}"
export ROS_DOMAIN_ID="${EXPLORE_ROS_DOMAIN_ID:-65}"
export ROS_LOCALHOST_ONLY=1
export GAZEBO_MASTER_URI="${EXPLORE_GAZEBO_MASTER_URI:-http://127.0.0.1:11385}"
export GAZEBO_MODEL_DATABASE_URI=""
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$EXPLORE_ROOT/config/cyclonedds.xml"
if [ -n "${YOLO_PYTHON:-}" ]; then export YOLO_PYTHON; fi

# 最后设置 TF2 库路径，避免被本机或模型配置覆盖。
# 项目内 TF2 修复库用于避免 Humble MessageFilter 死锁。
VLN_ENV_TF2_LIB="$(/usr/bin/python3 "$EXPLORE_ROOT/scripts/tf2_overlay.py" check "$EXPLORE_ROOT/work/tf2fix")" || { return 1 2>/dev/null || exit 1; }
if [ -n "$VLN_ENV_TF2_LIB" ]; then export LD_LIBRARY_PATH="$VLN_ENV_TF2_LIB:${LD_LIBRARY_PATH:-}"; fi
unset VLN_ENV_TF2_LIB
