#!/bin/bash
set -euo pipefail
YOLO_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_INTERPRETER_OVERRIDE="${YOLO_PYTHON:-}"
# 此处只加载本机路径，分割进程不加载云端凭据。
if [ -f "$YOLO_PROJECT_ROOT/config/local.env" ]; then
  source "$YOLO_PROJECT_ROOT/config/local.env"
fi
if [ -n "$YOLO_INTERPRETER_OVERRIDE" ]; then YOLO_PYTHON="$YOLO_INTERPRETER_OVERRIDE"; fi
unset YOLO_INTERPRETER_OVERRIDE
if [ -z "${YOLO_PYTHON:-}" ]; then
  for yolo_candidate in "$HOME/anaconda3/envs/yolo/bin/python" "$HOME/miniconda3/envs/yolo/bin/python"; do
    if [ -x "$yolo_candidate" ]; then YOLO_PYTHON="$yolo_candidate"; break; fi
  done
fi
if [ -z "${YOLO_PYTHON:-}" ] || [ ! -x "$YOLO_PYTHON" ]; then
  echo '缺少 yolo 环境解释器。运行 bash scripts/setup_local.sh；或设置 YOLO_PYTHON 的绝对路径。' >&2
  exit 2
fi
YOLO_ENV_BIN="$(cd -- "$(dirname -- "$YOLO_PYTHON")" && pwd)"
unset PYTHONPATH PYTHONHOME AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH LD_LIBRARY_PATH QT_QPA_PLATFORM_PLUGIN_PATH
unset DASHSCOPE_API_KEY CLOUD_AI_API_KEY OPENAI_API_KEY
export PYTHONNOUSERSITE=1 YOLO_AUTOINSTALL=false
export YOLO_CONFIG_DIR="$YOLO_PROJECT_ROOT/work/yolo/config"
export PATH="$YOLO_ENV_BIN:/usr/local/bin:/usr/bin:/bin"
mkdir -p "$YOLO_PROJECT_ROOT/work/yolo/assets" "$YOLO_CONFIG_DIR"
cd "$YOLO_PROJECT_ROOT/work/yolo/assets"
if [ "${1:-}" = python ]; then
  shift
  exec "$YOLO_PYTHON" "$@"
fi
if [ ! -x "$YOLO_ENV_BIN/yolo" ]; then
  echo 'yolo 环境缺少 Ultralytics CLI；请根据 config/environment.yml 安装。' >&2
  exit 2
fi
exec "$YOLO_ENV_BIN/yolo" "$@"
