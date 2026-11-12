#!/bin/bash
# 在项目内配置 Humble TF2 MessageFilter 死锁修复库。
set -eo pipefail
VLN_TF2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec /usr/bin/python3 "$VLN_TF2_ROOT/scripts/tf2_overlay.py" setup "$VLN_TF2_ROOT/work/tf2fix"
