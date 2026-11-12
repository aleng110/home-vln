#!/bin/bash
set -eo pipefail
source "$(dirname "$0")/env.sh"
cd "$EXPLORE_ROOT"
export PYTHONPYCACHEPREFIX=/tmp/home_explore_pycache
/usr/bin/python3 -m compileall -q src/home_explore scripts
PYTHONPATH="$EXPLORE_ROOT/src/home_explore:${PYTHONPATH:-}" /usr/bin/python3 -m unittest discover -s tests -v
