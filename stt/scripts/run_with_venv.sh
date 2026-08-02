#!/usr/bin/env bash
# Run STT scripts with a working PYTHONPATH (uv/venv activate is flaky here).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export VIRTUAL_ENV="$ROOT/venv"
export PYTHONPATH="$ROOT/venv/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ROOT/venv/bin:$PATH"
PY="$ROOT/venv/lib/python3.11/../.." 
# Prefer the real cpython that the venv was built from
REAL_PY="$(readlink -f "$ROOT/venv/bin/python3.11")"
cd "$ROOT"
exec "$REAL_PY" "$@"
