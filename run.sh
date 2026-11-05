#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mirror_python="${REVO3_PYTHON:-.venv/bin/python}"
if ! command -v "$mirror_python" >/dev/null 2>&1; then
  echo 'Run ./setup.sh first, or set REVO3_PYTHON to a Python environment with requirements.txt installed.' >&2
  exit 1
fi
exec "$mirror_python" live_viewer.py "$@"
