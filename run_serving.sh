#!/usr/bin/env bash
# Serve the two-stage recommender: uvicorn on port 8000.
# Requires artifacts/ from a completed run_all.sh first.
# Uses the repo's own .venv when present (see README setup), else python3.
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi

"$PY" -m uvicorn serving.app:app --host 0.0.0.0 --port 8000
