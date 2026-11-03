#!/usr/bin/env bash
# End-to-end run: download -> train -> evaluate -> sample recommendations.
# Uses the repo's own .venv when present (see README setup), else python3.
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi

"$PY" download.py
"$PY" train.py
"$PY" evaluate.py
"$PY" recommend.py --user_id 1 --top_n 10
