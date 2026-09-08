#!/usr/bin/env bash
# Use the server's existing environment. No install, overwrite, or YARN submit.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PYTHON="${PYTHON:-python3}"
: "${SMOKE_OUT:?set a NEW local output directory for synthetic smoke artifacts}"
"$PYTHON" -c 'import pyspark, pyarrow, numpy, torch, pytest, yaml; print("Python dependencies available")'
java -version
export PYSPARK_PYTHON="$PYTHON"
export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-127.0.0.1}"
"$PYTHON" -m pytest tests/test_windows.py -q
"$PYTHON" tools/smoke_windows.py --out "$SMOKE_OUT"
