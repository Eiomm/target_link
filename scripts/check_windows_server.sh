#!/usr/bin/env bash
# Use the server's existing environment. No install, overwrite, or YARN submit.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Python: explicit PYTHON wins; else the pod's qwen12 env if present (it has
# pyspark/pyarrow/numpy/torch/yaml); else plain python3.
QWEN12=/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12/bin/python
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$QWEN12" ]; then PYTHON="$QWEN12"; else PYTHON=python3; fi
fi
: "${SMOKE_OUT:?set a NEW local output directory for synthetic smoke artifacts}"
"$PYTHON" -c 'import pyspark, pyarrow, numpy, torch, pytest, yaml; print("Python dependencies available")'
java -version
export PYSPARK_PYTHON="$PYTHON"
export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-127.0.0.1}"
# pyspark local mode launches its JVM with a 1g heap by default; the window
# tests compile many distinct Spark SQL plans and GC-thrash at 1g (verified on
# this pod: ParOldGen 99%, suite stalled >10min while a single test alone
# passes). Raise the heap BEFORE any SparkSession starts.
export PYSPARK_SUBMIT_ARGS="--driver-memory ${DRIVER_MEMORY:-4g} pyspark-shell"
# The pod exports SPARK_HOME=/usr/local/spark-current (an older cluster JVM);
# pip pyspark must launch its own bundled spark-submit or SparkSession startup
# dies with "Constructor SparkSession([SparkContext, HashMap]) does not exist"
# (verified on this pod). Same convention as submit_*_yarn.sh local mode.
unset SPARK_HOME
# keep shuffle spill off the 20G-quota root disk when a scratch dir exists
if [ -z "${SPARK_LOCAL_DIRS:-}" ]; then
  for d in /nfs/dataset-ofs-494-1/project/user/junao/sparktmp; do
    [ -d "$d" ] && export SPARK_LOCAL_DIRS="$d" && break
  done
fi
"$PYTHON" -m pytest tests/test_windows.py -q
"$PYTHON" tools/smoke_windows.py --out "$SMOKE_OUT"
