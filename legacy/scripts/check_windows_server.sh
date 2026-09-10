#!/usr/bin/env bash
# Use the server's existing environment. No install, overwrite, or YARN submit.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# Python: explicit PYTHON wins; else the pod's qwen12 env if present (it has
# pyspark/pyarrow/numpy/torch/yaml); else plain python3.
QWEN12=/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12/bin/python
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$QWEN12" ]; then PYTHON="$QWEN12"; else PYTHON=python3; fi
fi
# The launcher only accepts a script, so this must run with no environment set.
# SMOKE_OUT stays overridable; unset means "pick a fresh timestamped directory".
# The directory must not exist: smoke_windows.py refuses to reuse an output dir,
# and a failed intermediate write must never be mistaken for a checked artifact.
if [ -z "${SMOKE_OUT:-}" ]; then
  SMOKE_OUT="runtime/windows_smoke_$(date -u +%Y%m%d_%H%M%S)"
  echo "[check_windows] SMOKE_OUT unset; writing to $SMOKE_OUT"
fi
if [ -e "$SMOKE_OUT" ]; then
  echo "SMOKE_OUT already exists; choose a new directory: $SMOKE_OUT" >&2
  exit 1
fi
"$PYTHON" -c 'import pyspark, pyarrow, numpy, torch, pytest, yaml; print("Python dependencies available")'
java -version
export PYSPARK_PYTHON="$PYTHON"
# --- Spark driver bind address -------------------------------------------------
# Local mode keeps driver and executor in one JVM, so loopback is the natural
# bind address. Never trust an inherited SPARK_LOCAL_IP: the job launcher
# injects SPARK_LOCAL_HOSTNAME (and may inject SPARK_LOCAL_IP) with a pod or
# node address that is not always assigned inside the container netns, which
# fails as
#   BindException: Cannot assign requested address: Service 'sparkDriver'
# Probe what is actually bindable, prefer loopback, and log what was inherited.
INHERITED_SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-<unset>}"
SPARK_LOCAL_IP="$("$PYTHON" -c '
import socket, sys


def bindable(addr):
    if not addr:
        return False
    probe = socket.socket()
    try:
        probe.bind((addr, 0))
        return True
    except OSError:
        return False
    finally:
        probe.close()


candidates = ["127.0.0.1"]
try:
    candidates.append(socket.gethostbyname(socket.gethostname()))
except OSError:
    pass
for candidate in candidates:
    if bindable(candidate):
        print(candidate)
        sys.exit(0)
sys.exit("no bindable IPv4 address for the Spark driver")
')"
export SPARK_LOCAL_IP
echo "[check_windows] hostname=$(hostname) hostname-i=$(hostname -i 2>/dev/null || true)" \
     "SPARK_LOCAL_HOSTNAME=${SPARK_LOCAL_HOSTNAME:-<unset>}" \
     "SPARK_LOCAL_IP(inherited)=$INHERITED_SPARK_LOCAL_IP" \
     "-> bind=$SPARK_LOCAL_IP"
# pyspark local mode launches its JVM with a 1g heap by default; the window
# tests compile many distinct Spark SQL plans and GC-thrash at 1g (verified on
# this pod: ParOldGen 99%, suite stalled >10min while a single test alone
# passes). Raise the heap BEFORE any SparkSession starts.
# Bind explicitly: an inherited spark.driver.bindAddress/host (platform
# spark-defaults.conf or JAVA_TOOL_OPTIONS) would otherwise beat SPARK_LOCAL_IP.
export PYSPARK_SUBMIT_ARGS="--driver-memory ${DRIVER_MEMORY:-4g} --conf spark.driver.bindAddress=$SPARK_LOCAL_IP --conf spark.driver.host=$SPARK_LOCAL_IP pyspark-shell"
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
"$PYTHON" -m pytest legacy/tests/test_windows.py -q
"$PYTHON" legacy/tools/smoke_windows.py --out "$SMOKE_OUT"
