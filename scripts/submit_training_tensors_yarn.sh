#!/usr/bin/env bash
# Convert existing observations_v2 on YARN; no local data conversion.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
MODE="${MODE:-dry}"
[[ "$MODE" == dry || "$MODE" == check || "$MODE" == yarn || "$MODE" == preflight ]] || {
  echo 'MODE must be dry, check, yarn or preflight' >&2; exit 2;
}
HDFS_BASE="${HDFS_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/junao/target_link}"
SOURCE="${SOURCE:-$HDFS_BASE/corpus_v1}"
QUEUE="${QUEUE:-root.xinsi_yanfaerzu_default}"
SPARK_SUBMIT="${SPARK_SUBMIT:-}"
MINIPY3_TGZ="${MINIPY3_TGZ:-hdfs://DClusterNmg3/user/bigdata-dp/common-env/minipy3.tgz}"
PARALLELISM="${PARALLELISM:-20}"
M_MAX="${M_MAX:-64}"
DATA_SEED="${DATA_SEED:-20260921}"
OUT_DIR="${OUT_DIR:-$HDFS_BASE/training_tensors_v1_m${M_MAX}_seed${DATA_SEED}}"
BUCKETS="${BUCKETS:-128}"
TRAIN_DAYS="${TRAIN_DAYS:-20260817 20260818 20260819 20260820 20260821 20260822}"
VAL_DAYS="${VAL_DAYS:-20260823}"
# This host-only path is used solely to discover a local spark-submit launcher.
# It is never propagated to YARN workers unless the caller explicitly sets
# TENSOR_PYTHON to that path.
LOCAL_SPARK_DISCOVERY_PYTHON="${SPARK_DISCOVERY_PYTHON:-/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12/bin/python}"
TENSOR_PYTHON="${TENSOR_PYTHON:-}"
HDFS_BIN="${HDFS_BIN:-}"
[[ "$SOURCE" == hdfs://* && "$OUT_DIR" == hdfs://* ]] || { echo 'SOURCE and OUT_DIR must be HDFS URIs' >&2; exit 2; }
[[ "$PARALLELISM" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid PARALLELISM' >&2; exit 2; }
# Prefer an explicitly selected launcher, then the host's configured installation.
if [[ -z "$SPARK_SUBMIT" ]]; then
  if [[ -n "${SPARK_HOME:-}" && -x "$SPARK_HOME/bin/spark-submit" ]]; then
    SPARK_SUBMIT="$SPARK_HOME/bin/spark-submit"
  elif command -v spark-submit >/dev/null 2>&1; then
    SPARK_SUBMIT="$(command -v spark-submit)"
  elif [[ -x /usr/local/spark-current/bin/spark-submit ]]; then
    SPARK_SUBMIT=/usr/local/spark-current/bin/spark-submit
  elif [[ -x "$LOCAL_SPARK_DISCOVERY_PYTHON" ]]; then
    # Locate the bundled distribution without importing PySpark or starting Java.
    SPARK_SUBMIT="$("$LOCAL_SPARK_DISCOVERY_PYTHON" -c 'import importlib.util,pathlib; s=importlib.util.find_spec("pyspark"); print(pathlib.Path(s.origin).parent/"bin/spark-submit" if s else "")')"
  fi
fi
if [[ -n "$SPARK_SUBMIT" && "$SPARK_SUBMIT" != */* ]]; then
  SPARK_SUBMIT="$(command -v "$SPARK_SUBMIT" || true)"
fi
# A stale SPARK_HOME otherwise redirects a valid launcher to the wrong jars.
if [[ -n "$SPARK_SUBMIT" && -x "$SPARK_SUBMIT" ]]; then
  SELECTED_SPARK_ROOT="$(cd "$(dirname "$(readlink -f "$SPARK_SUBMIT")")/.." && pwd)"
  if [[ -d "$SELECTED_SPARK_ROOT/jars" ]]; then
    export SPARK_HOME="$SELECTED_SPARK_ROOT"
  fi
fi
SCHEDULER_ARCHIVE="$MINIPY3_TGZ#minipy3"
SCHEDULER_PYTHON=./minipy3/minipy3/bin/python
CUSTOM_SCHEDULER=0
if [[ -n "${SPARK_PYTHON_ARCHIVE:-}" ]]; then
  SCHEDULER_ARCHIVE="$SPARK_PYTHON_ARCHIVE#spark_python"
  SCHEDULER_PYTHON="${SPARK_PYTHON_REL:-./spark_python/bin/python}"
  CUSTOM_SCHEDULER=1
fi
# Spark itself keeps the known-good scheduler interpreter. The converter runs
# as a subprocess on executors and must have its own supplied environment.
if [[ -n "${TENSOR_ENV_ARCHIVE:-}" ]]; then
  CONVERTER_PYTHON="${TENSOR_ENV_PYTHON:-./tensor_env/bin/python}"
elif [[ -n "$TENSOR_PYTHON" ]]; then
  CONVERTER_PYTHON="$TENSOR_PYTHON"
else
  # Dry/check remain inspectable, but actual YARN submission is rejected.
  CONVERTER_PYTHON=./tensor_env/bin/python
fi
if [[ "$MODE" == yarn || "$MODE" == preflight ]]; then
  if [[ -z "${TENSOR_ENV_ARCHIVE:-}" && -z "$TENSOR_PYTHON" ]]; then
    echo 'YARN submission requires TENSOR_ENV_ARCHIVE (preferred) or an explicit TENSOR_PYTHON; no job was submitted.' >&2
    exit 2
  fi
fi
if [[ "$MODE" == yarn || "$MODE" == preflight || "$MODE" == check ]]; then
  [[ -n "$SPARK_SUBMIT" && -x "$SPARK_SUBMIT" ]] || {
    echo 'No spark-submit found. Set SPARK_SUBMIT to an installed launcher or use the YARN submission host.' >&2
    exit 2
  }
  SPARK_VERSION_OUTPUT="$("$SPARK_SUBMIT" --version 2>&1)" || {
    printf '%s\n' "$SPARK_VERSION_OUTPUT" >&2; exit 2;
  }
  if [[ "$SPARK_VERSION_OUTPUT" =~ version[[:space:]]+([0-9]+)\.([0-9]+) ]]; then
    SPARK_MAJOR="${BASH_REMATCH[1]}"
    SPARK_MINOR="${BASH_REMATCH[2]}"
    echo "Spark launcher: $SPARK_SUBMIT (version $SPARK_MAJOR.$SPARK_MINOR)"
    if (( CUSTOM_SCHEDULER == 0 && (SPARK_MAJOR >= 4 || (SPARK_MAJOR == 3 && SPARK_MINOR >= 5)) )); then
      echo 'The default minipy3 archive uses Python 3.7, which is incompatible with this Spark version.' >&2
      if [[ "$MODE" == check ]]; then
        echo 'Check mode did not submit a job; provide SPARK_PYTHON_ARCHIVE before YARN submission.' >&2
      else
        echo 'Set SPARK_PYTHON_ARCHIVE (and optionally SPARK_PYTHON_REL) to a compatible scheduler Python environment. No job was submitted.' >&2
        exit 2
      fi
    fi
  else
    echo 'Unable to determine Spark version; refusing unchecked submission.' >&2; exit 2
  fi
  if [[ "$MODE" == check ]]; then
    if [[ -z "${TENSOR_ENV_ARCHIVE:-}" && -z "$TENSOR_PYTHON" ]]; then
      echo 'Launcher checks passed; no worker tensor environment was supplied, so cluster access and worker dependencies are not verified.'
    else
      echo 'Launcher and known Python compatibility checks passed; cluster access and worker dependencies are not verified.'
    fi
    exit 0
  fi
  : "${HADOOP_USER_NAME:?HADOOP_USER_NAME not set}"
fi
[[ -n "$SPARK_SUBMIT" ]] || SPARK_SUBMIT=spark-submit
PACKAGE_DIR="$(mktemp -d)"
trap 'rm -rf -- "$PACKAGE_DIR"' EXIT
# Bundle only the converter and its code dependencies, never data or environments.
"${PACKAGE_PYTHON:-python3}" - "$REPO" "$PACKAGE_DIR/tensor_code.zip" <<'PY'
from pathlib import Path
import sys, zipfile
root=Path(sys.argv[1])
files=['tools/prepare_tensors_yarn.py','experiments/__init__.py',
       'experiments/trajectory_mlp_v1/__init__.py',
       'experiments/trajectory_mlp_v1/data.py','experiments/trajectory_mlp_v1/prepared.py',
       'experiments/trajectory_mlp_v1/tensor_corpus.py',
       'experiments/trajectory_mlp_v1/tools/prepare_tensors.py']
with zipfile.ZipFile(sys.argv[2],'w',zipfile.ZIP_DEFLATED) as archive:
    for name in files:
        if (root/name).exists():
            archive.write(root/name,name)
        elif name.endswith('/__init__.py'):
            archive.writestr(name,'')
        else:
            raise FileNotFoundError(name)
    for name in ['tools/__init__.py','experiments/trajectory_mlp_v1/tools/__init__.py']:
        archive.writestr(name,'')
PY
# Spark/YARN deduplicates resources by source URI. Use distinct files for the
# Python ZIP and extracted archive so localization preserves both resources.
cp "$PACKAGE_DIR/tensor_code.zip" "$PACKAGE_DIR/tensor_modules.zip"
read -r -a TRAIN_DAY_ARGS <<< "$TRAIN_DAYS"
read -r -a VAL_DAY_ARGS <<< "$VAL_DAYS"
DIST_ARCHIVES="$SCHEDULER_ARCHIVE,$PACKAGE_DIR/tensor_code.zip#tensor_code"
if [[ -n "${TENSOR_ENV_ARCHIVE:-}" ]]; then
  DIST_ARCHIVES="$DIST_ARCHIVES,$TENSOR_ENV_ARCHIVE#tensor_env"
fi
DIST_ARCHIVE_CONF=()
if [[ -n "$DIST_ARCHIVES" ]]; then
  DIST_ARCHIVE_CONF=(--conf "spark.yarn.dist.archives=$DIST_ARCHIVES")
fi
SUBMIT_EXECUTORS="$PARALLELISM"
if [[ "$MODE" == preflight ]]; then
  SUBMIT_EXECUTORS=1
fi
CMD=("$SPARK_SUBMIT" --master yarn --deploy-mode cluster --queue "$QUEUE"
  --name prepare_training_tensors_v1 --py-files "$PACKAGE_DIR/tensor_modules.zip"
  "${DIST_ARCHIVE_CONF[@]}"
  --conf "spark.pyspark.python=$SCHEDULER_PYTHON"
  --conf "spark.pyspark.driver.python=$SCHEDULER_PYTHON"
  --conf spark.yarn.appMasterEnv.PYTHONUNBUFFERED=1
  --conf spark.executorEnv.PYTHONUNBUFFERED=1
  --conf spark.dynamicAllocation.enabled=false --num-executors "$SUBMIT_EXECUTORS"
  --executor-cores 1 --executor-memory "${EXECUTOR_MEMORY:-2g}"
  --conf "spark.executor.memoryOverhead=${EXECUTOR_OVERHEAD:-12288}"
  --driver-memory "${DRIVER_MEMORY:-4g}" --conf spark.speculation=false
  --conf spark.yarn.submit.waitAppCompletion=true
  tools/prepare_tensors_yarn.py --source "$SOURCE" --out "$OUT_DIR"
  --train-days "${TRAIN_DAY_ARGS[@]}" --val-days "${VAL_DAY_ARGS[@]}"
  --parallelism "$PARALLELISM" --buckets "$BUCKETS" --m-max "$M_MAX" --seed "$DATA_SEED"
  --worker-python "$CONVERTER_PYTHON")
if [[ -n "$HDFS_BIN" ]]; then
  CMD+=(--hdfs-bin "$HDFS_BIN")
fi
if [[ "$MODE" == preflight ]]; then
  CMD+=(--preflight-only)
fi
printf '%q ' "${CMD[@]}"
printf '\n'
if [[ "$MODE" == dry ]]; then
  if [[ -z "${TENSOR_ENV_ARCHIVE:-}" && -z "$TENSOR_PYTHON" ]]; then
    echo 'Dry run warning: no worker tensor environment was supplied; the command is not validated for worker dependencies.'
  fi
  echo 'Dry run: command assembled; no YARN submission or data conversion.'
else
  "${CMD[@]}"
fi
