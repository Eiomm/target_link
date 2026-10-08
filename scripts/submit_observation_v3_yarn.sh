#!/usr/bin/env bash
# Build filtered ragged observations and fixed group indices on YARN.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
MODE="${MODE:-dry}"
[[ "$MODE" == dry || "$MODE" == yarn || "$MODE" == preflight ]] || { echo 'MODE must be dry, preflight or yarn' >&2; exit 2; }
HDFS_BASE="${HDFS_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/junao/target_link}"
SOURCE="${SOURCE:-$HDFS_BASE/corpus_v1}"
OUT_DIR="${OUT_DIR:-$HDFS_BASE/observationv3}"
QUEUE="${QUEUE:-root.xinsi_yanfaerzu_default}"
PARALLELISM="${PARALLELISM:-20}"
M_MAX="${M_MAX:-64}"
DATA_SEED="${DATA_SEED:-20260921}"
BUCKETS="${BUCKETS:-128}"
TRAIN_DAYS="${TRAIN_DAYS:-20260817 20260818 20260819 20260820 20260821 20260822}"
VAL_DAYS="${VAL_DAYS:-20260823}"
if [[ -z "${SPARK_SUBMIT:-}" ]]; then
  if [[ -x "$REPO/runtime/observationv3_yarn_setup/spark31_vendor_hadoop/bin/spark-submit" ]]; then
    SPARK_SUBMIT="$REPO/runtime/observationv3_yarn_setup/spark31_vendor_hadoop/bin/spark-submit"
  elif [[ -n "${SPARK_HOME:-}" && -x "$SPARK_HOME/bin/spark-submit" ]]; then
    SPARK_SUBMIT="$SPARK_HOME/bin/spark-submit"
  else
    SPARK_SUBMIT=/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12/lib/python3.12/site-packages/pyspark/bin/spark-submit
  fi
fi
V3_ENV_ARCHIVE="${V3_ENV_ARCHIVE:-}"
V3_PYTHON="${V3_PYTHON:-./v3_env/bin/python}"
SCHEDULER_ENV_ARCHIVE="${SCHEDULER_ENV_ARCHIVE:-hdfs://DClusterNmg3/user/bigdata-dp/common-env/minipy3.tgz}"
SCHEDULER_PYTHON="${SCHEDULER_PYTHON:-./scheduler_env/minipy3/bin/python}"
for value in PARALLELISM M_MAX BUCKETS; do
  [[ "${!value}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid $value" >&2; exit 2; }
done
(( M_MAX >= 3 && BUCKETS <= 128 )) || { echo 'Invalid group/bucket count' >&2; exit 2; }
[[ "$SOURCE" == hdfs://* && "$OUT_DIR" == hdfs://* ]] || { echo 'SOURCE and OUT_DIR must be HDFS URIs' >&2; exit 2; }
if [[ "$MODE" != dry ]]; then
  [[ "$V3_ENV_ARCHIVE" == hdfs://* ]] || { echo 'V3_ENV_ARCHIVE must point to the packaged Python environment in HDFS' >&2; exit 2; }
  [[ -x "$SPARK_SUBMIT" ]] || { echo "Missing Spark launcher: $SPARK_SUBMIT" >&2; exit 2; }
  : "${HADOOP_USER_NAME:?HADOOP_USER_NAME not set}"
fi
if [[ -x "$SPARK_SUBMIT" ]]; then
  export SPARK_HOME="$(cd "$(dirname "$(readlink -f "$SPARK_SUBMIT")")/.." && pwd)"
fi
PACKAGE_DIR="$(mktemp -d)"
trap 'rm -rf -- "$PACKAGE_DIR"' EXIT
"${PACKAGE_PYTHON:-python3}" - "$REPO" "$PACKAGE_DIR/observation_v3_code.zip" <<'PY'
from pathlib import Path
import sys, zipfile
root=Path(sys.argv[1])
files=['tools/prepare_observation_v3_yarn.py','tools/prepare_tensors_yarn.py',
       'tools/__init__.py',
       'trajectory_mae/__init__.py',
       'trajectory_mae/tools/__init__.py',
       'trajectory_mae/tools/prepare_observation_v3.py',
       'trajectory_mae/grouping.py','trajectory_mae/columns.py','trajectory_mae/data.py',
       'trajectory_mae/prepared.py',
       'trajectory_mae/observation_v3.py',
       'trajectory_mae/tensor_corpus.py']
with zipfile.ZipFile(sys.argv[2],'w',zipfile.ZIP_DEFLATED) as archive:
    for name in files:
        path=root/name
        archive.write(path,name)
PY
cp "$PACKAGE_DIR/observation_v3_code.zip" "$PACKAGE_DIR/observation_v3_modules.zip"
# Executor FsShell subprocesses must use the same authenticated account as the
# submitter. Keep secrets out of command arguments and printed dry-run output.
"${PACKAGE_PYTHON:-python3}" - "$PACKAGE_DIR/credentials.properties" <<'PY'
import os, sys
def escape(value):
    return value.replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r')
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as out:
    for name in ('HADOOP_USER_NAME', 'HADOOP_USER_PASSWORD'):
        if name in os.environ:
            for prefix in ('spark.yarn.appMasterEnv.', 'spark.executorEnv.'):
                out.write(prefix + name + '=' + escape(os.environ[name]) + '\n')
PY
read -r -a TRAIN_DAY_ARGS <<< "$TRAIN_DAYS"
read -r -a VAL_DAY_ARGS <<< "$VAL_DAYS"
EXECUTORS="$PARALLELISM"
[[ "$MODE" != preflight ]] || EXECUTORS=1
CMD=("$SPARK_SUBMIT" --master yarn --deploy-mode cluster --queue "$QUEUE"
  --properties-file "$PACKAGE_DIR/credentials.properties"
  --name observationv3 --py-files "$PACKAGE_DIR/observation_v3_modules.zip"
  --conf "spark.yarn.dist.archives=${V3_ENV_ARCHIVE:-hdfs://REQUIRED/v3_env.tar.gz}#v3_env,$SCHEDULER_ENV_ARCHIVE#scheduler_env,$PACKAGE_DIR/observation_v3_code.zip#observation_v3_code"
  --conf "spark.pyspark.python=$SCHEDULER_PYTHON" --conf "spark.pyspark.driver.python=$SCHEDULER_PYTHON"
  --conf spark.dynamicAllocation.enabled=false --num-executors "$EXECUTORS"
  --executor-cores 1 --executor-memory "${EXECUTOR_MEMORY:-2g}"
  --conf "spark.executor.memoryOverhead=${EXECUTOR_OVERHEAD:-6144}"
  --driver-memory "${DRIVER_MEMORY:-4g}" --conf spark.speculation=false
  --conf spark.yarn.appMasterEnv.PYTHONUNBUFFERED=1
  --conf spark.executorEnv.PYTHONUNBUFFERED=1
  --conf spark.executorEnv.OMP_NUM_THREADS=1 --conf spark.executorEnv.MKL_NUM_THREADS=1
  --conf spark.yarn.submit.waitAppCompletion=true
  tools/prepare_observation_v3_yarn.py --source "$SOURCE" --out "$OUT_DIR"
  --train-days "${TRAIN_DAY_ARGS[@]}" --val-days "${VAL_DAY_ARGS[@]}"
  --parallelism "$PARALLELISM" --buckets "$BUCKETS" --m-max "$M_MAX" --seed "$DATA_SEED"
  --worker-python "$V3_PYTHON")
[[ "$MODE" != preflight ]] || CMD+=(--preflight-only)
printf '%q ' "${CMD[@]}"
printf '\n'
if [[ "$MODE" == dry ]]; then
  echo 'Dry run: no submission or conversion.'
else
  "${CMD[@]}"
fi
