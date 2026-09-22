#!/usr/bin/env bash
# Build final training tensors directly from the raw Parquet week in one YARN job.
#
# MODE=dry  prints a fully expanded command and never needs Hadoop credentials.
# MODE=yarn verifies that the destination is fresh, then submits the job.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODE="${MODE:-dry}"
[[ "$MODE" == dry || "$MODE" == yarn ]] || {
  echo "MODE must be dry or yarn (got: $MODE)" >&2
  exit 2
}

TRAIN_DAYS="${TRAIN_DAYS:-20260817 20260818 20260819 20260820 20260821 20260822}"
VAL_DAYS="${VAL_DAYS:-20260823}"
INPUT_DAYS="${INPUT_DAYS:-${TRAIN_DAYS} ${VAL_DAYS}}"
INPUT_BASE="${INPUT_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/liruifeng/traffic_traj_encoder/beijing_week_biz/samples}"
HDFS_BASE="${HDFS_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/junao/target_link}"
if [[ -z "${INPUTS:-}" ]]; then
  # By default read every requested train/validation day.  Override INPUT_DAYS
  # when a temporal-boundary investigation needs adjacent raw event hours.
  INPUTS=""
  for day in $INPUT_DAYS; do
    INPUTS="${INPUTS:+${INPUTS},}${INPUT_BASE}/event_hour=${day}*/part-*.parquet"
  done
fi
BUCKETS="${BUCKETS:-128}"
M_MAX="${M_MAX:-64}"
SEED="${SEED:-20260921}"
SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-1024}"
PARALLELISM="${PARALLELISM:-20}"
OUT_DIR="${OUT_DIR:-${HDFS_BASE}/raw_training_tensors_v1_m${M_MAX}_seed${SEED}}"
QUEUE="${QUEUE:-root.xinsi_yanfaerzu_default}"
SPARK_SUBMIT="${SPARK_SUBMIT:-/usr/local/spark-current/bin/spark-submit}"
HDFS_BIN="${HDFS_BIN:-hdfs}"
TENSOR_ENV_PYTHON="${TENSOR_ENV_PYTHON:-./tensor_env/bin/python}"

[[ "$INPUTS" == hdfs://* ]] || { echo 'INPUTS must be comma-separated hdfs:// Parquet globs' >&2; exit 2; }
[[ "$OUT_DIR" == hdfs://* ]] || { echo 'OUT_DIR must be an hdfs:// URI' >&2; exit 2; }
for value_name in BUCKETS M_MAX SHUFFLE_PARTITIONS PARALLELISM; do
  value="${!value_name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "$value_name must be a positive integer" >&2; exit 2; }
done
(( M_MAX >= 3 )) || { echo 'M_MAX must be at least 3' >&2; exit 2; }
(( BUCKETS <= 128 )) || { echo 'BUCKETS must be at most 128' >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo 'SEED must be a non-negative integer' >&2; exit 2; }
[[ "$TENSOR_ENV_PYTHON" == ./* ]] || {
  echo 'TENSOR_ENV_PYTHON must be a localized relative path (for example ./tensor_env/bin/python)' >&2
  exit 2
}

read -r -a TRAIN_DAY_ARGS <<< "$TRAIN_DAYS"
read -r -a VAL_DAY_ARGS <<< "$VAL_DAYS"
[[ ${#TRAIN_DAY_ARGS[@]} -gt 0 && ${#VAL_DAY_ARGS[@]} -gt 0 ]] || {
  echo 'TRAIN_DAYS and VAL_DAYS must both contain at least one day' >&2; exit 2;
}
declare -A SEEN_DAYS=()
for day in "${TRAIN_DAY_ARGS[@]}"; do
  [[ "$day" =~ ^[0-9]{8}$ ]] || { echo "Invalid training day: $day" >&2; exit 2; }
  [[ -z "${SEEN_DAYS[$day]:-}" ]] || { echo "Duplicate training day: $day" >&2; exit 2; }
  SEEN_DAYS["$day"]=train
done
for day in "${VAL_DAY_ARGS[@]}"; do
  [[ "$day" =~ ^[0-9]{8}$ ]] || { echo "Invalid validation day: $day" >&2; exit 2; }
  [[ -z "${SEEN_DAYS[$day]:-}" ]] || { echo "Train/validation days overlap: $day" >&2; exit 2; }
  SEEN_DAYS["$day"]=val
done
read -r -a INPUT_DAY_ARGS <<< "$INPUT_DAYS"
for day in "${INPUT_DAY_ARGS[@]}"; do
  [[ "$day" =~ ^[0-9]{8}$ ]] || { echo "Invalid input day: $day" >&2; exit 2; }
done

if [[ "$MODE" == yarn ]]; then
  [[ -n "${HADOOP_USER_NAME:-}" ]] || { echo 'HADOOP_USER_NAME not set' >&2; exit 2; }
  [[ -n "${TENSOR_ENV_ARCHIVE:-}" ]] || {
    echo 'YARN submission requires TENSOR_ENV_ARCHIVE containing Python >=3.9, PySpark, NumPy, PyArrow and PyTorch' >&2
    exit 2
  }
  [[ "$TENSOR_ENV_ARCHIVE" == hdfs://* ]] || { echo 'TENSOR_ENV_ARCHIVE must be an hdfs:// archive' >&2; exit 2; }
  [[ -x "$SPARK_SUBMIT" ]] || { echo "spark-submit is not executable: $SPARK_SUBMIT" >&2; exit 2; }
  command -v "$HDFS_BIN" >/dev/null 2>&1 || { echo "HDFS_BIN is not available: $HDFS_BIN" >&2; exit 2; }
  SPARK_VERSION_OUTPUT="$("$SPARK_SUBMIT" --version 2>&1)" || {
    printf '%s\n' "$SPARK_VERSION_OUTPUT" >&2; exit 2;
  }
  if [[ "$SPARK_VERSION_OUTPUT" =~ version[[:space:]]+([0-9]+)\.([0-9]+) ]]; then
    echo "Spark launcher version: ${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
    # The archive is deliberately the PySpark interpreter for both driver and
    # executors.  It must contain a PySpark release compatible with this
    # launcher (and a Java runtime supported by that Spark release); a Python
    # environment built for an unrelated Spark/Java stack is not safe to use.
  else
    echo 'Unable to determine Spark launcher version; refusing unchecked Python/Spark environment pairing.' >&2
    exit 2
  fi
  if "$HDFS_BIN" dfs -test -e "$OUT_DIR"; then
    echo "OUT_DIR already exists; choose a fresh version (refusing overwrite): $OUT_DIR" >&2
    exit 2
  fi
fi

PACKAGE_DIR="$(mktemp -d)"
trap 'rm -rf -- "$PACKAGE_DIR"' EXIT
"${PACKAGE_PYTHON:-python3}" - "$REPO" "$PACKAGE_DIR/raw_tensor_modules.zip" <<'PY'
from pathlib import Path
import sys
import zipfile

root, output = map(Path, sys.argv[1:])
files = [
    'tools/build_raw_training_tensors.py',
    'tools/prepare_tensors_yarn.py',
    'tools/__init__.py',
    'experiments/__init__.py',
    'experiments/trajectory_mlp_v1/__init__.py',
    'experiments/trajectory_mlp_v1/data.py',
    'experiments/trajectory_mlp_v1/prepared.py',
    'experiments/trajectory_mlp_v1/tensor_corpus.py',
]
with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in files:
        path = root / name
        if path.is_file():
            archive.write(path, name)
        elif name.endswith('__init__.py'):
            archive.writestr(name, '')
        else:
            raise FileNotFoundError(path)
PY

DIST_ARCHIVES="${TENSOR_ENV_ARCHIVE:-hdfs://REQUIRED_FOR_YARN/tensor_env.tgz}#tensor_env"
CMD=(
  "$SPARK_SUBMIT"
  --master yarn --deploy-mode cluster
  --queue "$QUEUE"
  --name raw_training_tensors_v1
  --py-files "$PACKAGE_DIR/raw_tensor_modules.zip"
  --conf "spark.yarn.dist.archives=$DIST_ARCHIVES"
  --conf "spark.pyspark.python=$TENSOR_ENV_PYTHON"
  --conf "spark.pyspark.driver.python=$TENSOR_ENV_PYTHON"
  --conf 'spark.dynamicAllocation.enabled=false'
  --num-executors "$PARALLELISM"
  --conf "spark.sql.shuffle.partitions=$SHUFFLE_PARTITIONS"
  --conf 'spark.speculation=false'
  --conf 'spark.yarn.submit.waitAppCompletion=true'
  --driver-memory "${DRIVER_MEMORY:-8g}"
  --executor-cores "${EXECUTOR_CORES:-1}"
  --executor-memory "${EXECUTOR_MEMORY:-8g}"
  --conf "spark.executor.memoryOverhead=${EXECUTOR_OVERHEAD:-4096}"
  tools/build_raw_training_tensors.py
  --inputs "$INPUTS"
  --out "$OUT_DIR"
  --train-days "${TRAIN_DAY_ARGS[@]}"
  --val-days "${VAL_DAY_ARGS[@]}"
  --buckets "$BUCKETS"
  --m-max "$M_MAX"
  --seed "$SEED"
  --shuffle-partitions "$SHUFFLE_PARTITIONS"
)

printf '%q ' "${CMD[@]}"
printf '\n'
if [[ "$MODE" == dry ]]; then
  if [[ -z "${TENSOR_ENV_ARCHIVE:-}" ]]; then
    echo 'Dry run warning: set TENSOR_ENV_ARCHIVE to one Python >=3.9 environment with PySpark, NumPy, PyArrow and PyTorch before MODE=yarn.'
  fi
  echo 'Dry run: command assembled; no HDFS checks, YARN submission, or data conversion.'
else
  "${CMD[@]}"
fi
