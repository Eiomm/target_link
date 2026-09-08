#!/usr/bin/env bash
# Submit tools/build_curves_spark.py (v2.1 merged pipeline: raw -> curve shards
# in ONE pure-SQL job — no executor python needed, minipy3 is enough).
# Three tiers, same pattern/env as scripts/submit_ingest_yarn.sh.
#
#   MODE=local  small-sample run on this pod (qwen12 + pyspark 3.5.9)
#   MODE=dry    print the yarn spark-submit command without executing
#   MODE=yarn   real submission (default)
#
#   OUT_DIR=hdfs://DClusterNmg3/user/bigdata-dp/user/junao/target_link/curves_spark/day20260817 \
#   DAY=20260817 MODE=yarn bash scripts/submit_curves_yarn.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODE="${MODE:-yarn}"
DAY="${DAY:?set DAY=YYYYMMDD}"
HOURS="${HOURS:-*}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-1.0}"

INPUT_BASE="${INPUT_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/liruifeng/traffic_traj_encoder/beijing_week_biz/samples}"
INPUT_GLOB="${INPUT_GLOB:-${INPUT_BASE}/event_hour=${DAY}${HOURS}/part-*.parquet}"
OUT_DIR="${OUT_DIR:-}"

QUEUE="${QUEUE:-root.xinsi_yanfaerzu_default}"
MAX_EXECUTORS="${MAX_EXECUTORS:-40}"
MIN_EXECUTORS="${MIN_EXECUTORS:-2}"
EXECUTOR_CORES="${EXECUTOR_CORES:-2}"
EXECUTOR_MEMORY="${EXECUTOR_MEMORY:-8g}"
EXECUTOR_OVERHEAD="${EXECUTOR_OVERHEAD:-4096}"
DRIVER_MEMORY="${DRIVER_MEMORY:-12g}"
SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-800}"
CURVES_PARTITIONS="${CURVES_PARTITIONS:-200}"

MINIPY3_TGZ="${MINIPY3_TGZ:-hdfs://DClusterNmg3/user/bigdata-dp/common-env/minipy3.tgz}"
PYSPARK_PYTHON_CLUSTER="./minipy3/minipy3/bin/python"
SPARK_SUBMIT="${SPARK_SUBMIT:-/usr/local/spark-current/bin/spark-submit}"

QWEN12_PY="${QWEN12_PY:-/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12/bin/python}"
SPARK_LOCAL_DIRS="${SPARK_LOCAL_DIRS:-/nfs/dataset-ofs-494-1/project/user/junao/sparktmp}"

suffix=""
[[ "$SAMPLE_FRACTION" != "1.0" ]] && suffix="_f${SAMPLE_FRACTION}"

case "$MODE" in
  local)
    exec env -u SPARK_HOME SPARK_LOCAL_DIRS="$SPARK_LOCAL_DIRS" \
        PYSPARK_PYTHON="$QWEN12_PY" "$QWEN12_PY" tools/build_curves_spark.py \
        --inputs "data/raw_hdfs/event_hour=${DAY}${HOURS}/part-*.parquet" \
        --out "${OUT_DIR:-data/curves_spark/day${DAY}${HOURS}${suffix}}" \
        --master 'local[8]' --driver-memory 6g \
        --sample-fraction "$SAMPLE_FRACTION" \
        --curves-partitions 8 --shuffle-partitions 32
    ;;
  dry | yarn)
    : "${OUT_DIR:?OUT_DIR is required for yarn (full hdfs:// path)}"
    : "${HADOOP_USER_NAME:?HADOOP_USER_NAME not set}"
    [[ "$OUT_DIR" == hdfs://* ]] || { echo "OUT_DIR must start with hdfs://" >&2; exit 1; }
    [[ "$INPUT_GLOB" == hdfs://* ]] || { echo "INPUT_GLOB must start with hdfs://" >&2; exit 1; }

    CMD=(
      "$SPARK_SUBMIT"
      --master yarn --deploy-mode cluster
      --queue "$QUEUE"
      --name "tl_curves_${DAY}${HOURS//\*/}"
      --conf "spark.yarn.dist.archives=${MINIPY3_TGZ}#minipy3"
      --conf "spark.pyspark.python=${PYSPARK_PYTHON_CLUSTER}"
      --driver-memory "$DRIVER_MEMORY"
      --executor-cores "$EXECUTOR_CORES"
      --executor-memory "$EXECUTOR_MEMORY"
      --conf "spark.executor.memoryOverhead=${EXECUTOR_OVERHEAD}"
      --conf "spark.dynamicAllocation.enabled=true"
      --conf "spark.dynamicAllocation.minExecutors=${MIN_EXECUTORS}"
      --conf "spark.dynamicAllocation.maxExecutors=${MAX_EXECUTORS}"
      --conf "spark.serializer=org.apache.spark.serializer.KryoSerializer"
      tools/build_curves_spark.py
      --inputs "$INPUT_GLOB"
      --out "$OUT_DIR"
      --master yarn
      --sample-fraction "$SAMPLE_FRACTION"
      --curves-partitions "$CURVES_PARTITIONS"
      --shuffle-partitions "$SHUFFLE_PARTITIONS"
    )
    echo "# command to run:"
    printf '  %q' "${CMD[@]}"; echo
    [[ "$MODE" == dry ]] && { echo "# DRY_RUN: not executing"; exit 0; }
    exec "${CMD[@]}"
    ;;
  *)
    echo "MODE must be local | dry | yarn (got: $MODE)" >&2; exit 1
    ;;
esac
