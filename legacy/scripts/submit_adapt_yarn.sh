#!/usr/bin/env bash
# Adapt raw corridor samples -> causal events (tools/adapt_samples_windows.py).
#
#   MODE=yarn   submit to the company cluster (default)
#   MODE=dry    print the spark-submit command only
#   MODE=local  one-hour local run on this pod (pip pyspark, no SPARK_HOME)
#
# Default input = the 24h production slice: the warm-up hour 2026082023 plus
# the 24 partitions of 20260821 from beijing_week_biz. That source is the only
# one with whole days (verified 2026-09-09: 168 partitions, 2026081700 ->
# 2026082323, no gaps). beijing_week1 is NOT usable for a day: it holds
# 2026081700-1720 and 2026082000-2026082201 only. The warm-up hour matters
# because the first anchor's [anchor-600, anchor) lookback falls in it.
#
# The output goes straight to HDFS: submit_windows_yarn.sh requires an hdfs://
# input, and the 25h slice is too large to stage through this pod.
#
# Scale (measured, not guessed): one raw part = 34.8M rows, of which ~12.1M
# survive as seg_mark==1 events after component merging; a partition holds 15
# parts, so 25h is ~4.5e9 events. The v4 smoke (2 parts) produced 22.7M events
# / 511 MB, i.e. ~23 B/row -> this run writes roughly 95-100 GB.
set -euo pipefail
# legacy/scripts/ -> repo root: only these entrypoints moved, tools/ and the
# rest of the code stayed in place (legacy/README.md)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

MODE="${MODE:-yarn}"
DAY="${DAY:-20260821}"
WARMUP_HOUR="${WARMUP_HOUR:-2026082023}"

RAW_BASE="${RAW_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/liruifeng/traffic_traj_encoder/beijing_week_biz/samples}"
HDFS_BASE="${HDFS_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/junao/target_link}"
INPUTS="${INPUTS:-${RAW_BASE}/event_hour=${WARMUP_HOUR}/part-*.parquet,${RAW_BASE}/event_hour=${DAY}*/part-*.parquet}"
OUT="${OUT:-${HDFS_BASE}/events/day${DAY}}"

# the window functions shuffle on (map_version, target_link_id, sample_id);
# 4.5e9 rows need far more than the 800-partition default
SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-8000}"
PARTITIONS="${PARTITIONS:-800}"

# --- cluster resources -------------------------------------------------------
QUEUE="${QUEUE:-root.xinsi_yanfaerzu_default}"
MAX_EXECUTORS="${MAX_EXECUTORS:-40}"
MIN_EXECUTORS="${MIN_EXECUTORS:-2}"
EXECUTOR_CORES="${EXECUTOR_CORES:-2}"
EXECUTOR_MEMORY="${EXECUTOR_MEMORY:-8g}"
EXECUTOR_OVERHEAD="${EXECUTOR_OVERHEAD:-4096}"
DRIVER_MEMORY="${DRIVER_MEMORY:-12g}"

# --- python env for cluster nodes (no system python3 there) ------------------
MINIPY3_TGZ="${MINIPY3_TGZ:-hdfs://DClusterNmg3/user/bigdata-dp/common-env/minipy3.tgz}"
SPARK_SUBMIT="${SPARK_SUBMIT:-/usr/local/spark-current/bin/spark-submit}"

# --- this pod's local env (MODE=local) ----------------------------------------
export HADOOP_USER_NAME="${HADOOP_USER_NAME:-bigdata-dp}"
QWEN12_PY="${QWEN12_PY:-/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12/bin/python}"
SPARK_LOCAL_DIRS="${SPARK_LOCAL_DIRS:-/nfs/dataset-ofs-494-1/project/user/junao/sparktmp}"

case "$MODE" in
  local)
    exec env -u SPARK_HOME SPARK_LOCAL_DIRS="$SPARK_LOCAL_DIRS" \
        PYSPARK_PYTHON="$QWEN12_PY" \
        PYSPARK_SUBMIT_ARGS="--driver-memory ${LOCAL_DRIVER_MEMORY:-6g} pyspark-shell" \
        "$QWEN12_PY" tools/adapt_samples_windows.py \
        --inputs "${LOCAL_INPUTS:-data/raw_hdfs/event_hour=${WARMUP_HOUR}/part-*.parquet,data/raw_hdfs/event_hour=${DAY}*/part-*.parquet}" \
        --out "${LOCAL_OUT:-data/windows_adapt_0909/events_${DAY}}" \
        --master "${LOCAL_MASTER:-local[4]}" \
        --partitions "${LOCAL_PARTITIONS:-8}" \
        --shuffle-partitions "${LOCAL_SHUFFLE_PARTITIONS:-32}"
    ;;
  dry | yarn)
    : "${HADOOP_USER_NAME:?export HADOOP_USER_NAME before submission}"
    [[ "$OUT" == hdfs://* ]] || { echo "OUT must start with hdfs:// (got: $OUT)" >&2; exit 1; }
    [[ "$INPUTS" == hdfs://* ]] || { echo "INPUTS must start with hdfs:// (got: $INPUTS)" >&2; exit 1; }

    CMD=(
      "$SPARK_SUBMIT"
      --master yarn --deploy-mode cluster
      --queue "$QUEUE"
      --name "tl_adapt_${DAY}"
      --conf "spark.yarn.dist.archives=${MINIPY3_TGZ}#minipy3"
      --conf "spark.pyspark.python=./minipy3/minipy3/bin/python"
      --driver-memory "$DRIVER_MEMORY"
      --executor-cores "$EXECUTOR_CORES"
      --executor-memory "$EXECUTOR_MEMORY"
      --conf "spark.executor.memoryOverhead=${EXECUTOR_OVERHEAD}"
      --conf "spark.dynamicAllocation.enabled=true"
      --conf "spark.dynamicAllocation.minExecutors=${MIN_EXECUTORS}"
      --conf "spark.dynamicAllocation.maxExecutors=${MAX_EXECUTORS}"
      --conf "spark.serializer=org.apache.spark.serializer.KryoSerializer"
      tools/adapt_samples_windows.py
      --inputs "$INPUTS"
      --out "$OUT"
      --master yarn
      --partitions "$PARTITIONS"
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
