#!/usr/bin/env bash
# Submit tools/ingest_spark.py — three tiers, per the team's RPGPT submit pattern.
#
#   MODE=local  small-sample run on this pod (qwen12 + pyspark 3.5.9 + Java 8)
#   MODE=dry    print the yarn spark-submit command without executing
#   MODE=yarn   real submission to the company cluster (default)
#
# Typical production run (one day of raw corridor data, HDFS in / HDFS out):
#   OUT_DIR=hdfs://DClusterNmg3/user/<you>/target_link/processed_spark/day20260820 \
#   DAY=20260820 MODE=yarn bash scripts/submit_ingest_yarn.sh
#
# Multiple days = one job per day (same data contract as the old per-day
# pipeline; downstream split_random merges processed dirs). spark-submit
# returns immediately, so a plain loop submits all days at once:
#   OUT_BASE=hdfs://DClusterNmg3/user/<you>/target_link/processed_spark
#   for D in 20260820 20260821; do
#     OUT_DIR=$OUT_BASE/day$D DAY=$D MODE=yarn bash scripts/submit_ingest_yarn.sh
#   done
#   # check: yarn application -list;  logs: yarn logs -applicationId <id> | tail -50
#
# Enlarge gradually: SAMPLE_FRACTION=0.001 local -> MODE=dry -> MAX_EXECUTORS=4 yarn
# -> full run. All knobs are env-overridable.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODE="${MODE:-yarn}"                       # local | dry | yarn
DAY="${DAY:-20260820}"
HOURS="${HOURS:-*}"                        # '*' = whole day, '07' = one hour
SAMPLE_FRACTION="${SAMPLE_FRACTION:-1.0}"

# --- data locations ----------------------------------------------------------
# Data source: beijing_week_biz (verified 2026-09-07: 7 full days 20260817-23,
# 24h partitions each, ~4.2e8 rows/hour, schema drop-in compatible with
# ingest_spark's RAW_COLS — extra bin_pt_bizstatus, lacks 5 unused n_pts_* cols).
# The beijing_week1 source (17/20/21 full + 22 partial) is the older backfill.
INPUT_BASE="${INPUT_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/liruifeng/traffic_traj_encoder/beijing_week_biz/samples}"
INPUT_GLOB="${INPUT_GLOB:-${INPUT_BASE}/event_hour=${DAY}${HOURS}/part-*.parquet}"
OUT_DIR="${OUT_DIR:-}"

# --- cluster resources -------------------------------------------------------
QUEUE="${QUEUE:-root.xinsi_yanfaerzu_default}"
MAX_EXECUTORS="${MAX_EXECUTORS:-40}"
MIN_EXECUTORS="${MIN_EXECUTORS:-2}"
EXECUTOR_CORES="${EXECUTOR_CORES:-2}"
EXECUTOR_MEMORY="${EXECUTOR_MEMORY:-5g}"
EXECUTOR_OVERHEAD="${EXECUTOR_OVERHEAD:-3072}"
DRIVER_MEMORY="${DRIVER_MEMORY:-8g}"
SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-800}"
BINS_SHARDS="${BINS_SHARDS:-64}"

# --- python env for cluster nodes (no system python3 there) ------------------
# verified 2026-09-07: 85MB, python 3.7 — target_link tools must stay 3.7-syntax-clean
MINIPY3_TGZ="${MINIPY3_TGZ:-hdfs://DClusterNmg3/user/bigdata-dp/common-env/minipy3.tgz}"
PYSPARK_PYTHON_CLUSTER="./minipy3/minipy3/bin/python"
SPARK_SUBMIT="${SPARK_SUBMIT:-/usr/local/spark-current/bin/spark-submit}"

# --- this pod's local env (MODE=local) ----------------------------------------
QWEN12_PY="${QWEN12_PY:-/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12/bin/python}"
SPARK_LOCAL_DIRS="${SPARK_LOCAL_DIRS:-$HOME/sparktmp}"

suffix=""
[[ "$SAMPLE_FRACTION" != "1.0" ]] && suffix="_f${SAMPLE_FRACTION}"

case "$MODE" in
  local)
    exec env -u SPARK_HOME SPARK_LOCAL_DIRS="$SPARK_LOCAL_DIRS" \
        PYSPARK_PYTHON="$QWEN12_PY" "$QWEN12_PY" tools/ingest_spark.py \
        --inputs "data/raw_hdfs/event_hour=${DAY}${HOURS}/part-*.parquet" \
        --out "${OUT_DIR:-data/processed_spark/day${DAY}${HOURS}${suffix}}" \
        --master 'local[8]' --driver-memory 6g \
        --sample-fraction "$SAMPLE_FRACTION" \
        --bins-shards "$BINS_SHARDS" \
        --shuffle-partitions "$SHUFFLE_PARTITIONS"
    ;;
  dry | yarn)
    : "${OUT_DIR:?OUT_DIR is required for yarn (must be the full hdfs:// path)}"
    : "${MINIPY3_TGZ:?MINIPY3_TGZ is required for yarn (hdfs path of minipy3.tgz)}"
    : "${HADOOP_USER_NAME:?export HADOOP_USER_NAME first}"
    [[ "$OUT_DIR" == hdfs://* ]] || { echo "OUT_DIR must start with hdfs:// (got: $OUT_DIR)" >&2; exit 1; }
    [[ "$INPUT_GLOB" == hdfs://* ]] || { echo "INPUT_GLOB must start with hdfs:// (got: $INPUT_GLOB)" >&2; exit 1; }

    CMD=(
      "$SPARK_SUBMIT"
      --master yarn --deploy-mode cluster
      --queue "$QUEUE"
      --name "tl_ingest_${DAY}${HOURS//\*/}"
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
      tools/ingest_spark.py
      --inputs "$INPUT_GLOB"
      --out "$OUT_DIR"
      --master yarn
      --sample-fraction "$SAMPLE_FRACTION"
      --bins-shards "$BINS_SHARDS"
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
