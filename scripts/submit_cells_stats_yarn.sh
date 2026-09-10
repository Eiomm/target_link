#!/usr/bin/env bash
# Submit tools/stats_cells.py — per (link, seg, 10-min window) trajectory-count
# census over the raw corridor table, under the NEW spec (md/最新讨论想法.md):
# cell = (map_version, target_link_id, seg_idx, window), window from
# t_seg_enter = t_ref + (T_cum - T_diff) at the segment's first bin.
#
#   MODE=local  single-day run on this pod over data/raw_hdfs (whatever hours exist)
#   MODE=dry    print the yarn spark-submit command without executing
#   MODE=yarn   real submission over the full day (default)
#
#   DAY=20260821 MODE=yarn bash scripts/submit_cells_stats_yarn.sh
#
# Smoke (one raw part, ~35M rows, minutes):
#   INPUT_GLOB='hdfs://.../beijing_week_biz/samples/event_hour=2026082107/part-00000*.parquet' \
#   OUT_DIR=hdfs://.../target_link/cells_stats/smoke_1part MAX_EXECUTORS=4 \
#   MODE=yarn bash scripts/submit_cells_stats_yarn.sh
#
# The day slice is the 24 event_hour partitions of DAY (no warm-up hour: the
# cell key needs no lookback across partitions). Scale, measured from the raw
# table: one part = 34.8M rows, 15 parts per hour, so ~12.5e9 raw rows/day, of
# which ~4.4e9 are seg_mark==1 rows; they collapse to ~1.7e8 observations.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODE="${MODE:-yarn}"
DAY="${DAY:-20260821}"
INPUT_BASE="${INPUT_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/liruifeng/traffic_traj_encoder/beijing_week_biz/samples}"
INPUT_GLOB="${INPUT_GLOB:-${INPUT_BASE}/event_hour=${DAY}*/part-*.parquet}"
HDFS_BASE="${HDFS_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/junao/target_link}"
OUT_DIR="${OUT_DIR:-${HDFS_BASE}/cells_stats/day${DAY}}"

# first groupBy keys on (link, sample_id, seg_idx) over ~4.4e9 marked rows
SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-4000}"
M_LIST="${M_LIST:-1,2,4,8,16,32,64,128,256,512}"
WINDOW_SECONDS="${WINDOW_SECONDS:-600}"
WRITE_CELLS="${WRITE_CELLS:-1}"     # 0 => --no-cells (histogram/coverage only)

QUEUE="${QUEUE:-root.xinsi_yanfaerzu_default}"
MAX_EXECUTORS="${MAX_EXECUTORS:-40}"
MIN_EXECUTORS="${MIN_EXECUTORS:-2}"
EXECUTOR_CORES="${EXECUTOR_CORES:-2}"
EXECUTOR_MEMORY="${EXECUTOR_MEMORY:-8g}"
EXECUTOR_OVERHEAD="${EXECUTOR_OVERHEAD:-4096}"
DRIVER_MEMORY="${DRIVER_MEMORY:-12g}"

MINIPY3_TGZ="${MINIPY3_TGZ:-hdfs://DClusterNmg3/user/bigdata-dp/common-env/minipy3.tgz}"
PYSPARK_PYTHON_CLUSTER="./minipy3/minipy3/bin/python"
SPARK_SUBMIT="${SPARK_SUBMIT:-/usr/local/spark-current/bin/spark-submit}"

QWEN12_PY="${QWEN12_PY:-/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12/bin/python}"
SPARK_LOCAL_DIRS="${SPARK_LOCAL_DIRS:-/nfs/dataset-ofs-494-1/project/user/junao/sparktmp}"

CELLS_ARG=()
[[ "$WRITE_CELLS" == "0" ]] && CELLS_ARG=(--no-cells)

case "$MODE" in
  local)
    exec env -u SPARK_HOME SPARK_LOCAL_DIRS="$SPARK_LOCAL_DIRS" \
        PYSPARK_PYTHON="$QWEN12_PY" "$QWEN12_PY" tools/stats_cells.py \
        --inputs "data/raw_hdfs/event_hour=${DAY}*/part-*.parquet" \
        --out "${LOCAL_OUT:-data/_stats/cells_${DAY}}" \
        --master "${LOCAL_MASTER:-local[8]}" --driver-memory 6g \
        --shuffle-partitions "${LOCAL_SHUFFLE_PARTITIONS:-64}" \
        --m-list "$M_LIST" --window-seconds "$WINDOW_SECONDS" "${CELLS_ARG[@]}"
    ;;
  dry | yarn)
    : "${HADOOP_USER_NAME:?HADOOP_USER_NAME not set}"
    [[ "$OUT_DIR" == hdfs://* ]] || { echo "OUT_DIR must start with hdfs://" >&2; exit 1; }
    [[ "$INPUT_GLOB" == hdfs://* ]] || { echo "INPUT_GLOB must start with hdfs://" >&2; exit 1; }

    CMD=(
      "$SPARK_SUBMIT"
      --master yarn --deploy-mode cluster
      --queue "$QUEUE"
      --name "tl_cells_day${DAY}"
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
      tools/stats_cells.py
      --inputs "$INPUT_GLOB"
      --out "$OUT_DIR"
      --master yarn
      --shuffle-partitions "$SHUFFLE_PARTITIONS"
      --m-list "$M_LIST"
      --window-seconds "$WINDOW_SECONDS"
      "${CELLS_ARG[@]}"
    )

    echo "# command to run:"
    printf '  %q' "${CMD[@]}"; echo
    [[ "$MODE" == "dry" ]] && { echo "# DRY_RUN: not executing"; exit 0; }
    exec "${CMD[@]}"
    ;;
  *)
    echo "MODE must be local | dry | yarn (got: $MODE)" >&2; exit 1
    ;;
esac
