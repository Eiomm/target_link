#!/usr/bin/env bash
# Submit tools/build_corpus.py — build the V1 canonical corpus
# (ragged observations + cells + policy-dependent groups) over the whole week.
#
#   MODE=local  one day on this pod over data/raw_hdfs (whatever hours exist)
#   MODE=dry    print the yarn spark-submit command without executing
#   MODE=yarn   real submission (default)
#
#   MODE=yarn bash scripts/submit_build_corpus_yarn.sh
#
# One-part smoke (minutes, validates the ragged arrays against the pyarrow
# reference /tmp/ref_cells.py item-for-item):
#   DAYS="" INPUT_GLOBS='hdfs://.../samples/event_hour=2026082107/part-00000*.parquet' \
#   OUT_DIR=hdfs://.../target_link/corpus_v1_smoke SHUFFLE_PARTITIONS=64 \
#   OBS_PARTITIONS=128 MAX_EXECUTORS=4 MODE=yarn bash scripts/submit_build_corpus_yarn.sh
#
# The week is 7*24 = 168 event_hour partitions, ~1 TB. One job (not per day):
# a day slice systematically undercounts K in the day's first/last window,
# because a pass is filed under the hour of its EVENT while t_seg_enter spreads
# ~15 min either side (measured 0.04% before / 0.05% after the hour), so the
# previous day's 23:00 partition feeds this day's 23:50 window and vice versa.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODE="${MODE:-yarn}"
DAYS="${DAYS:-20260817 20260818 20260819 20260820 20260821 20260822 20260823}"
INPUT_BASE="${INPUT_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/liruifeng/traffic_traj_encoder/beijing_week_biz/samples}"
HDFS_BASE="${HDFS_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/junao/target_link}"
OUT_DIR="${OUT_DIR:-${HDFS_BASE}/corpus_v1}"

if [[ -z "${INPUT_GLOBS:-}" ]]; then
  INPUT_GLOBS=""
  for d in $DAYS; do INPUT_GLOBS="${INPUT_GLOBS:+${INPUT_GLOBS},}${INPUT_BASE}/event_hour=${d}*/part-*.parquet"; done
fi

STAGES="${STAGES:-obs,cells,groups}"
# Rebuilding obs (e.g. to add a ragged array) can go to a side directory: the
# row set is unchanged, so cells/ and existing group tables stay valid and the live
# observations/ keeps serving the training side until the new one is verified.
OBS_DIR="${OBS_DIR:-observations_v2}"
GROUPS_DIR="${GROUPS_DIR:-training_groups_k3}"
SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-12000}"
OBS_PARTITIONS="${OBS_PARTITIONS:-1024}"
BUCKETS="${BUCKETS:-128}"
WINDOW_SECONDS="${WINDOW_SECONDS:-600}"
M_MAX="${M_MAX:-16}"
K_MIN="${K_MIN:-3}"
MAX_RECORDS_PER_FILE="${MAX_RECORDS_PER_FILE:-4000000}"

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

case "$MODE" in
  local)
    exec env -u SPARK_HOME SPARK_LOCAL_DIRS="$SPARK_LOCAL_DIRS" \
        PYSPARK_PYTHON="$QWEN12_PY" "$QWEN12_PY" tools/build_corpus.py \
        --inputs "data/raw_hdfs/event_hour=${DAYS%% *}*/part-*.parquet" \
        --out "${LOCAL_OUT:-data/_corpus/corpus_${DAYS%% *}}" \
        --master "${LOCAL_MASTER:-local[8]}" --driver-memory 6g \
        --shuffle-partitions "${LOCAL_SHUFFLE_PARTITIONS:-64}" \
        --obs-partitions 32 --buckets 16 --stages "$STAGES" \
        --window-seconds "$WINDOW_SECONDS" --m-max "$M_MAX" --k-min "$K_MIN" \
        --groups-dir "$GROUPS_DIR"
    ;;
  dry | yarn)
    : "${HADOOP_USER_NAME:?HADOOP_USER_NAME not set}"
    [[ "$OUT_DIR" == hdfs://* ]] || { echo "OUT_DIR must start with hdfs://" >&2; exit 1; }
    [[ "$INPUT_GLOBS" == hdfs://* ]] || { echo "INPUT_GLOBS must start with hdfs://" >&2; exit 1; }

    CMD=(
      "$SPARK_SUBMIT"
      --master yarn --deploy-mode cluster
      --queue "$QUEUE"
      --name "tl_build_corpus"
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
      --conf "spark.sql.adaptive.enabled=true"
      tools/build_corpus.py
      --inputs "$INPUT_GLOBS"
      --out "$OUT_DIR"
      --obs-dir "$OBS_DIR"
      --groups-dir "$GROUPS_DIR"
      --master yarn
      --stages "$STAGES"
      --shuffle-partitions "$SHUFFLE_PARTITIONS"
      --obs-partitions "$OBS_PARTITIONS"
      --buckets "$BUCKETS"
      --window-seconds "$WINDOW_SECONDS"
      --m-max "$M_MAX"
      --k-min "$K_MIN"
      --max-records-per-file "$MAX_RECORDS_PER_FILE"
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
