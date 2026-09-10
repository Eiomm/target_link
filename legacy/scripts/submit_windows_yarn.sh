#!/usr/bin/env bash
# New causal-window pipeline. MODE=dry prints only; no DAY requirement.
set -euo pipefail
# legacy/scripts/ -> repo root: only these entrypoints moved, tools/ and the
# rest of the code stayed in place (legacy/README.md)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
MODE="${MODE:-dry}"
: "${INPUT_GLOB:?comma-separated input Parquet paths required}"
: "${OUT_DIR:?new output version directory required}"
: "${ANCHOR_START:?inclusive UTC epoch seconds required}"
: "${ANCHOR_END:?exclusive UTC epoch seconds required}"
args=(tools/build_windows_spark.py --inputs "$INPUT_GLOB" --out "$OUT_DIR"
  --anchor-start "$ANCHOR_START" --anchor-end "$ANCHOR_END"
  --lookback-seconds "${LOOKBACK_SECONDS:-600}" --stride-seconds "${STRIDE_SECONDS:-600}"
  --max-passes "${MAX_PASSES:-0}" --seed "${CAP_SEED:-42}"
  --availability-column "${AVAILABILITY_COLUMN:-available_ts}"
  --position-column "${POSITION_COLUMN:-spatial_start_m}"
  --time-source "${TIME_SOURCE:-explicit}" --sub-length-m "${SUB_LENGTH_M:-200}"
  --max-bins "${MAX_BINS:-21}" --partitions "${CURVES_PARTITIONS:-200}"
  --shuffle-partitions "${SHUFFLE_PARTITIONS:-800}")
[[ -n "${LINKS:-}" ]] && args+=(--links "$LINKS")
case "$MODE" in
  local)
    # same three-tier conventions as submit_curves_yarn.sh: qwen12 python when
    # present, SPARK_HOME unset (pip pyspark), scratch dirs off the root disk,
    # and an explicit local-JVM heap (default 1g GC-thrashes on this pipeline)
    if [ -z "${PYTHON:-}" ] && [ -x /nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12/bin/python ]; then
      PYTHON=/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12/bin/python
    fi
    PYTHON="${PYTHON:-python3}"
    exec env -u SPARK_HOME \
        SPARK_LOCAL_DIRS="${SPARK_LOCAL_DIRS:-/nfs/dataset-ofs-494-1/project/user/junao/sparktmp}" \
        PYSPARK_PYTHON="$PYTHON" \
        PYSPARK_SUBMIT_ARGS="--driver-memory ${LOCAL_DRIVER_MEMORY:-6g} pyspark-shell" \
        "$PYTHON" "${args[@]}" --master "${LOCAL_MASTER:-local[2]}"
    ;;
  dry|yarn)
    [[ "$OUT_DIR" == hdfs://* && "$INPUT_GLOB" == hdfs://* ]] || {
      echo "YARN input/output must be hdfs:// paths" >&2; exit 1;
    }
    cmd=("${SPARK_SUBMIT:-/usr/local/spark-current/bin/spark-submit}"
      --master yarn --deploy-mode cluster --name "tl_windows_${ANCHOR_START}_${ANCHOR_END}"
      --queue "${QUEUE:-root.xinsi_yanfaerzu_default}"
      --conf "spark.yarn.dist.archives=${MINIPY3_TGZ:-hdfs://DClusterNmg3/user/bigdata-dp/common-env/minipy3.tgz}#minipy3"
      --conf spark.pyspark.python=./minipy3/minipy3/bin/python
      --driver-memory "${DRIVER_MEMORY:-12g}" --executor-memory "${EXECUTOR_MEMORY:-8g}"
      --executor-cores "${EXECUTOR_CORES:-2}"
      --conf "spark.executor.memoryOverhead=${EXECUTOR_OVERHEAD:-4096}"
      --conf spark.dynamicAllocation.enabled=true
      --conf "spark.dynamicAllocation.maxExecutors=${MAX_EXECUTORS:-40}"
      --conf "spark.dynamicAllocation.minExecutors=${MIN_EXECUTORS:-2}"
      --conf spark.serializer=org.apache.spark.serializer.KryoSerializer
      "${args[@]}" --master yarn)
    printf '%q ' "${cmd[@]}"
    printf '\n'
    [[ "$MODE" == dry ]] && exit 0
    : "${HADOOP_USER_NAME:?export HADOOP_USER_NAME before submission}"
    exec "${cmd[@]}"
    ;;
  *) echo "MODE must be local, dry, or yarn" >&2; exit 1 ;;
esac
