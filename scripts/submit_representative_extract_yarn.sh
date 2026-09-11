#!/usr/bin/env bash
# One-command representative export for local visual analysis.
# Run: MODE=yarn bash scripts/submit_representative_extract_yarn.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODE="${MODE:-dry}"
HDFS_BASE="${HDFS_BASE:-hdfs://DClusterNmg3/user/bigdata-dp/user/junao/target_link}"
CORPUS="${CORPUS:-$HDFS_BASE/corpus_v1}"
OBS_DIR="${OBS_DIR:-observations_v2}"
DAYS="${DAYS:-20260817,20260818,20260819,20260820,20260821,20260822,20260823}"
N_LINKS="${N_LINKS:-48}"
OUT_DIR="${OUT_DIR:-$HDFS_BASE/representative_extract_v1_${N_LINKS}}"
NFS_DIR="${NFS_DIR:-$REPO/runtime/representative_extract_v1_${N_LINKS}}"
DETAIL_SLOTS="${DETAIL_SLOTS:-1,4}"
MAX_TRAJ="${MAX_TRAJ:-32}"
MIN_OBS="${MIN_OBS:-100}"
MIN_ACTIVE_DAYS="${MIN_ACTIVE_DAYS:-7}"
MIN_HOUR_SLOTS="${MIN_HOUR_SLOTS:-24}"
MIN_ACTIVE_HOURS="${MIN_ACTIVE_HOURS:-84}"
SEED="${SEED:-20260911}"
SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-2000}"
OVERWRITE="${OVERWRITE:-0}"

QUEUE="${QUEUE:-root.xinsi_yanfaerzu_default}"
MAX_EXECUTORS="${MAX_EXECUTORS:-40}"
MIN_EXECUTORS="${MIN_EXECUTORS:-2}"
EXECUTOR_CORES="${EXECUTOR_CORES:-2}"
EXECUTOR_MEMORY="${EXECUTOR_MEMORY:-8g}"
DRIVER_MEMORY="${DRIVER_MEMORY:-12g}"
MINIPY3_TGZ="${MINIPY3_TGZ:-hdfs://DClusterNmg3/user/bigdata-dp/common-env/minipy3.tgz}"
SPARK_SUBMIT="${SPARK_SUBMIT:-/usr/local/spark-current/bin/spark-submit}"

CMD=(
  "$SPARK_SUBMIT" --master yarn --deploy-mode cluster --queue "$QUEUE"
  --name "tl_representative_extract_${N_LINKS}"
  --conf "spark.yarn.dist.archives=${MINIPY3_TGZ}#minipy3"
  --conf "spark.pyspark.python=./minipy3/minipy3/bin/python"
  --driver-memory "$DRIVER_MEMORY"
  --executor-cores "$EXECUTOR_CORES" --executor-memory "$EXECUTOR_MEMORY"
  --conf "spark.dynamicAllocation.enabled=true"
  --conf "spark.dynamicAllocation.minExecutors=${MIN_EXECUTORS}"
  --conf "spark.dynamicAllocation.maxExecutors=${MAX_EXECUTORS}"
  --conf "spark.serializer=org.apache.spark.serializer.KryoSerializer"
  tools/extract_representative_cells.py
  --corpus "$CORPUS" --obs-dir "$OBS_DIR" --out "$OUT_DIR"
  --days "$DAYS" --n-links "$N_LINKS"
  --detail-slots "$DETAIL_SLOTS" --max-trajectories-per-cell "$MAX_TRAJ"
  --min-observations "$MIN_OBS" --min-active-days "$MIN_ACTIVE_DAYS"
  --min-hour-slots "$MIN_HOUR_SLOTS" --min-active-hours "$MIN_ACTIVE_HOURS"
  --seed "$SEED" --shuffle-partitions "$SHUFFLE_PARTITIONS" --master yarn
)
[[ "$OVERWRITE" == "1" ]] && CMD+=(--overwrite)

printf '# command:'; printf ' %q' "${CMD[@]}"; echo
echo "# HDFS output: $OUT_DIR"
echo "# NFS output: $NFS_DIR"
[[ "$MODE" == "dry" ]] && exit 0
[[ "$MODE" == "yarn" ]] || { echo "MODE must be dry or yarn" >&2; exit 2; }
if [[ -e "$NFS_DIR" ]]; then
  echo "NFS target already exists: $NFS_DIR" >&2
  echo "set NFS_DIR to a new path, or move the existing directory" >&2
  exit 3
fi
export HADOOP_USER_NAME="${HADOOP_USER_NAME:-bigdata-dp}"
"${CMD[@]}"

hdfs dfs -get "$OUT_DIR" "$NFS_DIR"
echo "# done: $NFS_DIR"
du -sh "$NFS_DIR"
